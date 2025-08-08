import torch
import torch.nn.functional as F
from einops import rearrange, repeat

def causal_conv1d_fn(x, weight, bias=None, activation="silu", groups=1):
    padding = weight.shape[-1] - 1
    out = F.conv1d(x, weight.unsqueeze(1), bias=bias, padding=padding, groups=groups)
    if activation == "silu":
        out = F.silu(out)
    return out

def selective_scan_ref(u, delta, A, B, C, D=None, z=None, delta_bias=None, delta_softplus=False):
    dtype_in = u.dtype
    u = u.float()
    delta = delta.float()

    # Debug : afficher les tailles avant correction
    print(f"Avant correction: u.shape={u.shape}, delta.shape={delta.shape}, B.shape={B.shape}")

    # Forcer la même taille L sur tous
    min_L = min(u.shape[-1], delta.shape[-1], B.shape[-1])
    if u.shape[-1] != min_L or delta.shape[-1] != min_L or B.shape[-1] != min_L:
        print(f"[Correction] Ajustement des tailles à {min_L}")
        u = u[..., :min_L]
        delta = delta[..., :min_L]
        B = B[..., :min_L]

    # Si C existe, s'assurer aussi qu'il a la bonne taille
    if C is not None and C.dim() >= 3:
        if C.shape[-1] != min_L:
            print(f"[Correction] Tronque C de {C.shape[-1]} à {min_L}")
            C = C[..., :min_L]

    if delta_bias is not None:
        delta = delta + delta_bias[..., None].float()
    if delta_softplus:
        delta = F.softplus(delta)

    batch, dim, dstate = u.shape[0], A.shape[0], A.shape[1]
    is_variable_B = B.dim() >= 3
    is_variable_C = C.dim() >= 3 if C is not None else False

    if A.is_complex():
        if is_variable_B:
            B = torch.view_as_complex(rearrange(B.float(), "... (L two) -> ... L two", two=2))
        if is_variable_C:
            C = torch.view_as_complex(rearrange(C.float(), "... (L two) -> ... L two", two=2))
    else:
        B = B.float()
        if C is not None:
            C = C.float()

    x = A.new_zeros((batch, dim, dstate))
    ys = []

    deltaA = torch.exp(torch.einsum('bdl,dn->bdln', delta, A))

    if not is_variable_B:
        deltaB_u = torch.einsum('bdl,dn,bdl->bdln', delta, B, u)
    else:
        if B.dim() == 3:
            deltaB_u = torch.einsum('bdl,bnl,bdl->bdln', delta, B, u)
        else:
            B = repeat(B, "B G N L -> B (G H) N L", H=dim // B.shape[1])
            deltaB_u = torch.einsum('bdl,bdnl,bdl->bdln', delta, B, u)

    if is_variable_C and C.dim() == 4:
        C = repeat(C, "B G N L -> B (G H) N L", H=dim // C.shape[1])

    for i in range(u.shape[2]):
        x = deltaA[:, :, i] * x + deltaB_u[:, :, i]
        if not is_variable_C:
            y = torch.einsum('bdn,dn->bd', x, C)
        else:
            if C.dim() == 3:
                y = torch.einsum('bdn,bn->bd', x, C[:, :, i])
            else:
                y = torch.einsum('bdn,bdn->bd', x, C[:, :, :, i])
        if y.is_complex():
            y = y.real * 2
        ys.append(y)

    y = torch.stack(ys, dim=2)
    out = y if D is None else y + u * rearrange(D, "d -> d 1")
    if z is not None:
        out = out * F.silu(z)
    out = out.to(dtype=dtype_in)
    return out

def mamba_inner_fn_no_out_proj(
    xz,
    conv1d_weight,
    conv1d_bias,
    x_proj_weight,
    delta_proj_weight,
    A,
    B=None,
    C=None,
    D=None,
    delta_bias=None,
    B_proj_bias=None,
    C_proj_bias=None,
    delta_softplus=True,
):
    L = xz.shape[-1]
    delta_rank = delta_proj_weight.shape[1]
    d_state = A.shape[-1] * (2 if A.is_complex() else 1)

    x, z = xz.chunk(2, dim=1)

    total_seq_len = x.shape[-1]
    if total_seq_len % 8 != 0:
        usable_L = (total_seq_len // 8) * 8
        print(f"[Avertissement] Séquence tronquée de {total_seq_len} à {usable_L}")
        x = x[:, :, :usable_L]
        z = z[:, :, :usable_L]
        L = usable_L

    conv1d_out = causal_conv1d_fn(
        x, rearrange(conv1d_weight, "d 1 w -> d w"), conv1d_bias, groups=x.shape[1]
    )

    x_dbl = F.linear(rearrange(conv1d_out, 'b d l -> (b l) d'), x_proj_weight)

    tmp = delta_proj_weight @ x_dbl[:, :delta_rank].t()
    d = delta_proj_weight.shape[0]
    actual_elements = tmp.shape[1]

    if actual_elements % L != 0:
        usable_elements = (actual_elements // L) * L
        print(f"[Avertissement] Troncature automatique: {actual_elements - usable_elements} éléments ignorés.")
        tmp = tmp[:, :usable_elements]
        x_dbl = x_dbl[:usable_elements, :]
        conv1d_out = conv1d_out[:, :, :L * (usable_elements // L)]
        if z is not None:
            z = z[:, :, :L * (usable_elements // L)]

    batch = tmp.shape[1] // L
    tmp = tmp.contiguous().view(d, batch, L)
    delta = tmp.permute(1, 0, 2).contiguous()

    if B is None:
        B = x_dbl[:, delta_rank:delta_rank + d_state]
        if B_proj_bias is not None:
            B = B + B_proj_bias.to(dtype=B.dtype)
        if not A.is_complex():
            B = rearrange(B, "(b l) dstate -> b 1 dstate l", l=L).contiguous()
        else:
            B = rearrange(B, "(b l) (dstate two) -> b 1 dstate (l two)", l=L, two=2).contiguous()

    if C is None:
        C = x_dbl[:, -d_state:]
        if C_proj_bias is not None:
            C = C + C_proj_bias.to(dtype=C.dtype)
        if not A.is_complex():
            C = rearrange(C, "(b l) dstate -> b 1 dstate l", l=L).contiguous()
        else:
            C = rearrange(C, "(b l) (dstate two) -> b 1 dstate (l two)", l=L, two=2).contiguous()

    out = selective_scan_ref(conv1d_out, delta, A, B, C, D, z, delta_bias, delta_softplus)
    return out


# Exemple simplifié de boucle d'entraînement avec monitoring
def training_loop(model, dataloader, optimizer, criterion, device, num_epochs=5):
    model.to(device)
    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs} démarre")
        model.train()
        for batch_idx, batch in enumerate(dataloader):
            inputs, targets = batch
            inputs = inputs.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            if batch_idx % 10 == 0:
                print(f"Epoch {epoch+1}, batch {batch_idx}/{len(dataloader)}, loss: {loss.item():.4f}")

        print(f"Epoch {epoch+1} terminée")
    print("Entraînement terminé !")
