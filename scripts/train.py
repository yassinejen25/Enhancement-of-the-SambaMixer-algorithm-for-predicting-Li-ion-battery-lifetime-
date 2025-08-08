import os
os.environ["USE_TRITON"] = "0"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["USE_FLASH_ATTENTION"] = "0"

import logging
import warnings
from pathlib import Path

import hydra
import pytorch_lightning as pl
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint, ModelSummary
from pytorch_lightning.loggers import TensorBoardLogger

from samba_mixer_main.datasets.nasa_battery_dataset import NasaBatteryDataModuleFactory
from samba_mixer_main.datasets.oxford_battery_dataset import OxfordBatteryDataModule
from samba_mixer_main.model.samba import SambaPredictor
from samba_mixer_main.utils.omega_conf_resolver import register_new_resolver

warnings.filterwarnings("ignore")

register_new_resolver()

pl.seed_everything(42)
log = logging.getLogger(__name__)


@hydra.main(version_base="1.3", config_path="../configs", config_name="train_config")
def main(cfg: DictConfig) -> None:
    log.debug(f"Config:\n{OmegaConf.to_yaml(cfg)}")

    output_dir = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)
    log.info(f"Output directory: {output_dir}")

    # Choix dynamique du DataModule selon la config dataset
    dataset_target = cfg.dataset._target_
    if dataset_target.endswith("OxfordBatteryDataModule"):
        data_module = OxfordBatteryDataModule(
            mat_file_path=cfg.dataset.mat_file_path,
            batch_size=cfg.dataset.batch_size,
            seq_len=cfg.dataset.seq_len,
            num_workers=cfg.dataset.num_workers,
        )
    else:
        # Par défaut, NASA
        data_module = NasaBatteryDataModuleFactory.get_nasa_battery_data_module(dataset_config=cfg.dataset)

    model = SambaPredictor(
        model_config=cfg.model,
        trainer_config=cfg.trainer,
        data_module=data_module,
    )

    checkpoint_callback = ModelCheckpoint(
        dirpath=output_dir / "checkpoints",
        filename="best-checkpoint",
        save_top_k=1,
        verbose=True,
        monitor="val_loss",
        mode="min",
    )

    logger = TensorBoardLogger(save_dir=output_dir / "tensorboard", name=None, version=None)
    if cfg.log_graph:
        logger.experiment.add_graph(
            model.model.to(device="cpu"),
            input_to_model=data_module.get_dummy_data_batch(to_device="cpu"),
        )

    trainer = pl.Trainer(
        logger=logger,
        reload_dataloaders_every_n_epochs=cfg.dataset.get("dataloader_reload_period", 1),
        callbacks=[
            checkpoint_callback,
            ModelSummary(max_depth=2),
            LearningRateMonitor(logging_interval="epoch"),
        ],
        max_epochs=cfg.trainer.epochs,
        accelerator="cpu",  # adapte si GPU disponible
        devices=1,
        precision=cfg.trainer.get("precision", 32),  # correction de typo precision
    )

    trainer.fit(model, data_module)
    metrics = trainer.test(ckpt_path="best", dataloaders=data_module.test_dataloader())

    logger.log_hyperparams(cfg, metrics=metrics[0])

    if not cfg.keep_checkpoint:
        os.remove(output_dir / "checkpoints" / "best-checkpoint.ckpt")


if __name__ == "__main__":
    main()
