import argparse
import datetime
import os
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import scipy.io as sio
import yaml
from scipy import integrate
from tqdm import tqdm

from samba_mixer.dataset.preprocessing import drop_selected_cycles, drop_soh_outlier, filter_discharge_disconnected_load, pad_timesignals_discharge

parser = argparse.ArgumentParser(description="Convert Discharge Cycles of Nasa Battery Dataset")
parser.add_argument("-f","--filter", action="store_true", help="Applies filters as determined by dataset analysis.")
parser.add_argument("-p","--pad", action="store_true", help="Pads the time series signals.")
args = parser.parse_args()

# Utiliser un chemin relatif ou configurable pour Windows
BASE_DIR = Path(__file__).resolve().parents[2]  # Remonte à samba-mixer-main
DATASET_BASE_PATH = BASE_DIR / "third_party_packages" / "datasets" / "nasa_batteries_orig"


output_dir_name = "nasa_batteries_preprocessed_discharge"
if args.filter:
    output_dir_name += "_filtered"
if args.pad:
    output_dir_name += "_padded"
OUTPUT_PATH = BASE_DIR / "datasets" / output_dir_name


def get_discharge_cycle_df(cycles: np.ndarray, battery_id: str) -> pd.DataFrame:
    cycle_data = []
    cycles_charge = cycles[cycles["type"] == "charge"]
    cycles_discharge = cycles[cycles["type"] == "discharge"]

    for cycle_charge_id, cycle_charge_data in enumerate(cycles_charge):
        time_cycle_start = datetime.datetime(*cycle_charge_data["time"][0].astype(int))
        ambient_temperature = cycle_charge_data["ambient_temperature"][0][0]
        current_measured = (cycle_charge_data["data"]["Current_measured"][0][0][0],)
        voltage_measured = (cycle_charge_data["data"]["Voltage_measured"][0][0][0],)
        current_charge = (cycle_charge_data["data"]["Current_charge"][0][0][0],)
        voltage_charge = (cycle_charge_data["data"]["Voltage_charge"][0][0][0],)
        temperature_measured = (cycle_charge_data["data"]["Temperature_measured"][0][0][0],)
        time = (cycle_charge_data["data"]["Time"][0][0][0],)
        data = pd.DataFrame(
            np.concatenate(
                [
                    current_measured,
                    voltage_measured,
                    current_charge,
                    voltage_charge,
                    temperature_measured,
                    time,
                ]
            ).T,
            columns=[
                "current_measured",
                "voltage_measured",
                "current_charge",
                "voltage_charge",
                "temperature_measured",
                "time",
            ],
        )
        num_samples = len(data)
        cycle_data.append(
            [
                "charge",
                int(battery_id[1:]),
                cycle_charge_id,
                time_cycle_start,
                ambient_temperature,
                data,
                num_samples,
            ]
        )

    for cycle_discharge_id, cycle_discharge_data in enumerate(cycles_discharge):
        time_cycle_start = datetime.datetime(*cycle_discharge_data["time"][0].astype(int))
        ambient_temperature = cycle_discharge_data["ambient_temperature"][0][0]
        current_measured = (cycle_discharge_data["data"]["Current_measured"][0][0][0],)
        voltage_measured = (cycle_discharge_data["data"]["Voltage_measured"][0][0][0],)
        current_load = (cycle_discharge_data["data"]["Current_load"][0][0][0],)
        voltage_load = (cycle_discharge_data["data"]["Voltage_load"][0][0][0],)
        temperature_measured = (cycle_discharge_data["data"]["Temperature_measured"][0][0][0],)
        time = (cycle_discharge_data["data"]["Time"][0][0][0],)
        capacity_k_Ahr = cycle_discharge_data["data"]["Capacity"][0][0][0][0]
        data = pd.DataFrame(
            np.concatenate(
                [
                    current_measured,
                    voltage_measured,
                    current_load,
                    voltage_load,
                    temperature_measured,
                    time,
                ]
            ).T,
            columns=[
                "current_measured",
                "voltage_measured",
                "current_load",
                "voltage_load",
                "temperature_measured",
                "time",
            ],
        )
        num_samples = len(data)
        cycle_data.append(
            [
                "discharge",
                int(battery_id[1:]),
                cycle_discharge_id,
                time_cycle_start,
                ambient_temperature,
                data,
                num_samples,
                capacity_k_Ahr,
            ]
        )

    df = pd.DataFrame(
        cycle_data,
        columns=[
            "cycle_type",
            "battery_id",
            "cycle_id",
            "time_cycle_start",
            "ambient_temperature",
            "data",
            "num_samples",
            "capacity_k",
        ],
    )
    df.sort_values(by="time_cycle_start", inplace=True)

    if args.filter and df.iloc[0]["cycle_type"] == "discharge":
        df.drop(df.head(1).index, inplace=True)

    return df[df["cycle_type"] == "discharge"]


def append_instantanious_capacity(data: np.ndarray, capacity_k: float) -> np.ndarray:
    data["capacity_t"] = capacity_k + integrate.cumulative_trapezoid(
        data["current_measured"], x=data["time"], initial=0.0
    ) / (3_600)
    return data

def process_battery(battery_id: str, battery_meta_data: Dict[str, str]) -> pd.DataFrame:
    """Process each individual battery data."""

    mat_file = DATASET_BASE_PATH / battery_meta_data["sub_dir"] / f"{battery_id}.mat"
    mat_file_str = str(mat_file)

    # Si jamais le chemin contient un chemin Linux (pour compatibilité)
    if mat_file_str.startswith("/home/dev_user/samba-mixer"):
        linux_base = "/home/dev_user/samba-mixer"
        # Remplacer par chemin Windows correct
        windows_base = str(BASE_DIR)
        mat_file_str = mat_file_str.replace(linux_base, windows_base)
        mat_file = Path(mat_file_str)

    if not mat_file.exists():
        raise FileNotFoundError(f"Le fichier .mat n'existe pas: {mat_file}")

    mat_db = sio.loadmat(mat_file)[battery_id]

    cycles = mat_db["cycle"][0, 0][0, :]
    discharge_df = get_discharge_cycle_df(cycles, battery_id)

    # Ajout des métadonnées et traitements
    discharge_df["capacity_0"] = battery_meta_data["c0"]
    discharge_df["fade"] = battery_meta_data["fade_in_percent"]
    discharge_df["cutoff_voltage"] = battery_meta_data["discharge"]["cutoff_voltage"]
    discharge_df["discharge_type"] = battery_meta_data["discharge"]["discharge_type"]
    discharge_df["discharge_amplitude"] = battery_meta_data["discharge"]["discharge_amplitude"]
    discharge_df["discharge_frequency"] = battery_meta_data["discharge"]["discharge_frequency"]
    discharge_df["discharge_dutycycle"] = battery_meta_data["discharge"]["discharge_dutycycle"]

    discharge_df["soh"] = discharge_df["capacity_k"] / discharge_df["capacity_0"] * 100

    if args.filter:
        discharge_df = drop_selected_cycles(discharge_df, battery_meta_data["discharge"]["drop_cycles"])
        discharge_df = drop_soh_outlier(discharge_df, threshold=10)
        discharge_df["data"] = discharge_df.apply(
            lambda per_cycle_data: filter_discharge_disconnected_load(per_cycle_data["data"]),
            axis=1,
        )

    if args.pad:
        discharge_df = pad_timesignals_discharge(discharge_df)

    discharge_df["data"] = discharge_df[["data", "capacity_k"]].apply(
        lambda df: append_instantanious_capacity(df["data"], df["capacity_k"]), axis=1
    )
    discharge_df["data_file"] = discharge_df["cycle_id"].apply(
        lambda cycle_id: f"{battery_id}/discharge/discharge_{battery_id}_{cycle_id}.npy"
    )
    discharge_df[["data", "data_file"]].apply(
        lambda x: np.save(OUTPUT_PATH / x["data_file"], x["data"].to_numpy()), axis=1
    )

    discharge_df.drop(columns=["data", "cycle_type"], inplace=True)

    return discharge_df

def _calc_global_time_diff_in_hours(new_dataset: pd.DataFrame) -> pd.DataFrame:
    new_dataset["time_diff_hours"] = new_dataset["time_cycle_start"].diff().dt.total_seconds() // 3600
    grp_battary_id = new_dataset[["battery_id", "time_diff_hours"]].groupby(by="battery_id")
    new_dataset.loc[grp_battary_id.head(1).index, "time_diff_hours"] = 0
    new_dataset["time_diff_hours"] = new_dataset["time_diff_hours"].astype("int")
    return new_dataset

def post_process(new_dataset: pd.DataFrame) -> pd.DataFrame:
    return _calc_global_time_diff_in_hours(new_dataset)


if __name__ == "__main__":
    import os
    global_data: List[pd.DataFrame] = []

    current_dir = os.path.dirname(os.path.abspath(__file__))
    yaml_path = os.path.join(current_dir, "nasa_battery_metadata.yml")

    with open(yaml_path, "r") as yaml_file:
        batteries = yaml.safe_load(yaml_file)

    for battery_id, battery_meta_data in tqdm(batteries.items(), desc="Process batteries.", colour="red"):
        if args.filter and not battery_meta_data["usable"]:
            continue
        os.makedirs(OUTPUT_PATH / battery_id / "discharge", exist_ok=True)
        battery_data = process_battery(battery_id, battery_meta_data)
        global_data.append(battery_data)

    new_dataset = pd.concat(global_data, ignore_index=True)
    new_dataset = post_process(new_dataset)
    new_dataset.to_csv(OUTPUT_PATH / "data.csv", index=False)
    print(new_dataset)
