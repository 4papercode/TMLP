import numpy as np
import pandas as pd
from teaspoon.SP.tsa_tools import takens
from teaspoon.TDA.fast_zigzag import generate_input_file, plot_output_zigzag
from teaspoon.ML.feature_functions import F_Image
import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FZZ_DIR = os.path.join(BASE_DIR, "fzz")
sys.path.insert(0, FZZ_DIR)

from pyfzz import pyfzz
import pickle
from pathlib import Path

INPUT_DIR = "./cleaned_data_16-24"
OUTPUT_DIR = "./PI"

YEARS = list(range(2016, 2025))

TRAIN_RATIO = 0.7
VAL_RATIO = 0.1
TEST_RATIO = 0.2

WINDOW_SIZE = 168
horizon = 24
STEP = 1
N_SUBWINDOWS = 6
DIM = 2
DELAY = 1

RADIUS = 0.2
N_PERM = 30

HOMOLOGY_DIM = 0
PI_PIXEL_SIZE = 0.5
PI_SIGMA = 0.5
PI_PARALLEL = False

def safe_train_mean(train_raw: np.ndarray) -> float:
    train_mean = np.nanmean(train_raw)
    return float(train_mean)

def is_valid_stat(x: float) -> bool:
    return np.isfinite(x) and not np.isnan(x)

def fill_with_train_mean(arr: np.ndarray, train_mean: float) -> np.ndarray:
    arr = np.asarray(arr, dtype=float).copy()
    nan_mask = np.isnan(arr)
    if nan_mask.any():
        arr[nan_mask] = train_mean
    return arr

def split_and_scale_ts(ts, TRAIN_RATIO, VAL_RATIO, seq_len=168):
    n = len(ts)
    num_train = int(n * TRAIN_RATIO)
    num_test  = int(n * (1 - TRAIN_RATIO - VAL_RATIO))
    num_val   = n - num_train - num_test

    # overlapping borders (same as Dataset_WECC)
    border1s = [0, num_train - seq_len, num_train + num_val - seq_len]
    border2s = [num_train, num_train + num_val, n]

    train_raw = ts[border1s[0]:border2s[0]]
    val_raw   = ts[border1s[1]:border2s[1]]
    test_raw  = ts[border1s[2]:border2s[2]]

    train_fill_value = safe_train_mean(train_raw)

    train_filled = fill_with_train_mean(train_raw, train_fill_value)
    val_filled = fill_with_train_mean(val_raw, train_fill_value)
    test_filled = fill_with_train_mean(test_raw, train_fill_value)

    train_mean = float(train_filled.mean())
    train_std = float(train_filled.std())

    if not is_valid_stat(train_mean) or not is_valid_stat(train_std):
        return None

    train_ts = (train_filled - train_mean) / train_std
    val_ts = (val_filled - train_mean) / train_std
    test_ts = (test_filled - train_mean) / train_std
    
    return {
        "train": train_ts,
        "val": val_ts,
        "test": test_ts,
    }

def build_all_point_clouds(ts, window_size, step, n_subwindows, dim, delay, horizon=24):
    subwindow_size = window_size // n_subwindows
    all_point_clouds = []
    for start in range(0, len(ts) - window_size - horizon + 1, step):
        window = ts[start:start + window_size]
        subwindow_point_clouds = []
        for k in range(n_subwindows):
            sub_start = k * subwindow_size
            sub_end = sub_start + subwindow_size
            subwindow = window[sub_start:sub_end]
            point_cloud = takens(subwindow, n=dim, tau=delay)
            subwindow_point_clouds.append(point_cloud)

        all_point_clouds.append(subwindow_point_clouds)

    return all_point_clouds

def run_zigzag_pipeline(all_point_clouds, split_name, temp_dir, radius, n_perm):
    zz = pyfzz()
    all_results = []

    for i, point_clouds in enumerate(all_point_clouds):
        input_name = str(temp_dir / f"{split_name}_output_{i}")
        pers_name = str(temp_dir / f"{split_name}_output_{i}_pers")

        inserts, deletes = generate_input_file(
            point_clouds,
            filename=input_name,
            radius=radius,
            n_perm=n_perm,
            plotting=False
        )

        data = zz.read_file(input_name)
        bars = zz.compute_zigzag(data)
        zz.write_file(pers_name, bars)

        pdict = plot_output_zigzag(
            pers_name,
            inserts,
            deletes,
            plotH2=False,
            plot=False,
            filter=True
        )

        all_results.append(pdict)

        if os.path.exists(input_name):
            os.remove(input_name)
        if os.path.exists(pers_name):
            os.remove(pers_name)

    return all_results


def pdict_list_to_diagrams(all_results, homology_dim=1):
    diagrams = []
    for pdict in all_results:
        x = np.asarray(pdict[homology_dim]["x"], dtype=float)
        y = np.asarray(pdict[homology_dim]["y"], dtype=float)
        dgm = np.column_stack([x, y])
        diagrams.append(dgm)
    return np.array(diagrams, dtype=object)

def save_npy(path, obj):
    arr = np.asarray(obj)
    np.save(path, arr)

def ensure_dir(path):
    path.mkdir(parents=True, exist_ok=True)

def process_one_year(year_df, year, output_dir):
    ts = year_df["Cleaned_Demand_MWh"].to_numpy(dtype=float)
    split_dict = split_and_scale_ts(ts, TRAIN_RATIO, VAL_RATIO, seq_len=WINDOW_SIZE)
    if split_dict is None:
        print(f"Skipping year {year}: train mean/std is NaN or non-finite.")
        return None

    train_ts = split_dict["train"]
    val_ts = split_dict["val"]
    test_ts = split_dict["test"]

    year_dir = output_dir / f"year_{year}"
    temp_dir = year_dir / "temp"
    ensure_dir(year_dir)
    ensure_dir(temp_dir)

    train_point_clouds = build_all_point_clouds(train_ts, WINDOW_SIZE, STEP, N_SUBWINDOWS, DIM, DELAY, horizon)
    val_point_clouds = build_all_point_clouds(val_ts, WINDOW_SIZE, STEP, N_SUBWINDOWS, DIM, DELAY, horizon)
    test_point_clouds = build_all_point_clouds(test_ts, WINDOW_SIZE, STEP, N_SUBWINDOWS, DIM, DELAY, horizon)

    train_results = run_zigzag_pipeline(train_point_clouds, f"train_{year}", temp_dir, RADIUS, N_PERM)
    val_results = run_zigzag_pipeline(val_point_clouds, f"val_{year}", temp_dir, RADIUS, N_PERM)
    test_results = run_zigzag_pipeline(test_point_clouds, f"test_{year}", temp_dir, RADIUS, N_PERM)

    train_diagrams = pdict_list_to_diagrams(train_results, homology_dim=HOMOLOGY_DIM)
    val_diagrams = pdict_list_to_diagrams(val_results, homology_dim=HOMOLOGY_DIM)
    test_diagrams = pdict_list_to_diagrams(test_results, homology_dim=HOMOLOGY_DIM)

    train_pi_out = F_Image(
        train_diagrams,
        PS=PI_PIXEL_SIZE,
        var=PI_SIGMA,
        pers_imager=None,
        training=True,
        parallel=PI_PARALLEL
    )
    
    val_pi_out = F_Image(
        val_diagrams,
        PS=PI_PIXEL_SIZE,
        var=PI_SIGMA,
        pers_imager=train_pi_out["pers_imager"],
        training=False,
        parallel=PI_PARALLEL
    )

    test_pi_out = F_Image(
        test_diagrams,
        PS=PI_PIXEL_SIZE,
        var=PI_SIGMA,
        pers_imager=train_pi_out["pers_imager"],
        training=False,
        parallel=PI_PARALLEL
    )

    save_npy(year_dir / "train_pi_images.npy", train_pi_out["pers_images"])
    save_npy(year_dir / "val_pi_images.npy", val_pi_out["pers_images"])
    save_npy(year_dir / "test_pi_images.npy", test_pi_out["pers_images"])

def list_csv_files(input_dir):
    return sorted([p for p in input_dir.glob("*.csv") if p.is_file()])

def main() -> None:
    input_dir = Path(INPUT_DIR)
    output_dir = Path(OUTPUT_DIR)
    ensure_dir(output_dir)
    csv_files = list_csv_files(input_dir)

    for csv_path in csv_files:
        file_output_dir = output_dir / csv_path.stem
        df = pd.read_csv(csv_path)
        for year in YEARS:
            year_df = df[df["Year"] == year].copy()
            result = process_one_year(year_df, year, file_output_dir)



if __name__ == "__main__":
    main()
