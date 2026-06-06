import os
import sys
from pathlib import Path
from skimage.transform import resize

import numpy as np
import pandas as pd
from teaspoon.SP.tsa_tools import takens
from teaspoon.TDA.fast_zigzag import generate_input_file, plot_output_zigzag
from teaspoon.ML.feature_functions import F_Image

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FZZ_DIR = os.path.join(BASE_DIR, "fzz")
sys.path.insert(0, FZZ_DIR)

from pyfzz import pyfzz

INPUT_DIR = "./cleaned_data_16-24"
OUTPUT_DIR = "./PI_ba_groups"

YEARS = list(range(2016, 2025))

TRAIN_RATIO = 0.7
VAL_RATIO = 0.1
TEST_RATIO = 0.2

WINDOW_SIZE = 168
HORIZON = 24
STEP = 1
DIM = 2
DELAY = 1

RADIUS = 0.2
N_PERM = 30

HOMOLOGY_DIM = 0
PI_PIXEL_SIZE = 0.5
PI_SIGMA = 0.5
PI_PARALLEL = False

PI_TARGET_SIZE = (10, 10)

BA_GROUPS = {
    "group_1": ["SRP", "AZPS", "CISO", "TEPC", "WALC"],
    "group_2": ["NWMT", "AVA", "BPAT", "IPCO", "PACE", "WAUW"],
    "group_3": ["WALC", "AZPS", "CISO", "IID", "NEVP", "SRP", "TEPC", "WACM"],
    "group_4": ["CISO", "AZPS", "BANC", "BPAT", "IID", "NEVP", "PACW", "SRP", "TIDC", "WALC"],
    "group_5": ["PSEI", "BPAT", "CHPD", "GCPD", "SCL", "TPWR"],
    "group_6": ["TEPC", "AZPS", "EPE", "PNM", "SRP", "WALC"],
}

def resize_pi_images(pi_images, target_size=PI_TARGET_SIZE):
    resized = []
    for img in pi_images:
        resized_img = resize(
            img,
            target_size,
            preserve_range=True,
            anti_aliasing=True
        )
        resized.append(resized_img)
    return np.asarray(resized)

def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def save_npy(path: Path, obj) -> None:
    np.save(path, np.asarray(obj))


def infer_ba_name(csv_path: Path) -> str:
    """
    Supports filenames such as:
        AVA.csv
        AVA_cleaned_historical_data.csv
    """
    return csv_path.stem.split("_")[0].upper()


def list_csv_files(input_dir: Path):
    return sorted([p for p in input_dir.glob("*.csv") if p.is_file()])


def load_ba_frames(input_dir: Path) -> dict:
    ba_frames = {}
    for csv_path in list_csv_files(input_dir):
        ba = infer_ba_name(csv_path)
        df = pd.read_csv(csv_path)
        ba_frames[ba] = df.copy()
    return ba_frames


def fill_matrix_with_train_mean(arr: np.ndarray, train_mean: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr, dtype=float).copy()
    nan_rows, nan_cols = np.where(np.isnan(arr))
    if len(nan_rows) > 0:
        arr[nan_rows, nan_cols] = train_mean[nan_cols]
    return arr


def split_and_scale_matrix(X: np.ndarray, train_ratio: float, val_ratio: float, seq_len: int):
    X = np.asarray(X, dtype=float)
    n = X.shape[0]
    num_train = int(n * train_ratio)
    num_test = int(n * (1 - train_ratio - val_ratio))
    num_val = n - num_train - num_test

    border1s = [0, num_train - seq_len, num_train + num_val - seq_len]
    border2s = [num_train, num_train + num_val, n]

    if min(border1s) < 0:
        return None

    train_raw = X[border1s[0]:border2s[0], :]
    val_raw = X[border1s[1]:border2s[1], :]
    test_raw = X[border1s[2]:border2s[2], :]

    train_fill_value = np.nanmean(train_raw, axis=0)

    train_filled = fill_matrix_with_train_mean(train_raw, train_fill_value)
    val_filled = fill_matrix_with_train_mean(val_raw, train_fill_value)
    test_filled = fill_matrix_with_train_mean(test_raw, train_fill_value)

    train_mean = train_filled.mean(axis=0)
    train_std = train_filled.std(axis=0)

    if not np.all(np.isfinite(train_mean)):
        return None
    if not np.all(np.isfinite(train_std)):
        return None
    if np.any(train_std <= 0):
        return None

    return {
        "train": (train_filled - train_mean) / train_std,
        "val": (val_filled - train_mean) / train_std,
        "test": (test_filled - train_mean) / train_std,
        "train_mean": train_mean,
        "train_std": train_std,
    }


def build_group_year_matrix(ba_frames: dict, ba_order: list, year: int):
    missing_bas = [ba for ba in ba_order if ba not in ba_frames]
    if missing_bas:
        print(f"Missing BA files: {missing_bas}. Skip this group/year.")
        return None, None

    year_dfs = {}

    for ba in ba_order:
        df = ba_frames[ba]

        required = ["Year", "Month", "Day", "Hour", "Cleaned_Demand_MWh"]
        missing_cols = [c for c in required if c not in df.columns]
        if missing_cols:
            print(f"{ba} missing columns: {missing_cols}. Skip this group/year.")
            return None, None

        year_df = df[df["Year"] == year].copy()

        if year_df.empty:
            print(f"{ba} has no data for year {year}. Skip this group/year.")
            return None, None

        year_dfs[ba] = year_df.reset_index(drop=True)

    lengths = {ba: len(year_dfs[ba]) for ba in ba_order}
    if len(set(lengths.values())) != 1:
        print(f"Unequal lengths for year {year}: {lengths}. Skip this group/year.")
        return None, None

    # optional but recommended: check Year/Month/Day/Hour are aligned by row
    ref_ba = ba_order[0]
    ref_time = year_dfs[ref_ba][["Year", "Month", "Day", "Hour"]].reset_index(drop=True)

    for ba in ba_order[1:]:
        ba_time = year_dfs[ba][["Year", "Month", "Day", "Hour"]].reset_index(drop=True)

        if not ref_time.equals(ba_time):
            print(f"Time columns are not aligned between {ref_ba} and {ba} in year {year}.")
            return None, None

    # build output dataframe
    group_df = ref_time.copy()

    for ba in ba_order:
        group_df[ba] = year_dfs[ba]["Cleaned_Demand_MWh"].to_numpy(dtype=float)

    # shape: (T, n_ba)
    X = group_df[ba_order].to_numpy(dtype=float)

    return group_df, X


def build_all_ba_point_clouds(X_scaled: np.ndarray, window_size: int, step: int, dim: int, delay: int, horizon: int):

    all_point_clouds = []
    n_time, n_ba = X_scaled.shape

    max_start = n_time - window_size - horizon + 1
    if max_start <= 0:
        return all_point_clouds

    for start in range(0, max_start, step):
        window_matrix = X_scaled[start:start + window_size, :]  # shape: (window_size, n_ba)
        ba_point_clouds = []

        for j in range(n_ba):
            ba_window = window_matrix[:, j]
            point_cloud = takens(ba_window, n=dim, tau=delay)
            ba_point_clouds.append(point_cloud)

        all_point_clouds.append(ba_point_clouds)

    return all_point_clouds


def run_zigzag_pipeline(all_point_clouds, split_name: str, temp_dir: Path, radius: float, n_perm: int):
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
            plotting=False,
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
            filter=True,
        )

        all_results.append(pdict)

        if os.path.exists(input_name):
            os.remove(input_name)
        if os.path.exists(pers_name):
            os.remove(pers_name)

    return all_results


def pdict_list_to_diagrams(all_results, homology_dim: int = 0):
    diagrams = []
    for pdict in all_results:
        x = np.asarray(pdict[homology_dim]["x"], dtype=float)
        y = np.asarray(pdict[homology_dim]["y"], dtype=float)
        dgm = np.column_stack([x, y])
        diagrams.append(dgm)
    return np.array(diagrams, dtype=object)


def compute_and_save_pi(diagrams, year_dir: Path, split_name: str, pers_imager=None, training=False, target_size=(10, 10)):
    pi_out = F_Image(
        diagrams,
        PS=PI_PIXEL_SIZE,
        var=PI_SIGMA,
        pers_imager=pers_imager,
        training=training,
        parallel=PI_PARALLEL,
    )
    pi_images = np.asarray(pi_out["pers_images"])
    pi_images = resize_pi_images(pi_images, target_size=target_size)
    save_npy(year_dir / f"{split_name}_pi_images.npy", pi_images)
    return pi_out


def process_one_group_year(ba_frames: dict, group_name: str, ba_order: list, year: int, output_dir: Path):
    merged, X_raw = build_group_year_matrix(ba_frames, ba_order, year)
    if X_raw is None:
        return None

    split_dict = split_and_scale_matrix(X_raw, TRAIN_RATIO, VAL_RATIO, seq_len=WINDOW_SIZE)
    if split_dict is None:
        print(f"Skipping {group_name}, year {year}: invalid train mean/std or insufficient length.")
        return None

    group_dir = output_dir / group_name
    year_dir = group_dir / f"year_{year}"
    temp_dir = year_dir / "temp"
    ensure_dir(year_dir)
    ensure_dir(temp_dir)

    # Save metadata for reproducibility.
    pd.Series(ba_order).to_csv(year_dir / "ba_order.csv", index=False, header=["BA"])

    train_point_clouds = build_all_ba_point_clouds(
        split_dict["train"], WINDOW_SIZE, STEP, DIM, DELAY, HORIZON
    )
    val_point_clouds = build_all_ba_point_clouds(
        split_dict["val"], WINDOW_SIZE, STEP, DIM, DELAY, HORIZON
    )
    test_point_clouds = build_all_ba_point_clouds(
        split_dict["test"], WINDOW_SIZE, STEP, DIM, DELAY, HORIZON
    )

    print(
        f"{group_name}, year {year}: "
        f"train windows={len(train_point_clouds)}, "
        f"val windows={len(val_point_clouds)}, "
        f"test windows={len(test_point_clouds)}"
    )

    if len(train_point_clouds) == 0 or len(val_point_clouds) == 0 or len(test_point_clouds) == 0:
        print(f"Skipping {group_name}, year {year}: no valid windows.")
        return None

    train_results = run_zigzag_pipeline(train_point_clouds, f"{group_name}_train_{year}", temp_dir, RADIUS, N_PERM)
    val_results = run_zigzag_pipeline(val_point_clouds, f"{group_name}_val_{year}", temp_dir, RADIUS, N_PERM)
    test_results = run_zigzag_pipeline(test_point_clouds, f"{group_name}_test_{year}", temp_dir, RADIUS, N_PERM)

    train_diagrams = pdict_list_to_diagrams(train_results, homology_dim=HOMOLOGY_DIM)
    val_diagrams = pdict_list_to_diagrams(val_results, homology_dim=HOMOLOGY_DIM)
    test_diagrams = pdict_list_to_diagrams(test_results, homology_dim=HOMOLOGY_DIM)

    train_pi_out = compute_and_save_pi(train_diagrams, year_dir, "train", pers_imager=None, training=True, target_size=PI_TARGET_SIZE)
    compute_and_save_pi(val_diagrams, year_dir, "val", pers_imager=train_pi_out["pers_imager"], training=False, target_size=PI_TARGET_SIZE)
    compute_and_save_pi(test_diagrams, year_dir, "test", pers_imager=train_pi_out["pers_imager"], training=False, target_size=PI_TARGET_SIZE)

    # Optional cleanup of temp directory if it is empty.
    try:
        temp_dir.rmdir()
    except OSError:
        pass

    return True


def main() -> None:
    input_dir = Path(INPUT_DIR)
    output_dir = Path(OUTPUT_DIR)
    ensure_dir(output_dir)

    ba_frames = load_ba_frames(input_dir)
    print(f"Loaded BA files: {sorted(ba_frames.keys())}")

    for group_name, ba_order in BA_GROUPS.items():
        print(f"\nProcessing {group_name}: {ba_order}")
        for year in YEARS:
            process_one_group_year(ba_frames, group_name, ba_order, year, output_dir)


if __name__ == "__main__":
    main()
