#!/usr/bin/env python3
import os
import glob
import logging
from optuna.trial import TrialState
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import optuna
from optuna.samplers import TPESampler
import faulthandler, signal, sys

# Enable for fatal errors on all threads:
faulthandler.enable(file=sys.stderr, all_threads=True)

# If you want to trigger a dump manually via kill -USR1 <pid>:
faulthandler.register(signal.SIGUSR1, file=sys.stderr, all_threads=True)

from tools_cm import compute_confusion_matrix, f1_score
import plot_repository

# ----------------------------
# 1) Data Loading / Preprocess
# ----------------------------
def load_and_preprocess_data(pwd, folder_neolo, catalog_path, start, end):
    txt_files = []
    dates = pd.date_range(start, end, freq="D").strftime("%Y%m%d")
    for d in dates:
        matches = glob.glob(f"{pwd}/{folder_neolo}/vlp_{d}.txt")
        if matches:
            txt_files.append(matches[0])

    dfs = []
    for f in txt_files:
        try:
            dfs.append(pd.read_csv(f))
        except pd.errors.ParserError as e:
            print(f"❌ Failed to parse {f}: {e}")
    if not dfs:
        raise RuntimeError("No VLP files loaded!")
    df = pd.concat(dfs, ignore_index=True)

    # datetime cleanup
    df["start_vlp_iso"] = pd.to_datetime(df["start_vlp_iso"]).dt.tz_localize(None)
    df["tca_iso"] = pd.to_datetime(df["tca_iso"], errors="coerce").dt.tz_localize(None)
    df["tca_iso_sec"] = df["tca_iso"].dt.round("S")

    # drop dupes
    df.drop_duplicates(subset=["station", "start_vlp_iso"], inplace=True)
    df.drop_duplicates(subset=["station", "tca_iso_sec"], inplace=True)

    # time window filter
    mask = (df["start_vlp_iso"] >= start) & (df["start_vlp_iso"] < end)
    df = df.loc[mask].sort_values("start_vlp_iso")

    # manual catalog
    cat = pd.read_csv(catalog_path)
    cat["datetime"] = pd.to_datetime(cat["datetime"])
    cat = cat[(cat["datetime"] >= start) & (cat["datetime"] < end)]

    return df, cat

# ----------------------------
# 2) Objective
# ----------------------------
def objective(trial, station, df_neolo, cat_str, dti, tolerance, results_list):
    try:
        # parameter grid
        max_tca = trial.suggest_int("max_tca", 4000, 12000, step=1000)
        max_rsam = trial.suggest_int("max_rsam", 4000, 12000, step=1000)
        az_max = trial.suggest_int("azimuth_max_std", 8, 20)
        inc_max = trial.suggest_int("incidence_max_std", 8, 20)
        rect_min = trial.suggest_float("rect_min", 0.3, 0.8, step=0.05)
        rect_max = trial.suggest_float("rect_max", 0.8, 1, step=0.001)
        plan_min = trial.suggest_float("planarity_min", 0.3, 0.8, step=0.05)
        plan_max = trial.suggest_float("planarity_max", 0.8, 1, step=0.001)

        # filter
        sub = df_neolo.query(
            "station==@station & "
            "max_tca.fillna(1e12)<=@max_tca & rsam.fillna(1e12)<=@max_rsam & "
            "azimuth<=@az_max & incidence<=@inc_max & "
            "rectilinearity_tca>=@rect_min & rectilinearity_tca<=@rect_max & "
            "planarity_tca>=@plan_min & planarity_tca<=@plan_max"
        )

        # confusion matrix & F1
        cm_df = compute_confusion_matrix(dti, sub, cat_str, tolerance, [station])
        if cm_df.empty:
            return 0.0

        # average non-zero metrics
        for c in ["f1-score", "pr", "rc"]:
            cm_df[c] = cm_df[cm_df[c] != 0].groupby("station")[c].transform("mean")

        best_f1 = cm_df["f1-score"].max()
    except Exception as e:
        # catch underflows, overflows, etc.
        logging.warning(f"Trial error for station {station}: {e}")
        best_f1 = 0.0

    # record
    results_list.append({
        "iteration": len(results_list) + 1,
        "f1-score": best_f1,
        "params": {
            "station": station,
            "max_tca": max_tca,
            "max_rsam": max_rsam,
            "az_max": az_max,
            "inc_max": inc_max,
            "rect_min": rect_min,
            "rect_max": rect_max,
            "plan_min": plan_min,
            "plan_max": plan_max,
        }
    })
    return best_f1
# ----------------------------
# 3) Per‐Station Optimization
# ----------------------------


def run_optimizations(df_neolo, cat_str, dti, tolerance, pwd, test_name):
    sampler = TPESampler(
        consider_prior=True,
        prior_weight=1.0,
        consider_magic_clip=True,
        n_startup_trials=10,
        multivariate=True,
        warn_independent_sampling=False,
    )

    studies_dir = os.path.join(pwd, "optuna_studies_2")
    os.makedirs(studies_dir, exist_ok=True)
    result_dir = os.path.join(pwd, "vlp_result", test_name)
    os.makedirs(result_dir, exist_ok=True)

    stations = ["STRA", "STR1", "STR8", "STR6", "STR9", "STRD", "STRB", "STR5"]
    best_params = {}

    TARGET_TRIALS = 150

    for st in stations:
        db_path = os.path.join(studies_dir, f"study_{st}.db")
        storage_url = f"sqlite:///{db_path}"
        study = optuna.create_study(
            direction="maximize",
            sampler=sampler,
            study_name=f"station_{st}",
            storage=storage_url,
            load_if_exists=True,
        )
        
        # Count completed trials via Study.trials
        completed = sum(1 for t in study.trials if t.state == TrialState.COMPLETE)
        print(f"Station {st}: {completed} completed trials on disk.")

        if completed >= TARGET_TRIALS:
            print(f"✔ Skipping {st}: already has ≥{TARGET_TRIALS} trials.")
            best_params[st] = study.best_params
            continue

        to_run = TARGET_TRIALS - completed
        print(f"▶ Running {to_run} new trials for station {st}…")
        results = []
        study.optimize(
            lambda t: objective(t, st, df_neolo, cat_str, dti, tolerance, results),
            n_trials=to_run,
        )

        total_done = completed + sum(1 for t in study.trials if t.state == TrialState.COMPLETE) - completed
        print(f"✔ Station {st} now has {completed + len(results)} completed trials; Best F1 = {study.best_value:.3f}")

        pd.DataFrame(results).to_csv(
            f"{result_dir}/trials_{st}.csv", index=False
        )
        best_params[st] = study.best_params

    # Save best_params
    pd.DataFrame([
        {"station": st, **best_params[st]}
        for st in best_params
    ]).to_csv(f"{pwd}/{folder_neolo}/best_params_{test_name}.csv", index=False)

    return studies_dir, result_dir
def load_and_filter_trials(db_path, study_name=None):
    """Load an Optuna study from a SQLite file and return a filtered DataFrame."""
    storage = f"sqlite:///{db_path}"
    # If study_name is None, Optuna will pick the only study inside the DB.
    study = optuna.load_study(study_name=study_name, storage=storage)
    df = study.trials_dataframe(attrs=("number", "value", "state"))
    # keep only completed trials with finite, non-null F1 scores
    df = df[df["state"] == "COMPLETE"]
    df = df[df["value"].notnull() & df["value"].apply(lambda v: pd.api.types.is_scalar(v))]
    df = df[df["value"] != float("inf")][df["value"] != float("-inf")]
    return df.sort_values("number"), study


# ----------------------------
# 4) Plot Studies After Optuna
# ----------------------------
def plot_all_studies_improved(studies_dir, result_dir):
    dbs = glob.glob(os.path.join(studies_dir, "study_*.db"))
    target_stations = ["STRA", "STR1", "STR8", "STR6", "STR9", "STRD", "STRB", "STR5"]
    combined = {}

    for db in dbs:
        station = os.path.basename(db).replace("study_", "").replace(".db", "")
        if station not in target_stations:
            continue

        try:
            df, _ = load_and_filter_trials(db, study_name=None)
            combined[station] = df
        except Exception as e:
            logging.warning(f"Could not load data for {station}: {e}")

    if not combined:
        logging.warning("No valid studies found to plot.")
        return

    # Plot all stations in subplots
    num_stations = len(combined)
    fig, axes = plt.subplots(nrows=num_stations, ncols=1, figsize=(10, 3 * num_stations), sharex=True)
    
    if num_stations == 1:
        axes = [axes]  # ensure axes is iterable

    for ax, (station, df) in zip(axes, combined.items()):
        ax.plot(df["number"], df["value"], marker="o", linestyle="-", alpha=0.5, label="Raw F1")
        # Rolling average for smoothing
        df_sorted = df.sort_values("number")
        rolling = df_sorted["value"].rolling(window=5, min_periods=1).mean()
        ax.plot(df_sorted["number"], rolling, color="red", linewidth=2, label="Smoothed F1")
        ax.set_ylabel(f"{station}\nF1")
        ax.set_ylim(0, 1)
        ax.grid(True)
        ax.legend()

    axes[-1].set_xlabel("Trial number")
    fig.suptitle("F1 Score vs Iteration — All Stations", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.97])  # leave space for suptitle
    plt.savefig(os.path.join(result_dir, "f1_all_stations_improved.png"))
    plt.close()
def plot_all_studies(studies_dir, result_dir):
    dbs = glob.glob(os.path.join(studies_dir, "study_*.db"))
    combined = {}
    target_stations = ["STRA", "STR1", "STR8", "STR6", "STR9", "STRD", "STRB", "STR5"]

    for db in dbs:
        station = os.path.basename(db).replace("study_", "").replace(".db", "")
        
        if station not in target_stations:
            continue

        try:
            df, _ = load_and_filter_trials(db, study_name=None)
            combined[station] = df

            # Individual station plot
            plt.figure(figsize=(8, 5))
            plt.plot(df["number"], df["value"], marker="o")
            plt.title(f"{station} — F1 vs Iteration")
            plt.xlabel("Trial number")
            plt.ylabel("F1 score")
            plt.grid(True)
            plt.savefig(f"{result_dir}/f1_{station}.png")
            plt.close()
        except Exception as e:
            logging.warning(f"Could not plot {station}: {e}")

    # Multi-row subplot figure
    if combined:
        n_stations = len(combined)
        fig, axes = plt.subplots(n_stations, 1, figsize=(10, 3 * n_stations), sharex=True)
        if n_stations == 1:
            axes = [axes]  # Ensure iterable

        for ax, (station, df) in zip(axes, combined.items()):
            ax.plot(df["number"], df["value"], marker="o")
            ax.set_ylabel(f"{station}\nF1")
            ax.grid(True)

        axes[-1].set_xlabel("Trial number")
        fig.suptitle("F1 Score vs Iteration — All Stations", fontsize=14)
        plt.tight_layout(rect=[0, 0, 1, 0.97])
        plt.savefig(f"{result_dir}/f1_all_stations.png")
        plt.close()

# ----------------------------
# 5) Main Pipeline
# ----------------------------
if __name__ == "__main__":
    # parameters
    pwd            = "/home/sergio/Documenti/stromboli_vlp"
    folder_neolo   = "neolo"
    catalog_path   = "./catalogho_stromboli.csv"
    test_name      = "condition"
    start, end     = "2007-01-01", "2008-01-01"
    tolerance      = 60
    dti            = pd.date_range(start, end, freq="D")

    # 1) load data
    df_neolo, cat_str = load_and_preprocess_data(
        pwd, folder_neolo, catalog_path, start, end
    )

    # 2) run optuna per station
    studies_dir, results_dir = run_optimizations(
        df_neolo, cat_str, dti, tolerance, pwd, test_name
    )

    # 3) plot all studies
    plot_all_studies_improved(studies_dir, results_dir)
    # 4) combine best-filtered VLPs & compute final confusion matrix
    best_params_df = pd.read_csv(f"{pwd}/{folder_neolo}/best_params_{test_name}.csv")
    frames = []
    print(best_params_df.station.tolist())
    cm = compute_confusion_matrix(dti, df_neolo, cat_str, tolerance, best_params_df.station.tolist())
    cm.to_csv(f"{results_dir}/previous_confusion_matrix.csv", index=False)
    for _, row in best_params_df.iterrows():
        st = row["station"]
        filt = (
            (df_neolo.station == st) &
            (df_neolo.max_tca <= row.max_tca) &
            (df_neolo.rsam <= row.max_rsam) &
            (df_neolo.azimuth <= row.azimuth_max_std) &
            (df_neolo.incidence <= row.incidence_max_std) &
            (df_neolo.rectilinearity_tca.between(row.rect_min, row.rect_max)) &
            (df_neolo.planarity_tca.between(row.planarity_min, row.planarity_max))
        )
        frames.append(df_neolo.loc[filt])

    df_best_all = pd.concat(frames, ignore_index=True)
    df_best_all.to_csv(f"{pwd}/{folder_neolo}/vlp_best_filtered_{test_name}.csv", index=False)
    cm = compute_confusion_matrix(dti, df_best_all, cat_str, tolerance, best_params_df.station.tolist())
    cm.to_csv(f"{results_dir}/final_confusion_matrix.csv", index=False)

    # plot final rates/scores
    plot_repository.plot_rate_and_score(
        results_dir, tolerance, folder_neolo, start, end
    )
