"""
Presence/absence construction, raster sampling and BRT model fitting.
"""
import ast
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
import xgboost as xgb
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import train_test_split


def cell_center(lon, lat, res=0.25):
    """Snap lon/lat coordinates to the center of their enclosing grid cell.

    Args:
        lon (numpy.ndarray or float): Longitude.
        lat (numpy.ndarray or float): Latitude.
        res (float, optional): Grid resolution in degrees. Defaults to 0.25.

    Returns:
        tuple: (lon_c, lat_c) cell-center coordinates.
    """
    lon_c = np.floor(lon / res) * res + res / 2
    lat_c = np.floor(lat / res) * res + res / 2
    return lon_c, lat_c


def build_presence_absence(gaps_df, fishing_df, res=0.25, seed=42, gear_type=None):
    """Build a balanced 1:1 presence/absence dataset: presences are grid cells with
    a suspected disabling event, absences are randomly sampled from fishing-activity
    cells that had no disabling event.

    Args:
        gaps_df (pandas.DataFrame): Suspected disabling events with a 'position' and
            'gear_type' column.
        fishing_df (pandas.DataFrame): Fishing activity with 'lon', 'lat', and
            'gear_type' columns.
        res (float, optional): Grid resolution in degrees. Defaults to 0.25.
        seed (int, optional): Random seed for absence sampling. Defaults to 42.
        gear_type (str, optional): If given, restrict to this gear type only.
            Defaults to None (all gear types).

    Returns:
        pandas.DataFrame: One row per cell, with 'lon', 'lat', and a 'presence'
            column (1 or 0).
    """
    if gear_type is not None:
        gaps_df = gaps_df[gaps_df['gear_type'].str.upper() == gear_type.upper()]
        fishing_df = fishing_df[fishing_df['gear_type'].str.upper() == gear_type.upper()]

    gaps_df = gaps_df.copy()
    gaps_df['position'] = gaps_df['position'].apply(lambda p: p if isinstance(p, dict) else ast.literal_eval(p))
    gaps_lat = gaps_df['position'].apply(lambda p: p['lat']).values
    gaps_lon = gaps_df['position'].apply(lambda p: p['lon']).values

    lon_c, lat_c = cell_center(gaps_lon, gaps_lat, res)
    presence_cells = set(zip(lon_c, lat_c))

    lon_f, lat_f = cell_center(fishing_df['lon'].values, fishing_df['lat'].values, res)
    fishing_cells = set(zip(lon_f, lat_f))

    candidate_absences = list(fishing_cells - presence_cells)
    print("Number of presences vs absences: ", len(presence_cells), len(candidate_absences))

    rng = np.random.default_rng(seed)
    n = min(len(presence_cells), len(candidate_absences))
    absence_idx = rng.choice(len(candidate_absences), size=n, replace=False)
    absence_cells = [candidate_absences[i] for i in absence_idx]

    presence_df = pd.DataFrame(list(presence_cells)[:n], columns=['lon', 'lat'])
    presence_df['presence'] = 1
    absence_df = pd.DataFrame(absence_cells, columns=['lon', 'lat'])
    absence_df['presence'] = 0

    pa_df = pd.concat([presence_df, absence_df], ignore_index=True)
    label = gear_type if gear_type else "full"
    print(f"[{label}] {presence_df.shape[0]} presences, {absence_df.shape[0]} absences")
    return pa_df


def debug_presence_absence(pa_df, gaps_df=None, fishing_df=None):
    """Check and plot a presence/absence dataset: counts, duplicate cells,
    and any cells present in both classes.

    Args:
        pa_df (pandas.DataFrame): Output of build_presence_absence.
        gaps_df (pandas.DataFrame, optional): Unused, kept for call-site symmetry.
            Defaults to None.
        fishing_df (pandas.DataFrame, optional): Unused, kept for call-site symmetry.
            Defaults to None.

    Returns:
        dict: n_presence, n_absence, duplicates, and overlap counts.
    """
    n_presence = (pa_df['presence'] == 1).sum()
    n_absence = (pa_df['presence'] == 0).sum()
    print(f"Presences: {n_presence}, Absences: {n_absence}, ratio: {n_presence/n_absence:.2f}")

    dupes = pa_df.duplicated(subset=['lon', 'lat']).sum()
    print(f"Duplicate (lon,lat) rows: {dupes}")

    overlap = pd.merge(
        pa_df[pa_df['presence'] == 1][['lon', 'lat']],
        pa_df[pa_df['presence'] == 0][['lon', 'lat']],
        on=['lon', 'lat']
    )
    print(f"Cells present in both presence and absence sets: {len(overlap)}")  # should be 0

    print(pa_df.groupby('presence')[['lon', 'lat']].describe())

    fig, ax = plt.subplots(figsize=(15, 8))
    absences = pa_df[pa_df['presence'] == 0]
    presences = pa_df[pa_df['presence'] == 1]
    ax.scatter(absences['lon'], absences['lat'], s=2, c='red', alpha=0.4, label='Absence')
    ax.scatter(presences['lon'], presences['lat'], s=2, c='blue', alpha=0.6, label='Presence')
    ax.set_title(f"Presence/Absence check ({n_presence} vs {n_absence})")
    ax.legend()
    plt.show()

    return {"n_presence": n_presence, "n_absence": n_absence, "duplicates": dupes, "overlap": len(overlap)}


def sample_rasters(pa_df, raster_paths):
    """Sample rasters at each presence/absence cell's coordinates.

    Args:
        pa_df (pandas.DataFrame): Presence/absence cells with 'lon', 'lat' columns.
        raster_paths (dict): {column_name: filepath} for each raster to sample.

    Returns:
        pandas.DataFrame: pa_df with one new column per entry in raster_paths.
    """
    df = pa_df.copy()
    coords = list(zip(df['lon'], df['lat']))

    for col_name, path in raster_paths.items():
        with rasterio.open(path) as src:
            values = [v[0] for v in src.sample(coords)]
        df[col_name] = values

    return df


def get_raster_paths(period):
    """Build the {feature name: raster path} mapping.

    Args:
        period (str): Which period's driver rasters to use (only affects the
            time-varying ones: loitering, chl, sst, sst_var, eke).

    Returns:
        dict: {column_name: filepath} for all 8 BRT drivers.
    """
    return {
        "loitering_hours": f"data/processed/loitering_hours_{period}.tif",
        "dist2coast": "data/processed/dist2coast.tif",
        "dist_mpa": "data/processed/mpa_distance_combined.tif",
        "dist_piracy": "data/processed/asam_distance.tif",
        "chl": f"data/processed/chl_{period}.tif",
        "sst": f"data/processed/sst_{period}.tif",
        "sst_var": f"data/processed/sst_variability_{period}.tif",
        "eke": f"data/processed/eke_{period}.tif",
    }


def rebalance_1to1(df, seed=42):
    """Downsample the majority class so presence and absence rows are 1:1.

    Args:
        df (pandas.DataFrame): Rows with a 'presence' column (1 or 0).
        seed (int, optional): Random seed for sampling. Defaults to 42.

    Returns:
        pandas.DataFrame: A balanced subset of df.
    """
    n = df['presence'].value_counts().min()
    return (
        df.groupby('presence', group_keys=False)
        .apply(lambda g: g.sample(n, random_state=seed))
        .reset_index(drop=True)
    )


def explained_deviance(y_true, y_pred_proba):
    """Percent deviance explained.

    Args:
        y_true (array-like): Observed 0/1 presence labels.
        y_pred_proba (array-like): Predicted probabilities from the model.

    Returns:
        float: Explained deviance as percentage.
    """
    model_dev = 2 * log_loss(y_true, y_pred_proba, normalize=False)
    null_pred = np.full_like(y_true, y_true.mean(), dtype=float)
    null_dev = 2 * log_loss(y_true, null_pred, normalize=False)
    return (null_dev - model_dev) / null_dev * 100


def tune_learning_rate(X, y, seed=42, lr_grid=(0.1, 0.05, 0.01, 0.005, 0.001), min_trees=1000, max_trees=10000):
    """Pick the largest learning rate whose early-stopped tree count reaches min_trees.

    Args:
        X (pandas.DataFrame): Feature matrix.
        y (pandas.Series): 0/1 presence labels.
        seed (int, optional): Random seed for the train/validation split. Defaults to 42.
        lr_grid (tuple, optional): Learning rates to try, high to low. Defaults to
            (0.1, 0.05, 0.01, 0.005, 0.001).
        min_trees (int, optional): Minimum acceptable early-stopped tree count.
            Defaults to 1000.
        max_trees (int, optional): Upper bound passed to early stopping. Defaults to
            10000.

    Returns:
        tuple[float, int]: (learning_rate, n_trees) for the first lr_grid value that
            reached min_trees, or the last one tried if none did.
    """
    X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=0.25, random_state=seed, stratify=y)
    for lr in lr_grid:
        clf = xgb.XGBClassifier(n_estimators=max_trees, max_depth=3, learning_rate=lr,
                                 subsample=0.6, eval_metric="logloss", random_state=seed,
                                 early_stopping_rounds=50)
        clf.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
        n_trees = clf.best_iteration + 1
        if n_trees >= min_trees:
            return lr, n_trees
    return lr_grid[-1], n_trees  # smallest lr still short of min_trees; use what we got


def fit_brt(model_df, feature_cols, n_iter=50, seed=42, **xgb_params):
    """Fit a boosted regression tree on the full data, and estimate its predictive
    skill (AUC, TSS) with repeated 75/25 splits.

    Args:
        model_df (pandas.DataFrame): Presence/absence rows with feature_cols and a
            'presence' column.
        feature_cols (list[str]): Column names to use as model covariates.
        n_iter (int, optional): Number of 75/25 train/test splits for AUC/TSS.
            Defaults to 50.
        seed (int, optional): Base random seed. Defaults to 42.

    Returns:
        tuple[xgboost.XGBClassifier, dict]: The model fit on all the data, and a
            dict of explained_deviance, auc_mean/std, tss_mean/std, n_presence,
            n_absence.
    """
    X = model_df[feature_cols]
    y = model_df["presence"]

    learning_rate, n_trees = tune_learning_rate(X, y, seed=seed)
    print(f"Tuned learning_rate={learning_rate} -> {n_trees} trees")

    params = dict(
        n_estimators=max(n_trees, 1000),
        max_depth=3,
        learning_rate=learning_rate,
        subsample=0.6,
        eval_metric="logloss",
        random_state=seed,
        importance_type="total_gain",   # <- add this
    )
    params.update(xgb_params)

    aucs, tss_scores = [], []
    for i in range(n_iter):
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, random_state=seed + i, stratify=y
        )
        clf = xgb.XGBClassifier(**{**params, "random_state": seed + i})
        clf.fit(X_train, y_train)
        preds = clf.predict_proba(X_test)[:, 1]
        aucs.append(roc_auc_score(y_test, preds))

        pred_labels = (preds >= 0.5).astype(int)
        tp = ((pred_labels == 1) & (y_test == 1)).sum()
        tn = ((pred_labels == 0) & (y_test == 0)).sum()
        fp = ((pred_labels == 1) & (y_test == 0)).sum()
        fn = ((pred_labels == 0) & (y_test == 1)).sum()
        sensitivity = tp / (tp + fn)
        specificity = tn / (tn + fp)
        tss_scores.append(sensitivity + specificity - 1)

    # final model trained on all data
    final_model = xgb.XGBClassifier(**params)
    final_model.fit(X, y)
    dev = explained_deviance(y, final_model.predict_proba(X)[:, 1])

    metrics = {
        "explained_deviance": dev,
        "auc_mean": np.mean(aucs), "auc_std": np.std(aucs),
        "tss_mean": np.mean(tss_scores), "tss_std": np.std(tss_scores),
        "n_presence": int((y == 1).sum()), "n_absence": int((y == 0).sum()),
    }
    print(f"Explained deviance: {dev:.2f}%  AUC: {metrics['auc_mean']:.3f} ± {metrics['auc_std']:.3f}  "
          f"TSS: {metrics['tss_mean']:.3f} ± {metrics['tss_std']:.3f}")

    return final_model, metrics


def plot_importance(model, feature_cols, title="Feature importance"):
    """Plot a fitted model's feature importances as a bar chart.

    Args:
        model (xgboost.XGBClassifier): A fitted model.
        feature_cols (list[str]): Column names, in the same order used to fit model.
        title (str, optional): Plot title. Defaults to "Feature importance".

    Returns:
        None: Displays the plot.
    """
    importances = model.feature_importances_
    order = np.argsort(importances)[::-1]
    plt.barh([feature_cols[i] for i in order], importances[order])
    plt.title(title)
    plt.gca().invert_yaxis()
    plt.show()


def run_model(gaps_df, fishing_df, feature_cols, raster_paths, gear_type=None, outdir="model_results"):
    """Run the full pipeline for one gear type (or the full model): build
    presence/absence, sample covariates, fit the BRT, and save metrics + importance.

    Args:
        gaps_df (pandas.DataFrame): Suspected disabling events.
        fishing_df (pandas.DataFrame): Fishing activity (source of absences).
        feature_cols (list[str]): Covariate columns to use.
        raster_paths (dict): {column_name: filepath}, as from get_raster_paths.
        gear_type (str, optional): Restrict to one gear type. Defaults to None
            (full model).
        outdir (str, optional): Directory to write metrics/importance to. Defaults
            to "model_results".

    Returns:
        tuple[pandas.DataFrame, xgboost.XGBClassifier, dict]: The final model_df,
            fitted model, and metrics dict.
    """
    os.makedirs(outdir, exist_ok=True)
    label = gear_type if gear_type else "full"

    pa_df = build_presence_absence(gaps_df, fishing_df, gear_type=gear_type)
    model_df = sample_rasters(pa_df, raster_paths)
    # loitering_hours: NaN = no loitering recorded = 0 hours, not missing data
    model_df['loitering_hours'] = model_df['loitering_hours'].fillna(0)
    model_df = model_df.dropna(subset=[c for c in feature_cols if c != 'loitering_hours'])
    model_df = rebalance_1to1(model_df)

    model, metrics = fit_brt(model_df, feature_cols)

    with open(f"{outdir}/metrics_{label}.json", "w") as f:
        json.dump(metrics, f, indent=2)

    importances = model.feature_importances_
    importance_dict = dict(zip(feature_cols, [float(v) for v in importances]))
    importance_dict = dict(sorted(importance_dict.items(), key=lambda x: x[1], reverse=True))

    with open(f"{outdir}/importance_{label}.json", "w") as f:
        json.dump(importance_dict, f, indent=2)

    plot_importance(model, feature_cols, title=f"{label} feature importance")
    plt.savefig(f"{outdir}/importance_{label}.png", dpi=150, bbox_inches="tight")
    plt.close()

    print(f"Saved metrics + importance for '{label}' -> {outdir}")
    return model_df, model, metrics