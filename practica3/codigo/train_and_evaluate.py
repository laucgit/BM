from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, wasserstein_distance
from sklearn.base import clone
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score, roc_curve
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

RANDOM_STATE = 42
FIG_DPI = 180
RF_TREES = 500
GMM_MAX_COMPONENTS = 6
GMM_COVARIANCE_TYPES = ("diag", "full")
GMM_N_INIT = 10


def ensure_dirs(base_dir: Path) -> Dict[str, Path]:
    outputs = base_dir / "outputs"
    figures = outputs / "figures"
    outputs.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    return {"base": base_dir, "outputs": outputs, "figures": figures}


def load_dataset() -> Tuple[pd.DataFrame, pd.Series, Dict[int, str]]:
    dataset = load_breast_cancer(as_frame=True)
    return dataset.data.copy(), dataset.target.copy(), {0: "malignant", 1: "benign"}


def fit_class_conditional_gmms(X_train_scaled: np.ndarray, y_train: pd.Series, max_components: int = GMM_MAX_COMPONENTS) -> Dict[int, Dict[str, object]]:
    models: Dict[int, Dict[str, object]] = {}
    for cls in sorted(y_train.unique()):
        X_cls = X_train_scaled[y_train.values == cls]
        best_bic = np.inf
        best_model = None
        best_k = None
        best_covariance_type = None
        for covariance_type in GMM_COVARIANCE_TYPES:
            for n_components in range(1, max_components + 1):
                model = GaussianMixture(
                    n_components=n_components,
                    covariance_type=covariance_type,
                    reg_covar=1e-5,
                    random_state=RANDOM_STATE,
                    n_init=GMM_N_INIT,
                    max_iter=400,
                )
                model.fit(X_cls)
                bic = model.bic(X_cls)
                if bic < best_bic:
                    best_bic = bic
                    best_model = model
                    best_k = n_components
                    best_covariance_type = covariance_type
        models[int(cls)] = {
            "model": best_model,
            "n_components": int(best_k),
            "covariance_type": str(best_covariance_type),
            "bic": float(best_bic),
        }
    return models


def sample_synthetic_dataset(models, scaler, X_train, y_train) -> Tuple[pd.DataFrame, pd.Series]:
    feature_names = list(X_train.columns)
    class_counts = y_train.value_counts().sort_index()
    X_syn_parts: List[np.ndarray] = []
    y_syn_parts: List[np.ndarray] = []
    for cls, n_rows in class_counts.items():
        samples, _ = models[int(cls)]["model"].sample(int(n_rows))
        X_syn_parts.append(samples)
        y_syn_parts.append(np.full(int(n_rows), cls))
    X_syn_scaled = np.vstack(X_syn_parts)
    y_syn = pd.Series(np.concatenate(y_syn_parts), name="target")
    rng = np.random.default_rng(RANDOM_STATE)
    perm = rng.permutation(len(y_syn))
    X_syn_scaled = X_syn_scaled[perm]
    y_syn = y_syn.iloc[perm].reset_index(drop=True)
    X_syn = pd.DataFrame(scaler.inverse_transform(X_syn_scaled), columns=feature_names)
    X_syn = X_syn.clip(lower=X_train.min(), upper=X_train.max(), axis=1)
    return X_syn, y_syn


def compute_fidelity_metrics(X_train: pd.DataFrame, X_syn: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, float]]:
    rows = []
    ks_values, wd_values, mean_diffs, std_ratio_devs = [], [], [], []
    for col in X_train.columns:
        real = X_train[col].to_numpy()
        syn = X_syn[col].to_numpy()
        real_mean, syn_mean = float(real.mean()), float(syn.mean())
        real_std, syn_std = float(real.std(ddof=1)), float(syn.std(ddof=1))
        ks = float(ks_2samp(real, syn).statistic)
        wd = float(wasserstein_distance(real, syn))
        z_mean = abs(syn_mean - real_mean) / real_std if real_std else 0.0
        norm_wd = wd / real_std if real_std else 0.0
        rows.append({
            "feature": col,
            "real_mean": real_mean,
            "synthetic_mean": syn_mean,
            "real_std": real_std,
            "synthetic_std": syn_std,
            "abs_mean_diff_std_units": z_mean,
            "ks_statistic": ks,
            "wasserstein_distance": wd,
            "normalized_wasserstein": norm_wd,
        })
        ks_values.append(ks)
        wd_values.append(norm_wd)
        mean_diffs.append(z_mean)
        std_ratio_devs.append(abs(syn_std / real_std - 1.0) if real_std else 0.0)
    fidelity_df = pd.DataFrame(rows)
    corr_real = X_train.corr(numeric_only=True).values
    corr_syn = X_syn.corr(numeric_only=True).values
    mask = np.triu(np.ones_like(corr_real, dtype=bool), k=1)
    summary = {
        "avg_standardized_mean_diff": float(np.mean(mean_diffs)),
        "avg_std_ratio_deviation": float(np.mean(std_ratio_devs)),
        "avg_ks_statistic": float(np.mean(ks_values)),
        "max_ks_statistic": float(np.max(ks_values)),
        "avg_normalized_wasserstein": float(np.mean(wd_values)),
        "mean_abs_corr_gap": float(np.abs(corr_real - corr_syn)[mask].mean()),
    }
    return fidelity_df, summary


def _utility_models():
    return {
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=5000, random_state=RANDOM_STATE)),
        ]),
        "Random Forest": RandomForestClassifier(
            n_estimators=RF_TREES,
            min_samples_leaf=2,
            random_state=RANDOM_STATE,
            n_jobs=1,
        ),
    }


def compute_utility_metrics(X_train, y_train, X_syn, y_syn, X_test, y_test):
    rows = []
    feature_overlap: Dict[str, List[str] | int] = {}
    for model_name, model in _utility_models().items():
        real_model = clone(model)
        real_model.fit(X_train, y_train)
        pred_real = real_model.predict(X_test)
        proba_real = real_model.predict_proba(X_test)[:, 1]
        rows.append({
            "model": model_name,
            "setting": "TRTR",
            "accuracy": float(accuracy_score(y_test, pred_real)),
            "f1": float(f1_score(y_test, pred_real)),
            "roc_auc": float(roc_auc_score(y_test, proba_real)),
        })
        syn_model = clone(model)
        syn_model.fit(X_syn, y_syn)
        pred_syn = syn_model.predict(X_test)
        proba_syn = syn_model.predict_proba(X_test)[:, 1]
        rows.append({
            "model": model_name,
            "setting": "TSTR",
            "accuracy": float(accuracy_score(y_test, pred_syn)),
            "f1": float(f1_score(y_test, pred_syn)),
            "roc_auc": float(roc_auc_score(y_test, proba_syn)),
        })
        if model_name == "Random Forest":
            real_imp = pd.Series(real_model.feature_importances_, index=X_train.columns).sort_values(ascending=False)
            syn_imp = pd.Series(syn_model.feature_importances_, index=X_train.columns).sort_values(ascending=False)
            feature_overlap = {
                "rf_top10_real": real_imp.head(10).index.tolist(),
                "rf_top10_synthetic": syn_imp.head(10).index.tolist(),
                "rf_top10_overlap_count": int(len(set(real_imp.head(10).index) & set(syn_imp.head(10).index))),
            }
    return pd.DataFrame(rows), feature_overlap


def compute_membership_inference_metrics(X_train, X_test, X_syn, scaler):
    train_scaled = scaler.transform(X_train)
    test_scaled = scaler.transform(X_test)
    syn_scaled = scaler.transform(X_syn)
    nn_syn = NearestNeighbors(n_neighbors=1, metric="euclidean").fit(syn_scaled)
    train_to_syn = nn_syn.kneighbors(train_scaled, n_neighbors=1)[0][:, 0]
    test_to_syn = nn_syn.kneighbors(test_scaled, n_neighbors=1)[0][:, 0]
    labels = np.concatenate([np.ones(len(train_to_syn)), np.zeros(len(test_to_syn))])
    scores = -np.concatenate([train_to_syn, test_to_syn])
    auc = float(roc_auc_score(labels, scores))
    fpr, tpr, thresholds = roc_curve(labels, scores)
    advantage = float(np.max(tpr - fpr))
    best_bal_acc = 0.0
    best_threshold = float(thresholds[0]) if len(thresholds) else 0.0
    for threshold in thresholds:
        pred = (scores >= threshold).astype(int)
        bal_acc = float(balanced_accuracy_score(labels, pred))
        if bal_acc > best_bal_acc:
            best_bal_acc = bal_acc
            best_threshold = float(threshold)
    summary = {
        "membership_auc": auc,
        "membership_advantage": advantage,
        "membership_best_balanced_accuracy": best_bal_acc,
        "median_distance_train_to_synthetic": float(np.median(train_to_syn)),
        "median_distance_test_to_synthetic": float(np.median(test_to_syn)),
        "best_membership_threshold": best_threshold,
    }
    curves = pd.DataFrame({
        "candidate_type": ["train"] * len(train_to_syn) + ["test"] * len(test_to_syn),
        "distance_to_nearest_synthetic": np.concatenate([train_to_syn, test_to_syn]),
        "membership_score": scores,
    })
    roc_df = pd.DataFrame({"roc_fpr": fpr, "roc_tpr": tpr, "roc_threshold": thresholds})
    return summary, curves, roc_df


def compute_privacy_metrics(X_train, X_test, X_syn, scaler):
    train_scaled = scaler.transform(X_train)
    test_scaled = scaler.transform(X_test)
    syn_scaled = scaler.transform(X_syn)
    nn_train_2 = NearestNeighbors(n_neighbors=2, metric="euclidean").fit(train_scaled)
    syn_distances = nn_train_2.kneighbors(syn_scaled, n_neighbors=2)[0]
    syn_dcr = syn_distances[:, 0]
    syn_nndr = syn_distances[:, 0] / np.maximum(syn_distances[:, 1], 1e-12)
    nn_train_1 = NearestNeighbors(n_neighbors=1, metric="euclidean").fit(train_scaled)
    test_dcr = nn_train_1.kneighbors(test_scaled, n_neighbors=1)[0][:, 0]
    train_hash = X_train.round(10).astype(str).agg("|".join, axis=1)
    syn_hash = X_syn.round(10).astype(str).agg("|".join, axis=1)
    p01 = float(np.quantile(test_dcr, 0.01))
    p05 = float(np.quantile(test_dcr, 0.05))
    mia_summary, mia_curves, roc_df = compute_membership_inference_metrics(X_train, X_test, X_syn, scaler)
    summary = {
        "exact_duplicate_rate": float(syn_hash.isin(train_hash).mean()),
        "median_dcr_synthetic_to_train": float(np.median(syn_dcr)),
        "median_dcr_test_to_train": float(np.median(test_dcr)),
        "mean_dcr_synthetic_to_train": float(np.mean(syn_dcr)),
        "mean_dcr_test_to_train": float(np.mean(test_dcr)),
        "median_nndr_synthetic": float(np.median(syn_nndr)),
        "share_synthetic_below_test_dcr_p01": float(np.mean(syn_dcr < p01)),
        "share_synthetic_below_test_dcr_p05": float(np.mean(syn_dcr < p05)),
        "test_dcr_p01": p01,
        "test_dcr_p05": p05,
        **mia_summary,
    }
    privacy_curves = pd.DataFrame({
        "synthetic_dcr": pd.Series(syn_dcr),
        "synthetic_nndr": pd.Series(syn_nndr),
        "test_dcr": pd.Series(test_dcr),
    })
    return summary, privacy_curves, mia_curves, roc_df


def run_split_experiment(X_train, X_test, y_train, y_test):
    scaler = StandardScaler().fit(X_train)
    models = fit_class_conditional_gmms(scaler.transform(X_train), y_train)
    X_syn, y_syn = sample_synthetic_dataset(models, scaler, X_train, y_train)
    fidelity_df, fidelity_summary = compute_fidelity_metrics(X_train, X_syn)
    utility_df, utility_feature_overlap = compute_utility_metrics(X_train, y_train, X_syn, y_syn, X_test, y_test)
    privacy_summary, privacy_curves, mia_curves, mia_roc = compute_privacy_metrics(X_train, X_test, X_syn, scaler)
    return {
        "scaler": scaler,
        "models": models,
        "X_syn": X_syn,
        "y_syn": y_syn,
        "fidelity_df": fidelity_df,
        "fidelity_summary": fidelity_summary,
        "utility_df": utility_df,
        "utility_feature_overlap": utility_feature_overlap,
        "privacy_summary": privacy_summary,
        "privacy_curves": privacy_curves,
        "membership_curves": mia_curves,
        "membership_roc": mia_roc,
    }


def aggregate_scalar_metrics(df: pd.DataFrame, metric_cols: List[str]) -> Dict[str, Dict[str, float]]:
    return {c: {"mean": float(df[c].mean()), "std": float(df[c].std(ddof=1)) if len(df) > 1 else 0.0} for c in metric_cols}


def run_cross_validation(X, y, n_splits: int):
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    fidelity_rows, utility_rows, privacy_rows, generator_rows = [], [], [], []
    for fold, (train_idx, test_idx) in enumerate(cv.split(X, y), start=1):
        X_train, X_test = X.iloc[train_idx].reset_index(drop=True), X.iloc[test_idx].reset_index(drop=True)
        y_train, y_test = y.iloc[train_idx].reset_index(drop=True), y.iloc[test_idx].reset_index(drop=True)
        results = run_split_experiment(X_train, X_test, y_train, y_test)
        fidelity_rows.append({"fold": fold, **results["fidelity_summary"]})
        privacy_rows.append({"fold": fold, **results["privacy_summary"]})
        utility_df = results["utility_df"].copy()
        utility_df.insert(0, "fold", fold)
        utility_rows.extend(utility_df.to_dict(orient="records"))
        generator_rows.append({"fold": fold, **{f"class_{cls}_components": int(meta["n_components"]) for cls, meta in results["models"].items()}, **{f"class_{cls}_covariance_full": int(meta["covariance_type"] == "full") for cls, meta in results["models"].items()}})
    fidelity_cv_df = pd.DataFrame(fidelity_rows)
    utility_cv_df = pd.DataFrame(utility_rows)
    privacy_cv_df = pd.DataFrame(privacy_rows)
    generator_cv_df = pd.DataFrame(generator_rows)
    utility_cv_summary_df = utility_cv_df.groupby(["model", "setting"], as_index=False).agg(
        accuracy_mean=("accuracy", "mean"), accuracy_std=("accuracy", "std"),
        f1_mean=("f1", "mean"), f1_std=("f1", "std"),
        roc_auc_mean=("roc_auc", "mean"), roc_auc_std=("roc_auc", "std")
    ).fillna(0.0)
    cv_summary = {
        "n_splits": n_splits,
        "fidelity": aggregate_scalar_metrics(fidelity_cv_df, [
            "avg_standardized_mean_diff", "avg_std_ratio_deviation", "avg_ks_statistic",
            "max_ks_statistic", "avg_normalized_wasserstein", "mean_abs_corr_gap",
        ]),
        "utility": utility_cv_summary_df.to_dict(orient="records"),
        "privacy": aggregate_scalar_metrics(privacy_cv_df, [
            "exact_duplicate_rate", "median_dcr_synthetic_to_train", "median_dcr_test_to_train",
            "mean_dcr_synthetic_to_train", "mean_dcr_test_to_train", "median_nndr_synthetic",
            "share_synthetic_below_test_dcr_p01", "share_synthetic_below_test_dcr_p05",
            "test_dcr_p01", "test_dcr_p05", "membership_auc", "membership_advantage",
            "membership_best_balanced_accuracy", "median_distance_train_to_synthetic", "median_distance_test_to_synthetic",
        ]),
        "generator_components": aggregate_scalar_metrics(generator_cv_df, [c for c in generator_cv_df.columns if c != "fold"]),
    }
    return fidelity_cv_df, utility_cv_df, privacy_cv_df, utility_cv_summary_df, generator_cv_df, cv_summary


def save_figures(X_train, y_train, X_syn, y_syn, fidelity_df, utility_df, privacy_curves, membership_curves, membership_roc, utility_cv_summary_df, figures_dir: Path) -> None:
    class_counts = pd.DataFrame({"Real train": y_train.value_counts().sort_index(), "Synthetic": y_syn.value_counts().sort_index()})
    class_counts.index = ["Malignant", "Benign"]
    ax = class_counts.plot(kind="bar", rot=0, figsize=(6.4, 4.2))
    ax.set_ylabel("Número de muestras")
    ax.set_title("Balance de clases")
    ax.grid(axis="y", alpha=0.25)
    plt.tight_layout(); plt.savefig(figures_dir / "class_balance.png", dpi=FIG_DPI); plt.close()

    selected = ["mean radius", "mean texture", "mean area", "worst concave points"]
    fig, axes = plt.subplots(2, 2, figsize=(9.2, 6.6)); axes = axes.ravel()
    for ax, feature in zip(axes, selected):
        ax.hist(X_train[feature], bins=24, alpha=0.55, density=True, label="Real", edgecolor="white")
        ax.hist(X_syn[feature], bins=24, alpha=0.55, density=True, label="Sintético", edgecolor="white")
        ax.set_title(feature); ax.grid(axis="y", alpha=0.2)
    axes[0].legend(frameon=False)
    fig.suptitle("Distribuciones marginales representativas", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96]); fig.savefig(figures_dir / "feature_distributions.png", dpi=FIG_DPI); plt.close(fig)

    corr_real, corr_syn = X_train.corr(numeric_only=True), X_syn.corr(numeric_only=True)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.6))
    im0 = axes[0].imshow(corr_real, vmin=-1, vmax=1); axes[0].set_title("Correlaciones reales"); axes[0].set_xticks([]); axes[0].set_yticks([])
    axes[1].imshow(corr_syn, vmin=-1, vmax=1); axes[1].set_title("Correlaciones sintéticas"); axes[1].set_xticks([]); axes[1].set_yticks([])
    cbar = fig.colorbar(im0, ax=axes.ravel().tolist(), shrink=0.82); cbar.set_label("r de Pearson")
    fig.tight_layout(); fig.savefig(figures_dir / "correlation_heatmaps.png", dpi=FIG_DPI); plt.close(fig)

    ax = utility_df.pivot(index="model", columns="setting", values="roc_auc").plot(kind="bar", rot=0, figsize=(6.8, 4.2))
    ax.set_ylim(0.85, 1.0); ax.set_ylabel("ROC-AUC en test real"); ax.set_title("Utilidad predictiva (holdout: TRTR vs TSTR)"); ax.grid(axis="y", alpha=0.25)
    plt.tight_layout(); plt.savefig(figures_dir / "utility_roc_auc.png", dpi=FIG_DPI); plt.close()

    cv_pivot = utility_cv_summary_df.pivot(index="model", columns="setting", values="roc_auc_mean")
    cv_err = utility_cv_summary_df.pivot(index="model", columns="setting", values="roc_auc_std")
    ax = cv_pivot.plot(kind="bar", rot=0, figsize=(7.2, 4.4), yerr=cv_err)
    ax.set_ylim(0.85, 1.0); ax.set_ylabel("ROC-AUC medio en test real"); ax.set_title("Utilidad predictiva con validación cruzada"); ax.grid(axis="y", alpha=0.25)
    plt.tight_layout(); plt.savefig(figures_dir / "utility_roc_auc_cv.png", dpi=FIG_DPI); plt.close()

    fig, axes = plt.subplots(1, 2, figsize=(9.0, 4.2))
    axes[0].hist(privacy_curves["test_dcr"].dropna(), bins=20, alpha=0.65, label="Test real → train")
    axes[0].hist(privacy_curves["synthetic_dcr"].dropna(), bins=20, alpha=0.65, label="Sintético → train")
    axes[0].set_title("DCR"); axes[0].set_xlabel("Distancia euclídea"); axes[0].grid(axis="y", alpha=0.2); axes[0].legend(frameon=False)
    axes[1].hist(privacy_curves["synthetic_nndr"].dropna(), bins=20, alpha=0.75)
    axes[1].set_title("NNDR (sintético)"); axes[1].set_xlabel("d1 / d2"); axes[1].grid(axis="y", alpha=0.2)
    fig.tight_layout(); fig.savefig(figures_dir / "privacy_dcr_nndr.png", dpi=FIG_DPI); plt.close(fig)

    train_dist = membership_curves.loc[membership_curves["candidate_type"] == "train", "distance_to_nearest_synthetic"]
    test_dist = membership_curves.loc[membership_curves["candidate_type"] == "test", "distance_to_nearest_synthetic"]
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.2))
    axes[0].hist(train_dist, bins=20, alpha=0.65, label="Train real → sintético")
    axes[0].hist(test_dist, bins=20, alpha=0.65, label="Test real → sintético")
    axes[0].set_title("Ataque de membership: distancias")
    axes[0].set_xlabel("Distancia al sintético más cercano")
    axes[0].grid(axis="y", alpha=0.2); axes[0].legend(frameon=False)
    axes[1].plot(membership_roc["roc_fpr"], membership_roc["roc_tpr"], label="ROC del atacante")
    axes[1].plot([0, 1], [0, 1], linestyle="--", linewidth=1)
    axes[1].set_title("Ataque de membership: curva ROC"); axes[1].set_xlabel("FPR"); axes[1].set_ylabel("TPR"); axes[1].grid(alpha=0.2); axes[1].legend(frameon=False)
    fig.tight_layout(); fig.savefig(figures_dir / "privacy_membership_attack.png", dpi=FIG_DPI); plt.close(fig)

    top = fidelity_df.sort_values("ks_statistic", ascending=False).head(8)[["feature", "ks_statistic", "normalized_wasserstein", "abs_mean_diff_std_units"]].copy()
    top.columns = ["Variable", "KS", "Wass. norm.", "Dif. media (z)"]
    for col in ["KS", "Wass. norm.", "Dif. media (z)"]:
        top[col] = top[col].map(lambda x: f"{x:.3f}")
    fig, ax = plt.subplots(figsize=(8.5, 2.8)); ax.axis("off")
    table = ax.table(cellText=top.values, colLabels=top.columns, loc="center", cellLoc="left")
    table.auto_set_font_size(False); table.set_fontsize(8.5); table.scale(1, 1.4)
    ax.set_title("Variables con mayor divergencia univariante", pad=12)
    fig.tight_layout(); fig.savefig(figures_dir / "fidelity_table.png", dpi=FIG_DPI); plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Entrena un generador de datos sintéticos y evalúa fidelidad, utilidad y privacidad.")
    parser.add_argument("--base-dir", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--cv-folds", type=int, default=5)
    args = parser.parse_args()
    if args.cv_folds < 2:
        raise ValueError("--cv-folds debe ser al menos 2")
    dirs = ensure_dirs(args.base_dir)
    X, y, target_names = load_dataset()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE)
    holdout = run_split_experiment(X_train, X_test, y_train, y_test)
    fidelity_cv_df, utility_cv_df, privacy_cv_df, utility_cv_summary_df, generator_cv_df, cv_summary = run_cross_validation(X, y, args.cv_folds)
    save_figures(
        X_train, y_train, holdout["X_syn"], holdout["y_syn"], holdout["fidelity_df"], holdout["utility_df"], holdout["privacy_curves"], holdout["membership_curves"], holdout["membership_roc"], utility_cv_summary_df, dirs["figures"]
    )
    holdout["fidelity_df"].to_csv(dirs["outputs"] / "fidelity_metrics_by_feature.csv", index=False)
    holdout["utility_df"].to_csv(dirs["outputs"] / "utility_metrics.csv", index=False)
    holdout["privacy_curves"].to_csv(dirs["outputs"] / "privacy_curves.csv", index=False)
    holdout["membership_curves"].to_csv(dirs["outputs"] / "membership_attack_curves.csv", index=False)
    holdout["membership_roc"].to_csv(dirs["outputs"] / "membership_attack_roc.csv", index=False)
    holdout["X_syn"].assign(target=holdout["y_syn"]).to_csv(dirs["outputs"] / "synthetic_dataset.csv", index=False)
    fidelity_cv_df.to_csv(dirs["outputs"] / "cv_fidelity_summary_by_fold.csv", index=False)
    utility_cv_df.to_csv(dirs["outputs"] / "cv_utility_metrics_by_fold.csv", index=False)
    utility_cv_summary_df.to_csv(dirs["outputs"] / "cv_utility_metrics_summary.csv", index=False)
    privacy_cv_df.to_csv(dirs["outputs"] / "cv_privacy_summary_by_fold.csv", index=False)
    generator_cv_df.to_csv(dirs["outputs"] / "cv_generator_components_by_fold.csv", index=False)
    summary = {
        "dataset": {
            "name": "Breast Cancer Wisconsin (Diagnostic)",
            "n_rows_total": int(len(X)),
            "n_features": int(X.shape[1]),
            "n_rows_train": int(len(X_train)),
            "n_rows_test": int(len(X_test)),
            "class_mapping": target_names,
        },
        "generator": {
            "strategy": "Class-conditional Gaussian Mixture Models (GMM) selected by BIC",
            "library": "scikit-learn",
            "candidate_covariance_types": list(GMM_COVARIANCE_TYPES),
            "per_class_components": {str(k): int(v["n_components"]) for k, v in holdout["models"].items()},
            "per_class_covariance_type": {str(k): str(v["covariance_type"]) for k, v in holdout["models"].items()},
            "gmm_n_init": GMM_N_INIT,
            "max_components_tested": GMM_MAX_COMPONENTS,
            "random_state": RANDOM_STATE,
        },
        "fidelity": holdout["fidelity_summary"],
        "utility": {"metrics": holdout["utility_df"].to_dict(orient="records"), **holdout["utility_feature_overlap"]},
        "privacy": holdout["privacy_summary"],
        "cross_validation": cv_summary,
    }
    with open(dirs["outputs"] / "summary_metrics.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
