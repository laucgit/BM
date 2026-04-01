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
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

RANDOM_STATE = 42
FIG_DPI = 180


def ensure_dirs(base_dir: Path) -> Dict[str, Path]:
    outputs = base_dir / "outputs"
    figures = outputs / "figures"
    outputs.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    return {"base": base_dir, "outputs": outputs, "figures": figures}



def load_dataset() -> Tuple[pd.DataFrame, pd.Series, Dict[int, str]]:
    dataset = load_breast_cancer(as_frame=True)
    X = dataset.data.copy()
    y = dataset.target.copy()
    target_names = {0: "malignant", 1: "benign"}
    return X, y, target_names



def fit_class_conditional_gmms(
    X_train_scaled: np.ndarray,
    y_train: pd.Series,
    max_components: int = 4,
) -> Dict[int, Dict[str, object]]:
    models: Dict[int, Dict[str, object]] = {}

    for cls in sorted(y_train.unique()):
        X_cls = X_train_scaled[y_train.values == cls]
        best_bic = np.inf
        best_model = None
        best_k = None

        for n_components in range(1, max_components + 1):
            model = GaussianMixture(
                n_components=n_components,
                covariance_type="diag",
                reg_covar=1e-5,
                random_state=RANDOM_STATE,
                n_init=2,
                max_iter=200,
            )
            model.fit(X_cls)
            bic = model.bic(X_cls)

            if bic < best_bic:
                best_bic = bic
                best_model = model
                best_k = n_components

        models[int(cls)] = {
            "model": best_model,
            "n_components": int(best_k),
            "bic": float(best_bic),
        }

    return models



def sample_synthetic_dataset(
    models: Dict[int, Dict[str, object]],
    scaler: StandardScaler,
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> Tuple[pd.DataFrame, pd.Series]:
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

    X_syn = pd.DataFrame(
        scaler.inverse_transform(X_syn_scaled),
        columns=feature_names,
    )

    # Clipping keeps generated values within the empirical range observed in the
    # training set and avoids implausible negatives for biomedical measurements.
    X_syn = X_syn.clip(lower=X_train.min(), upper=X_train.max(), axis=1)
    return X_syn, y_syn



def compute_fidelity_metrics(
    X_train: pd.DataFrame,
    X_syn: pd.DataFrame,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    rows = []
    ks_values = []
    normalized_wd_values = []
    standardized_mean_diffs = []
    std_ratio_devs = []

    for col in X_train.columns:
        real = X_train[col].to_numpy()
        syn = X_syn[col].to_numpy()
        real_mean = float(real.mean())
        syn_mean = float(syn.mean())
        real_std = float(real.std(ddof=1))
        syn_std = float(syn.std(ddof=1))
        ks = float(ks_2samp(real, syn).statistic)
        wd = float(wasserstein_distance(real, syn))

        rows.append(
            {
                "feature": col,
                "real_mean": real_mean,
                "synthetic_mean": syn_mean,
                "real_std": real_std,
                "synthetic_std": syn_std,
                "abs_mean_diff_std_units": abs(syn_mean - real_mean) / real_std if real_std else 0.0,
                "ks_statistic": ks,
                "wasserstein_distance": wd,
                "normalized_wasserstein": wd / real_std if real_std else 0.0,
            }
        )

        ks_values.append(ks)
        normalized_wd_values.append(wd / real_std if real_std else 0.0)
        standardized_mean_diffs.append(abs(syn_mean - real_mean) / real_std if real_std else 0.0)
        std_ratio_devs.append(abs(syn_std / real_std - 1.0) if real_std else 0.0)

    fidelity_df = pd.DataFrame(rows)

    corr_real = X_train.corr(numeric_only=True).values
    corr_syn = X_syn.corr(numeric_only=True).values
    upper_mask = np.triu(np.ones_like(corr_real, dtype=bool), k=1)
    mean_abs_corr_gap = float(np.abs(corr_real - corr_syn)[upper_mask].mean())

    summary = {
        "avg_standardized_mean_diff": float(np.mean(standardized_mean_diffs)),
        "avg_std_ratio_deviation": float(np.mean(std_ratio_devs)),
        "avg_ks_statistic": float(np.mean(ks_values)),
        "max_ks_statistic": float(np.max(ks_values)),
        "avg_normalized_wasserstein": float(np.mean(normalized_wd_values)),
        "mean_abs_corr_gap": mean_abs_corr_gap,
    }
    return fidelity_df, summary



def compute_utility_metrics(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_syn: pd.DataFrame,
    y_syn: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> Tuple[pd.DataFrame, Dict[str, List[str]]]:
    models = {
        "Logistic Regression": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", LogisticRegression(max_iter=5000, random_state=RANDOM_STATE)),
            ]
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=500,
            min_samples_leaf=2,
            random_state=RANDOM_STATE,
        ),
    }

    rows = []
    feature_overlap = {}

    for model_name, model in models.items():
        real_model = clone(model)
        real_model.fit(X_train, y_train)
        pred_real = real_model.predict(X_test)
        proba_real = real_model.predict_proba(X_test)[:, 1]
        rows.append(
            {
                "model": model_name,
                "setting": "TRTR",
                "accuracy": float(accuracy_score(y_test, pred_real)),
                "f1": float(f1_score(y_test, pred_real)),
                "roc_auc": float(roc_auc_score(y_test, proba_real)),
            }
        )

        syn_model = clone(model)
        syn_model.fit(X_syn, y_syn)
        pred_syn = syn_model.predict(X_test)
        proba_syn = syn_model.predict_proba(X_test)[:, 1]
        rows.append(
            {
                "model": model_name,
                "setting": "TSTR",
                "accuracy": float(accuracy_score(y_test, pred_syn)),
                "f1": float(f1_score(y_test, pred_syn)),
                "roc_auc": float(roc_auc_score(y_test, proba_syn)),
            }
        )

        if model_name == "Random Forest":
            real_imp = pd.Series(real_model.feature_importances_, index=X_train.columns).sort_values(ascending=False)
            syn_imp = pd.Series(syn_model.feature_importances_, index=X_train.columns).sort_values(ascending=False)
            feature_overlap = {
                "rf_top10_real": real_imp.head(10).index.tolist(),
                "rf_top10_synthetic": syn_imp.head(10).index.tolist(),
                "rf_top10_overlap_count": int(len(set(real_imp.head(10).index) & set(syn_imp.head(10).index))),
            }

    utility_df = pd.DataFrame(rows)
    return utility_df, feature_overlap



def compute_privacy_metrics(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    X_syn: pd.DataFrame,
    scaler: StandardScaler,
) -> Tuple[Dict[str, float], pd.DataFrame]:
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
    exact_duplicate_rate = float(syn_hash.isin(train_hash).mean())

    p01 = float(np.quantile(test_dcr, 0.01))
    p05 = float(np.quantile(test_dcr, 0.05))

    summary = {
        "exact_duplicate_rate": exact_duplicate_rate,
        "median_dcr_synthetic_to_train": float(np.median(syn_dcr)),
        "median_dcr_test_to_train": float(np.median(test_dcr)),
        "mean_dcr_synthetic_to_train": float(np.mean(syn_dcr)),
        "mean_dcr_test_to_train": float(np.mean(test_dcr)),
        "median_nndr_synthetic": float(np.median(syn_nndr)),
        "share_synthetic_below_test_dcr_p01": float(np.mean(syn_dcr < p01)),
        "share_synthetic_below_test_dcr_p05": float(np.mean(syn_dcr < p05)),
        "test_dcr_p01": p01,
        "test_dcr_p05": p05,
    }

    privacy_curves = pd.DataFrame(
        {
            "synthetic_dcr": pd.Series(syn_dcr),
            "synthetic_nndr": pd.Series(syn_nndr),
        }
    )
    privacy_curves["test_dcr"] = pd.Series(test_dcr)

    return summary, privacy_curves



def save_figures(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_syn: pd.DataFrame,
    y_syn: pd.Series,
    fidelity_df: pd.DataFrame,
    utility_df: pd.DataFrame,
    privacy_curves: pd.DataFrame,
    figures_dir: Path,
) -> None:
    # Class balance
    class_counts = pd.DataFrame(
        {
            "Real train": y_train.value_counts().sort_index(),
            "Synthetic": y_syn.value_counts().sort_index(),
        }
    )
    class_counts.index = ["Malignant", "Benign"]
    ax = class_counts.plot(kind="bar", rot=0, figsize=(6.4, 4.2))
    ax.set_ylabel("Número de muestras")
    ax.set_title("Balance de clases")
    ax.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(figures_dir / "class_balance.png", dpi=FIG_DPI)
    plt.close()

    # Distributions for four representative features
    selected_features = ["mean radius", "mean texture", "mean area", "worst concave points"]
    fig, axes = plt.subplots(2, 2, figsize=(9.2, 6.6))
    axes = axes.ravel()
    for ax, feature in zip(axes, selected_features):
        ax.hist(X_train[feature], bins=24, alpha=0.55, density=True, label="Real", edgecolor="white")
        ax.hist(X_syn[feature], bins=24, alpha=0.55, density=True, label="Sintético", edgecolor="white")
        ax.set_title(feature)
        ax.grid(axis="y", alpha=0.2)
    axes[0].legend(frameon=False)
    fig.suptitle("Distribuciones marginales representativas", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(figures_dir / "feature_distributions.png", dpi=FIG_DPI)
    plt.close(fig)

    # Correlation comparison
    corr_real = X_train.corr(numeric_only=True)
    corr_syn = X_syn.corr(numeric_only=True)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.6))
    im0 = axes[0].imshow(corr_real, vmin=-1, vmax=1)
    axes[0].set_title("Correlaciones reales")
    axes[0].set_xticks([])
    axes[0].set_yticks([])
    axes[1].imshow(corr_syn, vmin=-1, vmax=1)
    axes[1].set_title("Correlaciones sintéticas")
    axes[1].set_xticks([])
    axes[1].set_yticks([])
    cbar = fig.colorbar(im0, ax=axes.ravel().tolist(), shrink=0.82)
    cbar.set_label("r de Pearson")
    fig.tight_layout()
    fig.savefig(figures_dir / "correlation_heatmaps.png", dpi=FIG_DPI)
    plt.close(fig)

    # Utility figure
    pivot = utility_df.pivot(index="model", columns="setting", values="roc_auc")
    ax = pivot.plot(kind="bar", rot=0, figsize=(6.8, 4.2))
    ax.set_ylim(0.85, 1.0)
    ax.set_ylabel("ROC-AUC en test real")
    ax.set_title("Utilidad predictiva (TRTR vs TSTR)")
    ax.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(figures_dir / "utility_roc_auc.png", dpi=FIG_DPI)
    plt.close()

    # Privacy figure: DCR distribution comparison
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 4.2))
    axes[0].hist(privacy_curves["test_dcr"].dropna(), bins=20, alpha=0.65, label="Test real → train")
    axes[0].hist(privacy_curves["synthetic_dcr"].dropna(), bins=20, alpha=0.65, label="Sintético → train")
    axes[0].set_title("DCR")
    axes[0].set_xlabel("Distancia euclídea")
    axes[0].grid(axis="y", alpha=0.2)
    axes[0].legend(frameon=False)

    axes[1].hist(privacy_curves["synthetic_nndr"].dropna(), bins=20, alpha=0.75)
    axes[1].set_title("NNDR (sintético)")
    axes[1].set_xlabel("d1 / d2")
    axes[1].grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(figures_dir / "privacy_dcr_nndr.png", dpi=FIG_DPI)
    plt.close(fig)

    # Fidelity ranking table figure (top 8 by KS)
    top = fidelity_df.sort_values("ks_statistic", ascending=False).head(8)[[
        "feature", "ks_statistic", "normalized_wasserstein", "abs_mean_diff_std_units"
    ]].copy()
    top.columns = ["Variable", "KS", "Wass. norm.", "Dif. media (z)"]
    top["KS"] = top["KS"].map(lambda x: f"{x:.3f}")
    top["Wass. norm."] = top["Wass. norm."].map(lambda x: f"{x:.3f}")
    top["Dif. media (z)"] = top["Dif. media (z)"].map(lambda x: f"{x:.3f}")

    fig, ax = plt.subplots(figsize=(8.5, 2.8))
    ax.axis("off")
    table = ax.table(cellText=top.values, colLabels=top.columns, loc="center", cellLoc="left")
    table.auto_set_font_size(False)
    table.set_fontsize(8.5)
    table.scale(1, 1.4)
    ax.set_title("Variables con mayor divergencia univariante", pad=12)
    fig.tight_layout()
    fig.savefig(figures_dir / "fidelity_table.png", dpi=FIG_DPI)
    plt.close(fig)



def main() -> None:
    parser = argparse.ArgumentParser(description="Train a synthetic data generator and evaluate fidelity, utility and privacy.")
    parser.add_argument("--base-dir", type=Path, default=Path(__file__).resolve().parents[1], help="Project base directory")
    args = parser.parse_args()

    dirs = ensure_dirs(args.base_dir)
    X, y, target_names = load_dataset()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=RANDOM_STATE,
    )

    scaler = StandardScaler().fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    models = fit_class_conditional_gmms(X_train_scaled, y_train)
    X_syn, y_syn = sample_synthetic_dataset(models, scaler, X_train, y_train)

    fidelity_df, fidelity_summary = compute_fidelity_metrics(X_train, X_syn)
    utility_df, utility_feature_overlap = compute_utility_metrics(X_train, y_train, X_syn, y_syn, X_test, y_test)
    privacy_summary, privacy_curves = compute_privacy_metrics(X_train, X_test, X_syn, scaler)

    save_figures(X_train, y_train, X_syn, y_syn, fidelity_df, utility_df, privacy_curves, dirs["figures"])

    fidelity_df.to_csv(dirs["outputs"] / "fidelity_metrics_by_feature.csv", index=False)
    utility_df.to_csv(dirs["outputs"] / "utility_metrics.csv", index=False)
    privacy_curves.to_csv(dirs["outputs"] / "privacy_curves.csv", index=False)
    X_syn.assign(target=y_syn).to_csv(dirs["outputs"] / "synthetic_dataset.csv", index=False)

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
            "strategy": "Class-conditional Gaussian Mixture Models (GMM)",
            "library": "scikit-learn",
            "covariance_type": "diag",
            "per_class_components": {str(k): int(v["n_components"]) for k, v in models.items()},
            "random_state": RANDOM_STATE,
        },
        "fidelity": fidelity_summary,
        "utility": {
            "metrics": utility_df.to_dict(orient="records"),
            **utility_feature_overlap,
        },
        "privacy": privacy_summary,
    }

    with open(dirs["outputs"] / "summary_metrics.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
