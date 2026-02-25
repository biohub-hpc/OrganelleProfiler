"""
Phenotypic Activity Assessment using copairs mAP (mean Average Precision).

Provides functions to compute mAP-based phenotypic activity, distinctiveness,
and consistency metrics. Used by the CP Challenge stage for head-to-head
comparison of Cell Painting vs live-cell features.

Based on: https://github.com/cytomining/copairs/blob/v0.5.1/examples/phenotypic_activity.ipynb
"""

import ast
import yaml
import numpy as np
import pandas as pd
import anndata as ad
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from tqdm import tqdm
from joblib import Parallel, delayed

import matplotlib.pyplot as plt
import logging

logger = logging.getLogger(__name__)


def _compute_single_complex_map(
    members, label, label_col, meta_base, feats_array,
):
    """
    Compute copairs mAP for a single complex/cluster.

    Designed to be called in parallel via joblib. The feats_array is read-only
    shared memory; only the lightweight meta_c is copied per call.

    Returns (label, complex_map_df) or None on failure.
    """
    from copairs import map as copairs_map

    meta_c = meta_base.copy()
    meta_c["in_complex"] = meta_c["perturbation"].isin(members)
    try:
        results_complex = copairs_map.average_precision(
            meta_c,
            feats_array,
            pos_sameby=["in_complex"],
            pos_diffby=["perturbation"],
            neg_sameby=[],
            neg_diffby=["in_complex"],
        )
        map_result = copairs_map.mean_average_precision(
            results_complex,
            sameby=["in_complex"],
            null_size=1000000,
            threshold=0.05,
            seed=0,
        )
        complex_map = map_result[map_result["in_complex"] == True].copy()
        complex_map[label_col] = label
        complex_map.drop(columns=["in_complex", "indices"], inplace=True, errors="ignore")
        return complex_map
    except Exception:
        return None


def adata_to_copairs_df(adata: ad.AnnData) -> pd.DataFrame:
    """
    Convert AnnData to DataFrame format expected by copairs.

    Parameters
    ----------
    adata : AnnData
        AnnData object with features in .X and metadata in .obs

    Returns
    -------
    df : DataFrame
        DataFrame with metadata and feature columns
    """
    if hasattr(adata.X, "toarray"):
        features_df = pd.DataFrame(
            adata.X.toarray(), index=adata.obs_names, columns=adata.var_names
        )
    else:
        features_df = pd.DataFrame(
            adata.X, index=adata.obs_names, columns=adata.var_names
        )

    df = pd.concat([adata.obs.reset_index(drop=True), features_df.reset_index(drop=True)], axis=1)
    return df


def compute_auc_score(map_df: pd.DataFrame, p_floor: float = 1e-6) -> float:
    """
    Compute significance-weighted mean mAP (AUC score).

    Each gene's contribution = mAP_i * w_i, where w_i is its -log10(p-value)
    normalized to [0, 1]. The overall score is the mean across all genes,
    giving a continuous, threshold-free measure of phenotypic signal strength.

    Parameters
    ----------
    map_df : DataFrame
        Copairs mAP output with 'mean_average_precision' and 'corrected_p_value'.
    p_floor : float
        Minimum p-value clamp to avoid infinite -log10. Default 1e-6.

    Returns
    -------
    float
        Weighted mAP score, approximately bounded [0, 1].
    """
    if len(map_df) < 3:
        logger.warning(f"  AUC score: too few entities ({len(map_df)}), returning NaN")
        return float("nan")

    mAP = map_df["mean_average_precision"].values.astype(np.float64)
    p = np.clip(map_df["corrected_p_value"].values.astype(np.float64), p_floor, 1.0)
    neg_log_p = -np.log10(p)
    w = neg_log_p / -np.log10(p_floor)  # normalize to [0, 1]
    return float(np.nanmean(mAP * w))


def compute_threshold_sweep_auc(
    map_df: pd.DataFrame, n_thresholds: int = 100, p_floor: float = 1e-6
) -> float:
    """
    Compute threshold-sweep AUC: integrate active_ratio * mean_mAP over
    a range of p-value thresholds.

    Generalizes the current single-threshold approach. The existing
    active_ratio at p=0.05 is a single point on this curve.

    Parameters
    ----------
    map_df : DataFrame
        Copairs mAP output with 'mean_average_precision' and 'corrected_p_value'.
    n_thresholds : int
        Number of log-spaced thresholds to sweep (1.0 down to p_floor).
    p_floor : float
        Minimum threshold in the sweep. Default 1e-6.

    Returns
    -------
    float
        Integrated AUC score.
    """
    if len(map_df) < 3:
        logger.warning(f"  Sweep AUC: too few entities ({len(map_df)}), returning NaN")
        return float("nan")

    mAP = map_df["mean_average_precision"].values.astype(np.float64)
    p = map_df["corrected_p_value"].values.astype(np.float64)
    n = len(map_df)

    # Log-spaced thresholds from 1.0 down to p_floor
    thresholds = np.logspace(0, np.log10(p_floor), n_thresholds)

    x_vals = -np.log10(thresholds)
    y_vals = np.empty(n_thresholds)

    for i, t in enumerate(thresholds):
        mask = p < t
        n_active = mask.sum()
        if n_active == 0:
            y_vals[i] = 0.0
        else:
            y_vals[i] = (n_active / n) * np.nanmean(mAP[mask])

    # Normalize x to [0, 1] for comparable AUC across experiments
    x_norm = x_vals / x_vals.max()
    return float(np.trapz(y_vals, x_norm))


def phenotypic_activity_assesment(
    adata: ad.AnnData, plot_results: bool = True
) -> Tuple[pd.DataFrame, float]:
    """
    Compute phenotypic activity via copairs mAP.

    Positive pairs: same perturbation, different sgRNA.
    Negative pairs: different NTC status.

    Parameters
    ----------
    adata : AnnData
        Guide-level AnnData with 'perturbation', 'sgRNA', 'n_cells' in .obs.
    plot_results : bool
        Whether to display a scatter plot.

    Returns
    -------
    activity_map : DataFrame
        Per-perturbation mAP results with 'below_corrected_p' column.
    active_ratio : float
        Fraction of perturbations that are phenotypically active.
    """
    from copairs import map as copairs_map

    df = adata_to_copairs_df(adata)
    df["is_NTC"] = df["perturbation"].apply(lambda x: x == "NTC")
    meta_cols = ["sgRNA", "n_cells", "perturbation", "is_NTC"]
    feat_cols = [col for col in df.columns if col not in meta_cols]
    meta = df[meta_cols]
    feats = df[feat_cols]

    results = copairs_map.average_precision(
        meta,
        np.asarray(feats),
        pos_sameby=["perturbation"],
        pos_diffby=["sgRNA"],
        neg_sameby=[],
        neg_diffby=["is_NTC"],
    )
    activity_map = copairs_map.mean_average_precision(
        results, sameby=["perturbation"], null_size=1000000, threshold=0.05, seed=0
    )
    activity_map["-log10(p-value)"] = -activity_map["corrected_p_value"].apply(np.log10)
    active_ratio = activity_map.below_corrected_p.mean()

    if plot_results:
        plt.scatter(
            data=activity_map,
            x="mean_average_precision",
            y="-log10(p-value)",
            c="below_corrected_p",
            cmap="tab10",
            s=10,
        )
        plt.title("Phenotypic activity assessment")
        plt.xlabel("mAP")
        plt.ylabel("-log10(p-value)")
        plt.axhline(-np.log10(0.05), color="black", linestyle="--")
        plt.text(
            0.65,
            1.5,
            f"Phenotypically active = {100 * active_ratio:.2f}%",
            va="center",
            ha="left",
        )
        plt.show()

    return activity_map, active_ratio


def phenotypic_distinctivness(
    adata: ad.AnnData, activity_map: pd.DataFrame, plot_results: bool = True
) -> Tuple[pd.DataFrame, float]:
    """
    Compute phenotypic distinctiveness among active perturbations.

    Parameters
    ----------
    adata : AnnData
        Guide-level AnnData.
    activity_map : DataFrame
        Output from phenotypic_activity_assesment.
    plot_results : bool
        Whether to display a scatter plot.

    Returns
    -------
    distinctiveness_map : DataFrame
    distinctive_ratio : float
    """
    from copairs import map as copairs_map

    active_perturbations = activity_map[activity_map["below_corrected_p"] == True][
        "perturbation"
    ].tolist()
    logger.info(f"Number of active perturbations: {len(active_perturbations)}")

    adata_filtered = adata[adata.obs["perturbation"].isin(active_perturbations)].copy()
    logger.info(
        f"Filtered to {adata_filtered.n_obs} observations from {len(active_perturbations)} active perturbations"
    )

    df = adata_to_copairs_df(adata_filtered)
    meta_cols = ["sgRNA", "n_cells", "perturbation"]
    meta_cols = [col for col in meta_cols if col in df.columns]
    feat_cols = [col for col in df.columns if col not in meta_cols]

    meta = df[meta_cols]
    feats = df[feat_cols]

    results = copairs_map.average_precision(
        meta,
        np.asarray(feats),
        pos_sameby=["perturbation"],
        pos_diffby=["sgRNA"],
        neg_sameby=[],
        neg_diffby=["perturbation"],
    )

    distinctiveness_map = copairs_map.mean_average_precision(
        results, sameby=["perturbation"], null_size=1000000, threshold=0.05, seed=0
    )
    distinctiveness_map["-log10(p-value)"] = -distinctiveness_map[
        "corrected_p_value"
    ].apply(np.log10)
    distinctive_ratio = distinctiveness_map.below_corrected_p.mean()
    logger.info(
        f"Proportion of phenotypically distinctive perturbations: {100 * distinctive_ratio:.2f}%"
    )

    if plot_results:
        plt.scatter(
            data=distinctiveness_map,
            x="mean_average_precision",
            y="-log10(p-value)",
            c="below_corrected_p",
            cmap="tab10",
            s=10,
        )
        plt.title("Phenotypic distinctiveness")
        plt.xlabel("mAP")
        plt.ylabel("-log10(p-value)")
        plt.axhline(-np.log10(0.05), color="black", linestyle="--")
        plt.text(
            0.65,
            1.5,
            f"Phenotypically distinct = {100 * distinctive_ratio:.2f}%",
            va="center",
            ha="left",
        )
        plt.show()

    return distinctiveness_map, distinctive_ratio


def phenotypic_consistency_corum(
    adata: ad.AnnData, activity_map: pd.DataFrame, plot_results: bool = True
) -> Tuple[pd.DataFrame, float]:
    """
    Compute phenotypic consistency using CORUM protein complex annotations.

    Parameters
    ----------
    adata : AnnData
        Gene-level AnnData.
    activity_map : DataFrame
        Output from phenotypic_activity_assesment.
    plot_results : bool
        Whether to display a scatter plot.

    Returns
    -------
    all_complex_results_df : DataFrame
    consistency_corum_ratio : float
    """
    from copairs import map as copairs_map

    path = "/hpc/projects/intracellular_dashboard/ops/configs/annotated_gene_panel_July2025.csv"
    gene_panel = pd.read_csv(path)

    active_genes = activity_map[activity_map["below_corrected_p"]][
        "perturbation"
    ].tolist()
    active_genes = [gene for gene in active_genes if gene != "NTC"]
    df = adata_to_copairs_df(adata)
    df = df[df["perturbation"].isin(active_genes)]

    # Build unique complexes: deduplicate so each complex is computed once,
    # not once per member gene. Key = frozenset of active members.
    active_genes_set = set(active_genes)
    seen_complexes: dict = {}  # frozenset(members) -> representative gene label
    for p in active_genes:
        complex_col = gene_panel.loc[gene_panel["Gene.name"] == p, "In_same_complex_with"]
        if complex_col.empty:
            continue
        raw = complex_col.iloc[0]
        raw_list = ast.literal_eval(raw) if isinstance(raw, str) and raw.strip() else []
        members = frozenset(g for g in (raw_list + [p]) if g in active_genes_set)
        if len(members) > 1 and members not in seen_complexes:
            seen_complexes[members] = p  # first gene encountered is the label

    n_complexes = len(seen_complexes)
    logger.info(f"  {len(active_genes)} active genes → {n_complexes} unique CORUM complexes")

    # Precompute base meta + features once
    static_meta_cols = ["perturbation", "n_cells", "guides", "reporter",
                        "experiment", "is_NTC", "n_experiments"]
    static_meta_cols = [c for c in static_meta_cols if c in df.columns]
    feature_cols = [c for c in df.columns if c not in static_meta_cols]
    meta_base = df[static_meta_cols].reset_index(drop=True)
    feats_array = np.asarray(df[feature_cols], dtype=np.float32)

    # Determine optimal workers for parallel complex computation
    try:
        from ops_utils.hpc.resource_manager import get_optimal_workers
        n_workers = get_optimal_workers(use_gpu=False, model_ram_gb=0.5, data_ram_gb=0.5, verbose=False)
    except Exception:
        n_workers = 4
    n_workers = min(n_workers, n_complexes)
    logger.info(f"  Using {n_workers} workers for CORUM complex computation")

    results = Parallel(n_jobs=n_workers, prefer="processes")(
        delayed(_compute_single_complex_map)(
            members, label, "complex_id", meta_base, feats_array,
        )
        for members, label in tqdm(seen_complexes.items(), total=n_complexes, desc="CORUM complexes")
    )
    complex_map_list = [r for r in results if r is not None]

    all_complex_results_df = pd.concat(complex_map_list, ignore_index=True)
    all_complex_results_df["-log10(p-value)"] = -all_complex_results_df[
        "corrected_p_value"
    ].apply(np.log10)

    consistency_corum_ratio = all_complex_results_df.below_corrected_p.mean()
    if plot_results:
        plt.scatter(
            data=all_complex_results_df,
            x="mean_average_precision",
            y="-log10(p-value)",
            c="below_corrected_p",
            cmap="tab10",
            s=10,
        )
        plt.title("Phenotypic consistency (CORUM)")
        plt.xlabel("mAP")
        plt.ylabel("-log10(p-value)")
        plt.axhline(-np.log10(0.05), color="black", linestyle="--")
        plt.text(
            0.65,
            1.5,
            f"Phenotypically distinct = {100 * consistency_corum_ratio:.2f}%",
            va="center",
            ha="left",
        )
        plt.show()

    return all_complex_results_df, consistency_corum_ratio


def phenotypic_consistency_manual_annotation(
    adata: ad.AnnData, activity_map: pd.DataFrame, plot_results: bool = True
) -> Tuple[pd.DataFrame, float]:
    """
    Compute phenotypic consistency using manual gene cluster annotations.

    Parameters
    ----------
    adata : AnnData
        Gene-level AnnData.
    activity_map : DataFrame
        Output from phenotypic_activity_assesment.
    plot_results : bool
        Whether to display a scatter plot.

    Returns
    -------
    all_complex_results_df : DataFrame
    phenotypic_consistency_ratio : float
    """
    from copairs import map as copairs_map

    path = "/hpc/projects/icd.ops/configs/gene_clusters/chad_positive_controls_v4.yml"
    with open(path, "r") as f:
        gene_clusters = yaml.safe_load(f)

    active_genes = activity_map[activity_map["below_corrected_p"]][
        "perturbation"
    ].tolist()
    active_genes = [gene for gene in active_genes if gene != "NTC"]
    df = adata_to_copairs_df(adata)
    df = df[df["perturbation"].isin(active_genes)]

    # Collect clusters that have >=2 active members
    active_genes_set = set(active_genes)
    valid_clusters = {
        k: [g for g in v["genes"] if g in active_genes_set]
        for k, v in gene_clusters.items()
        if len([g for g in v["genes"] if g in active_genes_set]) > 1
    }
    n_clusters = len(valid_clusters)
    logger.info(f"  {len(active_genes)} active genes → {n_clusters} manual clusters to test")

    # Precompute base meta + features once (shared read-only across threads)
    static_meta_cols = ["perturbation", "n_cells", "guides", "reporter",
                        "experiment", "is_NTC", "n_experiments"]
    static_meta_cols = [c for c in static_meta_cols if c in df.columns]
    feature_cols = [c for c in df.columns if c not in static_meta_cols]
    meta_base = df[static_meta_cols].reset_index(drop=True)
    feats_array = np.asarray(df[feature_cols], dtype=np.float32)

    # Determine optimal workers for parallel cluster computation
    try:
        from ops_utils.hpc.resource_manager import get_optimal_workers
        n_workers = get_optimal_workers(use_gpu=False, model_ram_gb=0.5, data_ram_gb=0.5, verbose=False)
    except Exception:
        n_workers = 4
    n_workers = min(n_workers, n_clusters)
    logger.info(f"  Using {n_workers} workers for manual cluster computation")

    results = Parallel(n_jobs=n_workers, prefer="processes")(
        delayed(_compute_single_complex_map)(
            members, k, "complex_num", meta_base, feats_array,
        )
        for k, members in tqdm(valid_clusters.items(), total=n_clusters, desc="Manual clusters")
    )
    complex_map_list = [r for r in results if r is not None]

    all_complex_results_df = pd.concat(complex_map_list, ignore_index=True)
    all_complex_results_df["-log10(p-value)"] = -all_complex_results_df[
        "corrected_p_value"
    ].apply(np.log10)

    phenotypic_consistency_ratio = all_complex_results_df.below_corrected_p.mean()
    if plot_results:
        plt.scatter(
            data=all_complex_results_df,
            x="mean_average_precision",
            y="-log10(p-value)",
            c="below_corrected_p",
            cmap="tab10",
            s=10,
        )
        plt.title("Phenotypic consistency (Manual)")
        plt.xlabel("mAP")
        plt.ylabel("-log10(p-value)")
        plt.axhline(-np.log10(0.05), color="black", linestyle="--")
        plt.text(
            0.65,
            1.5,
            f"Phenotypically distinct = {100 * phenotypic_consistency_ratio:.2f}%",
            va="center",
            ha="left",
        )
        plt.show()

    return all_complex_results_df, phenotypic_consistency_ratio


def map_main(
    adata_guide_path: Optional[str],
    adata_guide: Optional[ad.AnnData],
    adata_gene_path: Optional[str],
    adata_gene: Optional[ad.AnnData],
    save_dir: str,
) -> Dict:
    """
    Run full phenotypic assessment pipeline.

    1. Phenotypic activity assessment (guide level)
    2. Phenotypic distinctiveness (guide level)
    3. Phenotypic consistency - CORUM (gene level)
    4. Phenotypic consistency - manual annotation (gene level)

    Parameters
    ----------
    adata_guide_path : str or None
        Path to guide-level h5ad.
    adata_guide : AnnData or None
        Pre-loaded guide-level AnnData.
    adata_gene_path : str or None
        Path to gene-level h5ad.
    adata_gene : AnnData or None
        Pre-loaded gene-level AnnData.
    save_dir : str
        Directory to save results.

    Returns
    -------
    dict with all results and ratios.
    """
    if adata_guide is None and adata_guide_path is None:
        raise ValueError("Either adata_guide or adata_guide_path must be provided")
    if adata_gene is None and adata_gene_path is None:
        raise ValueError("Either adata_gene or adata_gene_path must be provided")

    if adata_guide is None:
        logger.info(f"Loading guide-level AnnData from {adata_guide_path}")
        adata_guide = ad.read_h5ad(adata_guide_path)

    if adata_gene is None:
        logger.info(f"Loading gene-level AnnData from {adata_gene_path}")
        adata_gene = ad.read_h5ad(adata_gene_path)

    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    logger.info("Running phenotypic activity assessment...")
    activity_map, active_ratio = phenotypic_activity_assesment(
        adata_guide, plot_results=False
    )
    activity_csv_path = save_path / "phenotypic_activity.csv"
    activity_map.to_csv(activity_csv_path, index=False)
    logger.info(f"Saved activity results to {activity_csv_path}")
    logger.info(f"Active ratio: {100 * active_ratio:.2f}%")

    logger.info("Running phenotypic distinctiveness...")
    distinctiveness_map, distinctive_ratio = phenotypic_distinctivness(
        adata_guide, activity_map, plot_results=False
    )
    distinctiveness_csv_path = save_path / "phenotypic_distinctiveness.csv"
    distinctiveness_map.to_csv(distinctiveness_csv_path, index=False)
    logger.info(f"Saved distinctiveness results to {distinctiveness_csv_path}")
    logger.info(f"Distinctive ratio: {100 * distinctive_ratio:.2f}%")

    logger.info("Running phenotypic consistency (CORUM)...")
    consistency_corum_map, consistency_corum_ratio = phenotypic_consistency_corum(
        adata_gene, activity_map, plot_results=False
    )
    consistency_corum_csv_path = save_path / "phenotypic_consistency_corum.csv"
    consistency_corum_map.to_csv(consistency_corum_csv_path, index=False)
    logger.info(f"Saved CORUM consistency results to {consistency_corum_csv_path}")
    logger.info(f"CORUM consistency ratio: {100 * consistency_corum_ratio:.2f}%")

    logger.info("Running phenotypic consistency (manual annotation)...")
    consistency_manual_map, consistency_manual_ratio = (
        phenotypic_consistency_manual_annotation(
            adata_gene, activity_map, plot_results=False
        )
    )
    consistency_manual_csv_path = save_path / "phenotypic_consistency_manual.csv"
    consistency_manual_map.to_csv(consistency_manual_csv_path, index=False)
    logger.info(f"Saved manual consistency results to {consistency_manual_csv_path}")
    logger.info(f"Manual consistency ratio: {100 * consistency_manual_ratio:.2f}%")

    logger.info(f"All results saved to {save_dir}")

    return {
        "activity_map": activity_map,
        "active_ratio": active_ratio,
        "distinctiveness_map": distinctiveness_map,
        "distinctive_ratio": distinctive_ratio,
        "consistency_corum_map": consistency_corum_map,
        "consistency_corum_ratio": consistency_corum_ratio,
        "consistency_manual_map": consistency_manual_map,
        "consistency_manual_ratio": consistency_manual_ratio,
    }
