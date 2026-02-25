"""
Cell Painting Challenge Stage: CP vs Live-Cell mAP Head-to-Head Comparison.

Compares Cell Painting (fixed staining) organelle features against live-cell
fluorescent reporter features to assess which modality better discriminates
gene knockout perturbations.

For each organelle type (e.g., mitochondria), this stage:
1. Extracts ONLY the single matching organelle's features from each side
2. Downsamples to equal cell counts for a fair comparison
3. Aggregates to guide level
4. Runs copairs mAP phenotypic activity assessment on both
5. Compares active gene counts, mAP distributions, and overlap

Configuration is read from ops_process/ops_analysis/configs/cp_challenge_config.yaml.

Usage:
  # Local mode — runs all comparisons sequentially in the current process:
  python -m organelle_profiler.fe_graphs.stages.fe_graphs_cp_challenge_stage -e 94
  python -m organelle_profiler.fe_graphs.stages.fe_graphs_cp_challenge_stage -e 94 -o /path/to/output

  # SLURM mode — submits each comparison as a separate SLURM job:
  python -m organelle_profiler.fe_graphs.stages.fe_graphs_cp_challenge_stage -e 94 --slurm
  python -m organelle_profiler.fe_graphs.stages.fe_graphs_cp_challenge_stage -e 94 --slurm --yes
  python -m organelle_profiler.fe_graphs.stages.fe_graphs_cp_challenge_stage -e 94 --slurm --no-wait
  python -m organelle_profiler.fe_graphs.stages.fe_graphs_cp_challenge_stage -e 94 --slurm --slurm-memory 256GB --slurm-time 90 --slurm-cpus 16

CLI arguments:
  -e, --experiment      CP experiment name or shorthand (default: ops0094_20251217)
  -o, --output-dir      Output directory (default: <fast_ops>/<experiment>/results/feature_extraction/graphs)

SLURM options:
  --slurm               Submit each comparison as a separate SLURM job
  --no-wait             Don't wait for SLURM jobs to complete (fire-and-forget)
  --yes, -y             Skip confirmation prompt
  --quiet, -q           Reduce output verbosity
  --slurm-memory        Memory per SLURM job (default: 128GB)
  --slurm-time          Time limit per SLURM job in minutes (default: 60)
  --slurm-cpus          CPUs per SLURM job (default: 16)

Aggregation mode:
  --aggregate           Re-run summary tables + plots from existing per-comparison CSVs
                        (skips all computation, just reads CSVs and generates plots)

  # Aggregate-only — regenerate summary tables + plots without re-running comparisons:
  python -m organelle_profiler.fe_graphs.stages.fe_graphs_cp_challenge_stage -e 94 --aggregate
"""

import time
import numpy as np
import pandas as pd
import anndata as ad
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import yaml
import shutil
from pathlib import Path
from typing import Optional, Dict, List, Tuple
import logging

try:
    from .fe_graphs_stage_base import BaseStage, StageResult
    from ..plotting.fe_graphs_utils import save_figure
    from ..analysis.fe_graphs_map_analysis import (
        phenotypic_activity_assesment,
        phenotypic_distinctivness,
        phenotypic_consistency_corum,
        phenotypic_consistency_manual_annotation,
        compute_auc_score,
        compute_threshold_sweep_auc,
    )
except ImportError:
    from organelle_profiler.fe_graphs.stages.fe_graphs_stage_base import BaseStage, StageResult
    from organelle_profiler.fe_graphs.plotting.fe_graphs_utils import save_figure
    from organelle_profiler.fe_graphs.analysis.fe_graphs_map_analysis import (
        phenotypic_activity_assesment,
        phenotypic_distinctivness,
        phenotypic_consistency_corum,
        phenotypic_consistency_manual_annotation,
        compute_auc_score,
        compute_threshold_sweep_auc,
    )

logger = logging.getLogger(__name__)

# Suppress copairs internal INFO logging (repetitive "Indexing metadata...", "Finding positive pairs...")
logging.getLogger("copairs").setLevel(logging.WARNING)

# Path to the default config
DEFAULT_CONFIG_PATH = Path(__file__).parents[4] / "configs" / "cp_challenge_config.yaml"

# Label-free phase channel used as orthogonal negative control
CONTROL_PATTERN = "phase2d_tubular"


class CPChallengeStage(BaseStage):
    """
    Head-to-head comparison of Cell Painting vs live-cell features via mAP.

    For each organelle defined in the config, loads the corresponding live-cell
    experiment's cell-level data, filters to the single matching organelle's
    features on each side, downsamples to equal cell counts, aggregates to
    guide level, and runs phenotypic activity assessment.
    """

    STAGE_NUMBER = 12
    STAGE_NAME = "cp_challenge"

    def run(self) -> StageResult:
        """Run CP vs live-cell challenge comparisons."""
        self.log_start("Cell Painting Challenge: CP vs Live-Cell mAP comparison")
        result = StageResult()

        # Only run at guide level
        if self.level != "guide":
            logger.info(f"CP Challenge stage only runs at guide level, skipping {self.level}")
            return result

        # Load config
        config = self._load_config()
        if config is None:
            result.add_error(f"Config not found at {DEFAULT_CONFIG_PATH}")
            return result

        # Save copy of config used
        config_copy_path = self.output_dir / "cp_challenge_config.yaml"
        shutil.copy2(DEFAULT_CONFIG_PATH, config_copy_path)
        result.add_file(config_copy_path)

        # Load CP experiment cell-level data
        cp_experiment = config["cp_experiment"]
        logger.info(f"CP experiment: {cp_experiment}")
        cp_cell_adata = self._load_cell_adata(cp_experiment)
        if cp_cell_adata is None:
            result.add_error(f"Could not load cell-level data for CP experiment {cp_experiment}")
            return result

        # Discover CP organelle names
        cp_organelles = self._discover_organelles(cp_cell_adata)
        logger.info(f"CP experiment organelles: {sorted(cp_organelles.keys())}")

        # Pre-discover CP phase2d_tubular features for control comparisons
        cp_ctrl_name, cp_ctrl_cols = self._match_organelle(cp_organelles, CONTROL_PATTERN)
        if cp_ctrl_name is None:
            logger.warning(
                f"No CP control features matching '{CONTROL_PATTERN}' — "
                f"control comparisons will be skipped"
            )
        else:
            logger.info(
                f"CP control organelle: {cp_ctrl_name} ({len(cp_ctrl_cols)} features)"
            )

        # Prepare comparison tasks (shared across normalization methods)
        comparisons = config.get("comparisons", {})
        comparison_tasks = []
        discovery_records = []

        for organelle_name, organelle_config in comparisons.items():
            cp_pattern = organelle_config["cp_organelle_pattern"]

            # Find matching CP organelle features
            cp_org_name, cp_feat_cols = self._match_organelle(
                cp_organelles, cp_pattern
            )
            if cp_org_name is None:
                logger.warning(
                    f"No CP organelle matching '{cp_pattern}' found in {cp_experiment}. "
                    f"Available: {sorted(cp_organelles.keys())}"
                )
                result.add_error(f"No CP organelle matching '{cp_pattern}'")
                continue

            logger.info(
                f"\n{'='*60}\n  {organelle_name.upper()}: CP organelle = {cp_org_name} "
                f"({len(cp_feat_cols)} features)\n{'='*60}"
            )
            discovery_records.append({
                "organelle_type": organelle_name,
                "side": "cell_painting",
                "experiment": cp_experiment,
                "matched_organelle": cp_org_name,
                "n_features": len(cp_feat_cols),
            })

            # Collect tasks
            for live_entry in organelle_config.get("live_cell", []):
                comparison_tasks.append({
                    "cp_org_name": cp_org_name,
                    "cp_feat_cols": cp_feat_cols,
                    "live_experiment": live_entry["experiment"],
                    "live_pattern": live_entry["organelle_pattern"],
                    "organelle_name": organelle_name,
                    "notes": live_entry.get("notes", ""),
                })

        # Save organelle discovery table (shared)
        if discovery_records:
            discovery_df = pd.DataFrame(discovery_records)
            discovery_path = self.output_dir / "organelle_discovery.csv"
            discovery_df.to_csv(discovery_path, index=False)
            result.add_file(discovery_path)

        # Run comparisons for BOTH normalization methods
        norm_methods = ["global", "ntc"]
        base_output_dir = self.output_dir

        for norm_method in norm_methods:
            norm_label = f"{norm_method}_norm"
            norm_output_dir = base_output_dir / norm_label
            norm_output_dir.mkdir(parents=True, exist_ok=True)

            logger.info(f"\n{'#'*60}")
            logger.info(f"  NORMALIZATION: {norm_method.upper()}")
            logger.info(f"  Output: {norm_output_dir}")
            logger.info(f"{'#'*60}")

            all_comparisons = []
            all_control_comparisons = []
            control_seen_experiments = set()  # deduplicate controls by live experiment
            norm_discovery = list(discovery_records)  # copy shared discovery

            for task in comparison_tasks:
                org_output_dir = norm_output_dir / "per_organelle" / task["organelle_name"]
                org_output_dir.mkdir(parents=True, exist_ok=True)

                comp_result = self._run_single_comparison(
                    cp_cell_adata=cp_cell_adata,
                    cp_org_name=task["cp_org_name"],
                    cp_feat_cols=task["cp_feat_cols"],
                    live_experiment=task["live_experiment"],
                    live_pattern=task["live_pattern"],
                    organelle_name=task["organelle_name"],
                    org_output_dir=org_output_dir,
                    notes=task.get("notes", ""),
                    norm_method=norm_method,
                )
                for err in comp_result.get("errors", []):
                    result.add_error(err)
                for f in comp_result.get("files", []):
                    result.add_file(f)

                comparison = comp_result.get("comparison")
                if comparison is None:
                    continue

                comparison["notes"] = task["notes"]
                comparison["norm_method"] = norm_method
                all_comparisons.append(comparison)

                norm_discovery.append({
                    "organelle_type": task["organelle_name"],
                    "side": "live_cell",
                    "experiment": task["live_experiment"],
                    "matched_organelle": comparison.get("live_organelle", ""),
                    "n_features": comparison.get("live_n_features", 0),
                })

                # --- Control comparison: phase2d_tubular on BOTH sides ---
                # Deduplicate: the control only depends on the live experiment,
                # not the organelle, so skip if we've already run this experiment.
                live_exp = task["live_experiment"]
                if cp_ctrl_name is not None and live_exp not in control_seen_experiments:
                    control_seen_experiments.add(live_exp)

                    live_short = live_exp.split("_")[0] if "_" in live_exp else live_exp
                    ctrl_output_dir = (
                        norm_output_dir / "per_organelle_control" / live_short
                    )
                    ctrl_output_dir.mkdir(parents=True, exist_ok=True)

                    logger.info(
                        f"\n  [CONTROL] {CONTROL_PATTERN}: "
                        f"CP vs {live_exp}"
                    )
                    ctrl_result = self._run_single_comparison(
                        cp_cell_adata=cp_cell_adata,
                        cp_org_name=cp_ctrl_name,
                        cp_feat_cols=cp_ctrl_cols,
                        live_experiment=live_exp,
                        live_pattern=CONTROL_PATTERN,
                        organelle_name=live_short,
                        org_output_dir=ctrl_output_dir,
                        notes=f"CONTROL: {CONTROL_PATTERN}",
                        norm_method=norm_method,
                    )
                    for err in ctrl_result.get("errors", []):
                        result.add_error(err)
                    for f in ctrl_result.get("files", []):
                        result.add_file(f)

                    ctrl_comparison = ctrl_result.get("comparison")
                    if ctrl_comparison is not None:
                        ctrl_comparison["notes"] = f"CONTROL: {CONTROL_PATTERN}"
                        ctrl_comparison["norm_method"] = norm_method
                        all_control_comparisons.append(ctrl_comparison)

            # Build and save summary for this normalization method
            if all_comparisons:
                summary_df = pd.DataFrame(all_comparisons)
                summary_path = norm_output_dir / "cp_challenge_summary.csv"
                summary_df.to_csv(summary_path, index=False)
                result.add_file(summary_path)
                result.data[f"summary_df_{norm_method}"] = summary_df

                # Generate overall comparison plots into norm subdir
                # Temporarily override output_dir for plot generation
                saved_output_dir = self._output_dir
                self._output_dir = norm_output_dir
                self._plot_overall_comparison(summary_df, result)
                self._output_dir = saved_output_dir

                # Log summary with prettytable
                _print_summary_table(summary_df, f"{norm_method}_norm", use_logger=True)

                result.add_metric(f"n_comparisons_{norm_method}", len(all_comparisons))
            else:
                result.add_error(f"No successful comparisons for {norm_method} normalization")

            # Build and save CONTROL summary for this normalization method
            if all_control_comparisons:
                control_summary_df = pd.DataFrame(all_control_comparisons)
                control_summary_path = norm_output_dir / "cp_challenge_control_summary.csv"
                control_summary_df.to_csv(control_summary_path, index=False)
                result.add_file(control_summary_path)
                result.data[f"control_summary_df_{norm_method}"] = control_summary_df

                saved_output_dir = self._output_dir
                self._output_dir = norm_output_dir
                self._plot_overall_comparison(
                    control_summary_df, result,
                    subdir_prefix="control_", title_extra=" (Control)",
                )
                self._output_dir = saved_output_dir

                _print_summary_table(
                    control_summary_df, f"{norm_method}_norm CONTROL", use_logger=True,
                )
                result.add_metric(
                    f"n_control_comparisons_{norm_method}", len(all_control_comparisons),
                )

            # Build BATCH-CORRECTED summary for this normalization method
            if all_comparisons and all_control_comparisons:
                normalized_df = self._compute_normalized_summary(summary_df, control_summary_df)
                if not normalized_df.empty:
                    norm_summary_path = norm_output_dir / "cp_challenge_normalized_summary.csv"
                    normalized_df.to_csv(norm_summary_path, index=False)
                    result.add_file(norm_summary_path)

                    saved_output_dir = self._output_dir
                    self._output_dir = norm_output_dir
                    self._plot_overall_comparison(
                        normalized_df, result,
                        subdir_prefix="normalized_", title_extra=" (Batch-Corrected)",
                    )
                    self._output_dir = saved_output_dir

                    _print_summary_table(
                        normalized_df, f"{norm_method}_norm BATCH-CORRECTED", use_logger=True,
                    )

        logger.info(f"\nResults saved to:\n  {base_output_dir / 'global_norm'}\n  {base_output_dir / 'ntc_norm'}")
        self.log_complete(result)
        return result

    # -------------------------------------------------------------------------
    # Config and data loading
    # -------------------------------------------------------------------------

    def _load_config(self) -> Optional[Dict]:
        """Load the CP challenge config YAML."""
        if not DEFAULT_CONFIG_PATH.exists():
            logger.error(f"Config not found: {DEFAULT_CONFIG_PATH}")
            return None
        with open(DEFAULT_CONFIG_PATH) as f:
            return yaml.safe_load(f)

    def _load_cell_adata(self, experiment: str) -> Optional[ad.AnnData]:
        """Load cell-level AnnData for an experiment from fast_ops partition."""
        from ops_utils.data.filesystem import resolve_experiment_name
        from ops_utils.data.experiment import OpsDataset

        try:
            resolved = resolve_experiment_name(
                experiment, allow_interactive=False, autoselect=True
            )
            dataset = OpsDataset(resolved)
            h5ad_path = (
                dataset.results_fast / "feature_extraction" / f"{resolved}_cell_features.h5ad"
            )

            if not h5ad_path.exists():
                logger.error(f"Cell features not found: {h5ad_path}")
                return None

            logger.info(f"Loading cell features from: {h5ad_path}")
            adata = ad.read_h5ad(h5ad_path)
            logger.info(f"  Loaded {adata.n_obs} cells, {adata.n_vars} features")
            return adata
        except Exception as e:
            logger.error(f"Failed to load cell data for {experiment}: {e}")
            return None

    # -------------------------------------------------------------------------
    # Organelle discovery and matching
    # -------------------------------------------------------------------------

    def _discover_organelles(self, adata: ad.AnnData) -> Dict[str, List[str]]:
        """
        Discover organelle groups from AnnData var metadata.

        Returns mapping of organelle_name -> list of feature column names.
        """
        organelle_groups = {}
        for feat_name, row in adata.var.iterrows():
            organelle = row.get("organelle", None)
            if organelle is None or (isinstance(organelle, float) and pd.isna(organelle)):
                continue
            if organelle not in organelle_groups:
                organelle_groups[organelle] = []
            organelle_groups[organelle].append(feat_name)
        return organelle_groups

    def _match_organelle(
        self, organelle_groups: Dict[str, List[str]], pattern: str
    ) -> Tuple[Optional[str], List[str]]:
        """
        Find the organelle name matching a pattern.

        Returns (organelle_name, feature_columns) or (None, []).
        """
        matches = [
            name for name in organelle_groups if pattern.lower() in name.lower()
        ]
        if not matches:
            return None, []
        # Prefer exact match, then shortest match (most specific)
        matches.sort(key=len)
        best = matches[0]
        return best, organelle_groups[best]

    def _match_features_by_metric(
        self,
        cp_adata: ad.AnnData,
        cp_feat_cols: List[str],
        live_adata: ad.AnnData,
        live_feat_cols: List[str],
    ) -> Tuple[List[str], List[str]]:
        """
        Match features between CP and live-cell by their base metric name.

        Feature columns have organelle-specific prefixes (e.g. cp1_nuclei_hoechst_area
        vs mcherry_area) but share the same underlying metric stored in adata.var["metric"].
        This method finds metrics present in both groups and returns the filtered
        feature column lists so both sides measure the same set of properties.

        Returns (filtered_cp_feat_cols, filtered_live_feat_cols).
        """
        # Build metric -> feature name mappings from adata.var
        cp_metric_map = {}  # metric -> feature_name
        for feat in cp_feat_cols:
            if feat in cp_adata.var.index and "metric" in cp_adata.var.columns:
                metric = cp_adata.var.loc[feat, "metric"]
                if metric and not (isinstance(metric, float) and pd.isna(metric)):
                    cp_metric_map[str(metric)] = feat

        live_metric_map = {}
        for feat in live_feat_cols:
            if feat in live_adata.var.index and "metric" in live_adata.var.columns:
                metric = live_adata.var.loc[feat, "metric"]
                if metric and not (isinstance(metric, float) and pd.isna(metric)):
                    live_metric_map[str(metric)] = feat

        # Find common metrics
        common_metrics = sorted(set(cp_metric_map.keys()) & set(live_metric_map.keys()))
        cp_only = set(cp_metric_map.keys()) - set(live_metric_map.keys())
        live_only = set(live_metric_map.keys()) - set(cp_metric_map.keys())

        matched_cp = [cp_metric_map[m] for m in common_metrics]
        matched_live = [live_metric_map[m] for m in common_metrics]

        cp_total = len(cp_metric_map)
        live_total = len(live_metric_map)
        n_matched = len(common_metrics)
        cp_pct = n_matched / cp_total * 100 if cp_total else 0
        live_pct = n_matched / live_total * 100 if live_total else 0
        logger.info(
            f"  Feature matching by metric: "
            f"CP {cp_total} metrics, Live {live_total} metrics "
            f"-> {n_matched} matched "
            f"(CP: {cp_pct:.1f}% retained, Live: {live_pct:.1f}% retained)"
        )
        # Show what matched: metric -> (CP feature, Live feature)
        logger.info(f"    Matched metrics ({len(common_metrics)}):")
        for m in common_metrics:
            logger.info(f"      {m:30s}  CP: {cp_metric_map[m]:50s}  Live: {live_metric_map[m]}")
        if cp_only:
            logger.info(f"    CP-only (dropped {len(cp_only)}):")
            for m in sorted(cp_only):
                logger.info(f"      {m:30s}  -> {cp_metric_map[m]}")
        if live_only:
            logger.info(f"    Live-only (dropped {len(live_only)}):")
            for m in sorted(live_only):
                logger.info(f"      {m:30s}  -> {live_metric_map[m]}")

        return matched_cp, matched_live

    # -------------------------------------------------------------------------
    # Downsampling and aggregation
    # -------------------------------------------------------------------------

    def _downsample_cells(
        self,
        df_a: pd.DataFrame,
        df_b: pd.DataFrame,
        seed: int = 42,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, int]:
        """
        Random downsample the larger dataset to match the smaller.

        Returns (df_a_sampled, df_b_sampled, n_cells_used).
        """
        n_a, n_b = len(df_a), len(df_b)
        target = min(n_a, n_b)
        rng = np.random.RandomState(seed)

        if n_a > target:
            idx = rng.choice(n_a, target, replace=False)
            df_a = df_a.iloc[idx].reset_index(drop=True)
        elif n_b > target:
            idx = rng.choice(n_b, target, replace=False)
            df_b = df_b.iloc[idx].reset_index(drop=True)

        logger.info(
            f"  Downsampled: {n_a} vs {n_b} -> {target} cells each"
        )
        return df_a, df_b, target

    # Match canonical aggregation from feature_extraction_slurm.py
    AGG_FUNCS = ["sum", "mean", "median", "std", "min", "max"]

    def _aggregate_to_guide(
        self, cell_df: pd.DataFrame, feature_cols: List[str]
    ) -> Tuple[pd.DataFrame, List[str]]:
        """
        Aggregate cell-level data to guide level using 7 agg functions.

        Matches feature_extraction_slurm.py canonical pipeline.
        """
        if "sgRNA" not in cell_df.columns:
            raise ValueError("Missing 'sgRNA' column - cannot aggregate to guide level")

        sgrna_col = cell_df["sgRNA"]
        valid_mask = sgrna_col.notna() & (sgrna_col != "") & (sgrna_col != "None")
        cells = cell_df[valid_mask]
        n_filtered = (~valid_mask).sum()
        if n_filtered > 0:
            logger.info(f"  Filtered {n_filtered} cells without valid sgRNA")

        # Aggregate features with all 7 functions
        grouped = cells.groupby("sgRNA", observed=True)[feature_cols]
        guide_features = grouped.agg(self.AGG_FUNCS)
        guide_features.columns = ["_".join(col) for col in guide_features.columns]
        guide_features = guide_features.reset_index()

        # Metadata
        meta_cols = [c for c in ["gene_name", "barcode", "gene_effect", "NCBI_ID"]
                     if c in cells.columns and c != "sgRNA"]
        guide_meta = cells.groupby("sgRNA", observed=True)[meta_cols].first().reset_index()
        guide_df = pd.merge(guide_features, guide_meta, on="sgRNA", how="left")

        # n_cells
        sgrna_counts = cells["sgRNA"].value_counts()
        guide_df["n_cells"] = guide_df["sgRNA"].map(sgrna_counts).fillna(0).astype(int)

        # perturbation = gene_name (required by copairs mAP)
        guide_df["perturbation"] = guide_df.get("gene_name", "")

        # Filter guides with very few cells
        min_cells = 3
        before = len(guide_df)
        guide_df = guide_df[guide_df["n_cells"] >= min_cells]
        if len(guide_df) < before:
            logger.info(f"  Filtered {before - len(guide_df)} guides with <{min_cells} cells")

        # Aggregated feature column names (with suffixes)
        agg_feature_cols = [
            c for c in guide_df.columns
            if any(c.endswith(f"_{agg}") for agg in self.AGG_FUNCS)
        ]

        return guide_df, agg_feature_cols

    def _aggregate_to_gene(
        self, cell_df: pd.DataFrame, feature_cols: List[str],
        guide_df: Optional[pd.DataFrame] = None,
    ) -> Tuple[pd.DataFrame, List[str]]:
        """
        Aggregate cell-level data to gene level using 7 agg functions.

        Aggregates directly from cell data (NOT from guide means).
        """
        gene_col = "gene_name" if "gene_name" in cell_df.columns else "perturbation"

        grouped = cell_df.groupby(gene_col, observed=True)[feature_cols]
        gene_features = grouped.agg(self.AGG_FUNCS)
        gene_features.columns = ["_".join(col) for col in gene_features.columns]
        gene_features = gene_features.reset_index()

        # Metadata
        meta_cols = [c for c in ["gene_effect", "NCBI_ID"] if c in cell_df.columns]
        if meta_cols:
            gene_meta = cell_df.groupby(gene_col, observed=True)[meta_cols].first().reset_index()
            gene_df = pd.merge(gene_features, gene_meta, on=gene_col, how="left")
        else:
            gene_df = gene_features

        gene_cell_counts = cell_df[gene_col].value_counts()
        gene_df["n_cells"] = gene_df[gene_col].map(gene_cell_counts).fillna(0).astype(int)

        if guide_df is not None and gene_col in guide_df.columns:
            gene_guide_counts = guide_df[gene_col].value_counts()
            gene_df["n_guides"] = gene_df[gene_col].map(gene_guide_counts).fillna(0).astype(int)
        else:
            gene_df["n_guides"] = 0

        gene_df["perturbation"] = gene_df[gene_col]

        agg_feature_cols = [
            c for c in gene_df.columns
            if any(c.endswith(f"_{agg}") for agg in self.AGG_FUNCS)
        ]

        return gene_df, agg_feature_cols

    def _zscore_normalize(
        self, guide_df: pd.DataFrame, feature_cols: List[str],
        method: str = "global",
    ) -> pd.DataFrame:
        """
        Z-score normalize features.

        Parameters
        ----------
        method : str
            "global" (default) — use all-sample mean/std. Better for
            inter-perturbation comparisons (distinctiveness, consistency).
            "ntc" — use NTC-only mean/std. Centers relative to negative
            control baseline.
        """
        if method == "ntc":
            ntc_mask = guide_df["perturbation"] == "NTC"
            n_ref = ntc_mask.sum()
            if n_ref < 2:
                logger.warning(f"  Only {n_ref} NTC guides - falling back to global z-score")
                ref_mask = pd.Series(True, index=guide_df.index)
                n_ref = len(guide_df)
                label = "all samples (NTC fallback)"
            else:
                ref_mask = ntc_mask
                label = f"{n_ref} NTC guides"
        else:
            ref_mask = pd.Series(True, index=guide_df.index)
            n_ref = len(guide_df)
            label = f"all {n_ref} samples"

        ref_features = guide_df.loc[ref_mask, feature_cols].values.astype(np.float64)
        means = np.nanmean(ref_features, axis=0)
        stds = np.nanstd(ref_features, axis=0, ddof=1)
        stds[stds == 0] = 1.0  # avoid division by zero

        guide_df = guide_df.copy()
        guide_df[feature_cols] = (
            (guide_df[feature_cols].values.astype(np.float64) - means) / stds
        ).astype(np.float32)

        logger.info(
            f"  Z-score normalized using {label} "
            f"(mean range: {means.min():.2f} to {means.max():.2f})"
        )
        return guide_df

    def _df_to_adata(
        self, df: pd.DataFrame, feature_cols: List[str], obs_cols: List[str]
    ) -> ad.AnnData:
        """
        Convert a DataFrame to AnnData for copairs mAP functions.
        Filters out zero-variance features that add noise to cosine similarity.
        """
        obs = df[[c for c in obs_cols if c in df.columns]].copy()
        obs.index = obs.index.astype(str)

        X = df[feature_cols].values.astype(np.float32)
        X = np.nan_to_num(X, nan=0.0)

        # Drop zero-variance features
        variances = np.var(X, axis=0)
        keep = variances > 0
        n_dropped = (~keep).sum()
        if n_dropped > 0:
            logger.info(f"  Dropped {n_dropped}/{len(feature_cols)} zero-variance features")
            X = X[:, keep]
            feature_cols = [f for f, k in zip(feature_cols, keep) if k]

        adata = ad.AnnData(
            X=X,
            obs=obs.reset_index(drop=True),
            var=pd.DataFrame(index=feature_cols),
        )
        return adata

    # -------------------------------------------------------------------------
    # Single comparison
    # -------------------------------------------------------------------------

    def _run_single_comparison(
        self,
        cp_cell_adata: ad.AnnData,
        cp_org_name: str,
        cp_feat_cols: List[str],
        live_experiment: str,
        live_pattern: str,
        organelle_name: str,
        org_output_dir: Path,
        notes: str = "",
        norm_method: str = "global",
    ) -> Dict:
        """
        Run a single CP vs live-cell comparison for one organelle.

        Generates plots immediately after each step completes.
        Returns dict with keys: comparison, files, errors,
        cp_activity, live_activity, cp_active_ratio, live_active_ratio.
        """
        files = []
        errors = []
        _empty = {
            "comparison": None, "files": files, "errors": errors,
            "cp_activity": None, "live_activity": None,
            "cp_active_ratio": 0.0, "live_active_ratio": 0.0,
        }

        logger.info(f"\n  [{organelle_name.upper()}] Comparing CP vs {live_experiment}")

        # Load live-cell data
        live_adata = self._load_cell_adata(live_experiment)
        if live_adata is None:
            errors.append(f"Could not load {live_experiment}")
            return _empty

        # Discover live-cell organelles and match
        live_organelles = self._discover_organelles(live_adata)
        live_org_name, live_feat_cols = self._match_organelle(
            live_organelles, live_pattern
        )
        if live_org_name is None:
            errors.append(
                f"No organelle matching '{live_pattern}' in {live_experiment}. "
                f"Available: {sorted(live_organelles.keys())}"
            )
            return _empty

        logger.info(
            f"  Live-cell organelle: {live_org_name} ({len(live_feat_cols)} features)"
        )

        # Filter to common features by metric name — a measurement type must
        # be present in BOTH groups (feature names differ by organelle prefix,
        # but the underlying metric in adata.var["metric"] is comparable)
        cp_feat_cols, live_feat_cols = self._match_features_by_metric(
            cp_cell_adata, cp_feat_cols, live_adata, live_feat_cols,
        )
        if len(cp_feat_cols) == 0:
            errors.append(
                f"No common metrics between CP ({cp_org_name}) "
                f"and live-cell ({live_org_name}) for {organelle_name}"
            )
            return _empty

        # Build cell DataFrames with ONLY the organelle features + metadata
        cp_cell_df = self._build_cell_df(cp_cell_adata, cp_feat_cols)
        live_cell_df = self._build_cell_df(live_adata, live_feat_cols)

        # Find common genes
        cp_genes = set(cp_cell_df["gene_name"].dropna().unique())
        live_genes = set(live_cell_df["gene_name"].dropna().unique())
        common_genes = cp_genes & live_genes

        # Remove empty/None gene names
        common_genes = {g for g in common_genes if g and str(g) != "None" and str(g) != "nan"}

        if len(common_genes) < 10:
            errors.append(
                f"Only {len(common_genes)} common genes between {cp_org_name} and "
                f"{live_experiment}. Need at least 10."
            )
            return _empty

        logger.info(
            f"  Common genes: {len(common_genes)} "
            f"(CP={len(cp_genes)}, Live={len(live_genes)})"
        )

        # Filter to common genes
        cp_cell_df = cp_cell_df[cp_cell_df["gene_name"].isin(common_genes)].copy()
        live_cell_df = live_cell_df[live_cell_df["gene_name"].isin(common_genes)].copy()

        # Downsample to equal cell counts
        cp_cell_df, live_cell_df, n_cells_used = self._downsample_cells(
            cp_cell_df, live_cell_df
        )

        # Aggregate to guide level (7 agg functions matching canonical pipeline)
        cp_guide_df, cp_guide_feat_cols = self._aggregate_to_guide(cp_cell_df, cp_feat_cols)
        live_guide_df, live_guide_feat_cols = self._aggregate_to_guide(live_cell_df, live_feat_cols)

        logger.info(
            f"  Guide-level: CP={len(cp_guide_df)} guides ({len(cp_guide_feat_cols)} features), "
            f"Live={len(live_guide_df)} guides ({len(live_guide_feat_cols)} features)"
        )

        # Minimum guide count check
        if len(cp_guide_df) < 10 or len(live_guide_df) < 10:
            errors.append(
                f"Too few guides after aggregation: CP={len(cp_guide_df)}, "
                f"Live={len(live_guide_df)}"
            )
            return _empty

        # Aggregate to gene level from CELL data (matching canonical pipeline)
        cp_gene_df, cp_gene_feat_cols = self._aggregate_to_gene(
            cp_cell_df, cp_feat_cols, guide_df=cp_guide_df
        )
        live_gene_df, live_gene_feat_cols = self._aggregate_to_gene(
            live_cell_df, live_feat_cols, guide_df=live_guide_df
        )

        logger.info(
            f"  Gene-level: CP={len(cp_gene_df)} genes ({len(cp_gene_feat_cols)} features), "
            f"Live={len(live_gene_df)} genes ({len(live_gene_feat_cols)} features)"
        )

        # Z-score normalize features
        logger.info(f"  Z-score normalization method: {norm_method}")
        cp_guide_df = self._zscore_normalize(cp_guide_df, cp_guide_feat_cols, method=norm_method)
        live_guide_df = self._zscore_normalize(live_guide_df, live_guide_feat_cols, method=norm_method)
        cp_gene_df = self._zscore_normalize(cp_gene_df, cp_gene_feat_cols, method=norm_method)
        live_gene_df = self._zscore_normalize(live_gene_df, live_gene_feat_cols, method=norm_method)

        # Convert to AnnData for guide and gene levels
        guide_obs_cols = ["perturbation", "sgRNA", "n_cells"]
        gene_obs_cols = ["perturbation", "n_cells", "n_guides"]

        cp_guide_adata = self._df_to_adata(cp_guide_df, cp_guide_feat_cols, guide_obs_cols)
        live_guide_adata = self._df_to_adata(live_guide_df, live_guide_feat_cols, guide_obs_cols)
        cp_gene_adata = self._df_to_adata(cp_gene_df, cp_gene_feat_cols, gene_obs_cols)
        live_gene_adata = self._df_to_adata(live_gene_df, live_gene_feat_cols, gene_obs_cols)

        # Determine short experiment name for filenames
        live_short = live_experiment.split("_")[0] if "_" in live_experiment else live_experiment

        # Initialize comparison dict
        comparison = {
            "organelle_type": organelle_name,
            "cp_experiment": self.data.experiment if hasattr(self.data, 'experiment') else "ops0094",
            "cp_organelle": cp_org_name,
            "cp_n_features": len(cp_feat_cols),
            "live_experiment": live_experiment,
            "live_organelle": live_org_name,
            "live_n_features": len(live_feat_cols),
            "n_matched_features": len(cp_feat_cols),
            "n_common_genes": len(common_genes),
            "n_cells_used": n_cells_used,
            "cp_n_guides": len(cp_guide_df),
            "live_n_guides": len(live_guide_df),
            "cp_n_genes": len(cp_gene_df),
            "live_n_genes": len(live_gene_df),
        }

        # --- Diagnostics ---
        for label, adata_check in [("CP guide", cp_guide_adata), ("Live guide", live_guide_adata)]:
            perturb = adata_check.obs["perturbation"]
            n_ntc = (perturb == "NTC").sum()
            n_unique = perturb.nunique()
            X = adata_check.X
            zero_frac = (X == 0).mean()
            nan_frac = np.isnan(X).mean() if np.issubdtype(X.dtype, np.floating) else 0
            var = np.var(X, axis=0)
            zero_var = (var == 0).sum()
            logger.info(
                f"  [{label}] {adata_check.n_obs} obs, {adata_check.n_vars} features, "
                f"{n_unique} unique perturbations, {n_ntc} NTC guides, "
                f"zero_frac={zero_frac:.3f}, nan_frac={nan_frac:.3f}, "
                f"zero_var_features={zero_var}/{adata_check.n_vars}, "
                f"X.dtype={X.dtype}"
            )

            # Perturbation distribution check
            perturb_counts = perturb.value_counts()
            multi_guide = (perturb_counts >= 2).sum()
            logger.info(
                f"  [{label}] Perturbations with >=2 guides: {multi_guide}/{n_unique}"
            )

        # --- 1. Phenotypic Activity (guide level) ---
        t_comparison_start = time.time()
        try:
            t0 = time.time()
            logger.info(
                f"\n  --- [{organelle_name.upper()}] 1/4 Phenotypic Activity: "
                f"CP {cp_org_name} vs Live {live_experiment} ---"
            )
            cp_activity, cp_active_ratio = phenotypic_activity_assesment(cp_guide_adata, False)
            live_activity, live_active_ratio = phenotypic_activity_assesment(live_guide_adata, False)
            logger.info(f"  Activity done in {time.time() - t0:.1f}s")

            # Post-mAP diagnostics
            for label, activity in [("CP", cp_activity), ("Live", live_activity)]:
                mAP = activity["mean_average_precision"]
                pvals = activity["corrected_p_value"]
                logger.info(
                    f"  [{label} post-mAP] mAP: min={mAP.min():.4f}, med={mAP.median():.4f}, "
                    f"max={mAP.max():.4f}, std={mAP.std():.4f}"
                )
                logger.info(
                    f"  [{label} post-mAP] p-value: min={pvals.min():.4e}, "
                    f"med={pvals.median():.4e}, max={pvals.max():.4e}"
                )
                logger.info(
                    f"  [{label} post-mAP] Active: "
                    f"{activity['below_corrected_p'].sum()}/{len(activity)}"
                )

            cp_active_genes = set(
                cp_activity[cp_activity["below_corrected_p"]]["perturbation"]
            )
            live_active_genes = set(
                live_activity[live_activity["below_corrected_p"]]["perturbation"]
            )
            overlap = cp_active_genes & live_active_genes
            union = cp_active_genes | live_active_genes
            jaccard = len(overlap) / len(union) if union else 0.0

            # AUC scores (threshold-free)
            cp_auc_act = compute_auc_score(cp_activity)
            live_auc_act = compute_auc_score(live_activity)
            cp_sweep_act = compute_threshold_sweep_auc(cp_activity)
            live_sweep_act = compute_threshold_sweep_auc(live_activity)

            comparison.update({
                "cp_active_ratio": cp_active_ratio,
                "live_active_ratio": live_active_ratio,
                "cp_n_active": len(cp_active_genes),
                "live_n_active": len(live_active_genes),
                "cp_mean_mAP_activity": cp_activity["mean_average_precision"].mean(),
                "live_mean_mAP_activity": live_activity["mean_average_precision"].mean(),
                "active_overlap": len(overlap),
                "active_jaccard": jaccard,
                "cp_auc_activity": cp_auc_act,
                "live_auc_activity": live_auc_act,
                "cp_sweep_auc_activity": cp_sweep_act,
                "live_sweep_auc_activity": live_sweep_act,
            })

            # Add per-gene AUC contribution column (mAP * normalized -log10(p))
            for df in [cp_activity, live_activity]:
                p_clamped = np.clip(df["corrected_p_value"].values.astype(np.float64), 1e-6, 1.0)
                w = -np.log10(p_clamped) / 6.0
                df["auc_contribution"] = df["mean_average_precision"].values * w

            # Save activity CSVs
            cp_act_path = org_output_dir / f"cp_activity_{live_short}.csv"
            live_act_path = org_output_dir / f"livecell_{live_short}_activity.csv"
            cp_activity.to_csv(cp_act_path, index=False)
            live_activity.to_csv(live_act_path, index=False)
            files.extend([cp_act_path, live_act_path])

            # Generate activity plots immediately
            self._plot_step_scatter(
                cp_activity, live_activity, cp_active_ratio, live_active_ratio,
                "activity", organelle_name, cp_org_name, live_org_name,
                live_experiment, org_output_dir, live_short, notes, files,
                n_cells_used=n_cells_used,
            )
            try:
                overlap_path = self._plot_active_gene_overlap(
                    cp_activity, live_activity,
                    organelle_name, live_experiment,
                    org_output_dir, live_short, notes=notes,
                    n_cells_used=n_cells_used,
                )
                if overlap_path:
                    files.append(overlap_path)
            except Exception as e_plot:
                logger.warning(f"  Overlap plot failed (non-fatal): {e_plot}")

            # Activity UMAPs
            for side_label, side_adata, side_map in [
                ("CP", cp_guide_adata, cp_activity),
                ("Live", live_guide_adata, live_activity),
            ]:
                try:
                    umap_path = self._plot_metric_umap(
                        side_adata, side_map, "activity", side_label,
                        organelle_name, org_output_dir, live_short, notes,
                    )
                    if umap_path:
                        files.append(umap_path)
                except Exception as e_umap:
                    logger.warning(f"  {side_label} activity UMAP failed (non-fatal): {e_umap}")

        except Exception as e:
            logger.error(f"  Activity assessment failed: {e}")
            errors.append(f"Activity mAP failed for {organelle_name} vs {live_experiment}: {e}")
            return _empty

        # --- 2. Phenotypic Distinctiveness (guide level) ---
        try:
            t0 = time.time()
            logger.info(
                f"  --- [{organelle_name.upper()}] 2/4 Phenotypic Distinctiveness ---"
            )
            cp_distinct, cp_distinct_ratio = phenotypic_distinctivness(
                cp_guide_adata, cp_activity, plot_results=False
            )
            live_distinct, live_distinct_ratio = phenotypic_distinctivness(
                live_guide_adata, live_activity, plot_results=False
            )
            logger.info(f"  Distinctiveness done in {time.time() - t0:.1f}s")
            comparison.update({
                "cp_distinctive_ratio": cp_distinct_ratio,
                "live_distinctive_ratio": live_distinct_ratio,
                "cp_mean_mAP_distinctiveness": cp_distinct["mean_average_precision"].mean(),
                "live_mean_mAP_distinctiveness": live_distinct["mean_average_precision"].mean(),
                "cp_auc_distinctiveness": compute_auc_score(cp_distinct),
                "live_auc_distinctiveness": compute_auc_score(live_distinct),
                "cp_sweep_auc_distinctiveness": compute_threshold_sweep_auc(cp_distinct),
                "live_sweep_auc_distinctiveness": compute_threshold_sweep_auc(live_distinct),
            })

            for df in [cp_distinct, live_distinct]:
                p_clamped = np.clip(df["corrected_p_value"].values.astype(np.float64), 1e-6, 1.0)
                w = -np.log10(p_clamped) / 6.0
                df["auc_contribution"] = df["mean_average_precision"].values * w

            cp_dist_path = org_output_dir / f"cp_distinctiveness_{live_short}.csv"
            live_dist_path = org_output_dir / f"livecell_{live_short}_distinctiveness.csv"
            cp_distinct.to_csv(cp_dist_path, index=False)
            live_distinct.to_csv(live_dist_path, index=False)
            files.extend([cp_dist_path, live_dist_path])

            # Generate distinctiveness plot immediately
            self._plot_step_scatter(
                cp_distinct, live_distinct, cp_distinct_ratio, live_distinct_ratio,
                "distinctiveness", organelle_name, cp_org_name, live_org_name,
                live_experiment, org_output_dir, live_short, notes, files,
                n_cells_used=n_cells_used,
            )

            # Distinctiveness UMAPs
            for side_label, side_adata, side_map in [
                ("CP", cp_guide_adata, cp_distinct),
                ("Live", live_guide_adata, live_distinct),
            ]:
                try:
                    umap_path = self._plot_metric_umap(
                        side_adata, side_map, "distinctiveness", side_label,
                        organelle_name, org_output_dir, live_short, notes,
                    )
                    if umap_path:
                        files.append(umap_path)
                except Exception as e_umap:
                    logger.warning(f"  {side_label} distinctiveness UMAP failed (non-fatal): {e_umap}")

        except Exception as e:
            logger.warning(f"  Distinctiveness failed (non-fatal): {e}")
            comparison.update({
                "cp_distinctive_ratio": np.nan, "live_distinctive_ratio": np.nan,
                "cp_mean_mAP_distinctiveness": np.nan, "live_mean_mAP_distinctiveness": np.nan,
                "cp_auc_distinctiveness": np.nan, "live_auc_distinctiveness": np.nan,
                "cp_sweep_auc_distinctiveness": np.nan, "live_sweep_auc_distinctiveness": np.nan,
            })

        # --- 3. Phenotypic Consistency - CORUM (gene level) ---
        try:
            t0 = time.time()
            logger.info(
                f"  --- [{organelle_name.upper()}] 3/4 CORUM Consistency ---"
            )
            cp_corum, cp_corum_ratio = phenotypic_consistency_corum(
                cp_gene_adata, cp_activity, plot_results=False
            )
            live_corum, live_corum_ratio = phenotypic_consistency_corum(
                live_gene_adata, live_activity, plot_results=False
            )
            logger.info(f"  CORUM done in {time.time() - t0:.1f}s")
            comparison.update({
                "cp_consistency_corum_ratio": cp_corum_ratio,
                "live_consistency_corum_ratio": live_corum_ratio,
                "cp_mean_mAP_corum": cp_corum["mean_average_precision"].mean(),
                "live_mean_mAP_corum": live_corum["mean_average_precision"].mean(),
                "cp_auc_corum": compute_auc_score(cp_corum),
                "live_auc_corum": compute_auc_score(live_corum),
                "cp_sweep_auc_corum": compute_threshold_sweep_auc(cp_corum),
                "live_sweep_auc_corum": compute_threshold_sweep_auc(live_corum),
            })

            for df in [cp_corum, live_corum]:
                p_clamped = np.clip(df["corrected_p_value"].values.astype(np.float64), 1e-6, 1.0)
                w = -np.log10(p_clamped) / 6.0
                df["auc_contribution"] = df["mean_average_precision"].values * w

            cp_cor_path = org_output_dir / f"cp_consistency_corum_{live_short}.csv"
            live_cor_path = org_output_dir / f"livecell_{live_short}_consistency_corum.csv"
            cp_corum.to_csv(cp_cor_path, index=False)
            live_corum.to_csv(live_cor_path, index=False)
            files.extend([cp_cor_path, live_cor_path])

            # Generate CORUM plot immediately
            self._plot_step_scatter(
                cp_corum, live_corum, cp_corum_ratio, live_corum_ratio,
                "corum", organelle_name, cp_org_name, live_org_name,
                live_experiment, org_output_dir, live_short, notes, files,
                n_cells_used=n_cells_used,
            )

        except Exception as e:
            logger.warning(f"  CORUM consistency failed (non-fatal): {e}")
            comparison.update({
                "cp_consistency_corum_ratio": np.nan, "live_consistency_corum_ratio": np.nan,
                "cp_mean_mAP_corum": np.nan, "live_mean_mAP_corum": np.nan,
                "cp_auc_corum": np.nan, "live_auc_corum": np.nan,
                "cp_sweep_auc_corum": np.nan, "live_sweep_auc_corum": np.nan,
            })

        # --- 4. Phenotypic Consistency - CHAD Annotations (gene level) ---
        try:
            t0 = time.time()
            logger.info(
                f"  --- [{organelle_name.upper()}] 4/4 CHAD Annotation Consistency ---"
            )
            cp_manual, cp_manual_ratio = phenotypic_consistency_manual_annotation(
                cp_gene_adata, cp_activity, plot_results=False
            )
            live_manual, live_manual_ratio = phenotypic_consistency_manual_annotation(
                live_gene_adata, live_activity, plot_results=False
            )
            logger.info(f"  CHAD done in {time.time() - t0:.1f}s")
            comparison.update({
                "cp_consistency_manual_ratio": cp_manual_ratio,
                "live_consistency_manual_ratio": live_manual_ratio,
                "cp_mean_mAP_manual": cp_manual["mean_average_precision"].mean(),
                "live_mean_mAP_manual": live_manual["mean_average_precision"].mean(),
                "cp_auc_manual": compute_auc_score(cp_manual),
                "live_auc_manual": compute_auc_score(live_manual),
                "cp_sweep_auc_manual": compute_threshold_sweep_auc(cp_manual),
                "live_sweep_auc_manual": compute_threshold_sweep_auc(live_manual),
            })

            for df in [cp_manual, live_manual]:
                p_clamped = np.clip(df["corrected_p_value"].values.astype(np.float64), 1e-6, 1.0)
                w = -np.log10(p_clamped) / 6.0
                df["auc_contribution"] = df["mean_average_precision"].values * w

            cp_man_path = org_output_dir / f"cp_consistency_manual_{live_short}.csv"
            live_man_path = org_output_dir / f"livecell_{live_short}_consistency_manual.csv"
            cp_manual.to_csv(cp_man_path, index=False)
            live_manual.to_csv(live_man_path, index=False)
            files.extend([cp_man_path, live_man_path])

            # Generate CHAD annotation plot immediately
            self._plot_step_scatter(
                cp_manual, live_manual, cp_manual_ratio, live_manual_ratio,
                "manual", organelle_name, cp_org_name, live_org_name,
                live_experiment, org_output_dir, live_short, notes, files,
                n_cells_used=n_cells_used,
            )

        except Exception as e:
            logger.warning(f"  CHAD consistency failed (non-fatal): {e}")
            comparison.update({
                "cp_consistency_manual_ratio": np.nan, "live_consistency_manual_ratio": np.nan,
                "cp_mean_mAP_manual": np.nan, "live_mean_mAP_manual": np.nan,
                "cp_auc_manual": np.nan, "live_auc_manual": np.nan,
                "cp_sweep_auc_manual": np.nan, "live_sweep_auc_manual": np.nan,
            })

        # Determine winner based on activity
        comparison["winner_activity"] = (
            "CP" if comparison.get("cp_active_ratio", 0) > comparison.get("live_active_ratio", 0)
            else "Live-Cell"
        )
        comparison["winner_auc_activity"] = (
            "CP" if comparison.get("cp_auc_activity", 0) > comparison.get("live_auc_activity", 0)
            else "Live-Cell"
        )

        logger.info(
            f"  Activity:        CP={comparison.get('cp_active_ratio', 0):.1%} "
            f"({comparison.get('cp_n_active', 0)} genes), "
            f"Live={comparison.get('live_active_ratio', 0):.1%} "
            f"({comparison.get('live_n_active', 0)} genes) "
            f"-> {comparison['winner_activity']}"
        )
        logger.info(
            f"  AUC Activity:    CP={comparison.get('cp_auc_activity', float('nan')):.4f}, "
            f"Live={comparison.get('live_auc_activity', float('nan')):.4f} "
            f"-> {comparison['winner_auc_activity']}"
        )
        logger.info(
            f"  Sweep AUC:       CP={comparison.get('cp_sweep_auc_activity', float('nan')):.4f}, "
            f"Live={comparison.get('live_sweep_auc_activity', float('nan')):.4f}"
        )
        logger.info(
            f"  Distinctiveness: CP={comparison.get('cp_distinctive_ratio', float('nan'))}, "
            f"Live={comparison.get('live_distinctive_ratio', float('nan'))}"
        )
        logger.info(
            f"  CORUM:           CP={comparison.get('cp_consistency_corum_ratio', float('nan'))}, "
            f"Live={comparison.get('live_consistency_corum_ratio', float('nan'))}"
        )
        logger.info(
            f"  CHAD:            CP={comparison.get('cp_consistency_manual_ratio', float('nan'))}, "
            f"Live={comparison.get('live_consistency_manual_ratio', float('nan'))}"
        )

        logger.info(f"  Total comparison time: {time.time() - t_comparison_start:.1f}s")
        return {
            "comparison": comparison,
            "files": files,
            "errors": errors,
            "cp_activity": cp_activity,
            "live_activity": live_activity,
            "cp_active_ratio": cp_active_ratio,
            "live_active_ratio": live_active_ratio,
        }

    def _plot_step_scatter(
        self,
        cp_results: pd.DataFrame,
        live_results: pd.DataFrame,
        cp_ratio: float,
        live_ratio: float,
        metric_name: str,
        organelle_name: str,
        cp_org_name: str,
        live_org_name: str,
        live_experiment: str,
        org_output_dir: Path,
        live_short: str,
        notes: str,
        files: List[Path],
        n_cells_used: int = 0,
    ) -> None:
        """Generate a scatter plot for one metric step, appending path to files."""
        try:
            path = self._plot_comparison_scatter(
                cp_results, live_results,
                organelle_name, cp_org_name, live_org_name,
                live_experiment, cp_ratio, live_ratio,
                org_output_dir, live_short,
                metric_name=metric_name, notes=notes,
                n_cells_used=n_cells_used,
            )
            if path:
                files.append(path)
        except Exception as e:
            logger.warning(f"  {metric_name} plot failed (non-fatal): {e}")

    def _build_cell_df(
        self, adata: ad.AnnData, feature_cols: List[str]
    ) -> pd.DataFrame:
        """
        Build a cell-level DataFrame with only the specified organelle features + metadata.
        Includes NTC normalization matching feature_extraction_slurm.py.
        """
        # Get feature matrix for this organelle only
        valid_cols = [c for c in feature_cols if c in adata.var_names]
        if not valid_cols:
            raise ValueError(f"None of the feature columns found in AnnData: {feature_cols[:5]}...")

        feat_idx = [list(adata.var_names).index(c) for c in valid_cols]
        X = adata.X[:, feat_idx]
        if hasattr(X, "toarray"):
            X = X.toarray()

        features_df = pd.DataFrame(
            np.asarray(X, dtype=np.float32), columns=valid_cols
        )

        # Add metadata (convert categoricals to plain types to avoid groupby issues)
        meta_cols = ["gene_name", "sgRNA", "barcode", "NCBI_ID"]
        meta = adata.obs[[c for c in meta_cols if c in adata.obs.columns]].reset_index(drop=True)
        for col in meta.columns:
            if hasattr(meta[col], "cat"):
                meta[col] = meta[col].astype(str)

        cell_df = pd.concat([meta, features_df], axis=1)

        # NTC normalization (matching canonical lines 1390-1396)
        if "gene_name" in cell_df.columns and "NCBI_ID" in cell_df.columns:
            ntc_mask = cell_df["NCBI_ID"] == -1
            n_ntc = ntc_mask.sum()
            if n_ntc > 0:
                cell_df.loc[ntc_mask, "gene_name"] = "NTC"
                logger.info(f"  Normalized {n_ntc} NTC cells (NCBI_ID = -1)")

        # Drop cells with missing gene_name or sgRNA
        if "gene_name" in cell_df.columns:
            cell_df = cell_df.dropna(subset=["gene_name"])
            cell_df = cell_df[cell_df["gene_name"].astype(str) != "None"]

        return cell_df.reset_index(drop=True)

    # -------------------------------------------------------------------------
    # Plotting
    # -------------------------------------------------------------------------

    def _plot_comparison_scatter(
        self,
        cp_results: pd.DataFrame,
        live_results: pd.DataFrame,
        organelle_name: str,
        cp_org_name: str,
        live_org_name: str,
        live_experiment: str,
        cp_ratio: float,
        live_ratio: float,
        output_dir: Path,
        live_short: str,
        metric_name: str = "activity",
        notes: str = "",
        n_cells_used: int = 0,
    ) -> Optional[Path]:
        """Overlaid mAP scatter plot: CP and live-cell on the same axes."""
        fig, ax = plt.subplots(figsize=(10, 7))

        cp_active = cp_results["below_corrected_p"] == True
        live_active = live_results["below_corrected_p"] == True

        # Plot inactive points (lighter, behind)
        ax.scatter(
            cp_results.loc[~cp_active, "mean_average_precision"],
            cp_results.loc[~cp_active, "-log10(p-value)"],
            c="#4C72B0", s=18, alpha=0.2, marker="o",
        )
        ax.scatter(
            live_results.loc[~live_active, "mean_average_precision"],
            live_results.loc[~live_active, "-log10(p-value)"],
            c="#DD8452", s=18, alpha=0.2, marker="s",
        )

        # Build live-cell label — always include experiment number
        live_label = f"Live {live_short} — {notes}" if notes else f"Live ({live_short})"

        # Plot active points (bold, on top)
        ax.scatter(
            cp_results.loc[cp_active, "mean_average_precision"],
            cp_results.loc[cp_active, "-log10(p-value)"],
            c="#4C72B0", s=40, alpha=0.9, marker="o", edgecolors="black", linewidths=0.5,
            label=f"CP {cp_org_name} ({cp_active.sum()}, {cp_ratio:.1%})",
        )
        ax.scatter(
            live_results.loc[live_active, "mean_average_precision"],
            live_results.loc[live_active, "-log10(p-value)"],
            c="#DD8452", s=40, alpha=0.9, marker="s", edgecolors="black", linewidths=0.5,
            label=f"{live_label} ({live_active.sum()}, {live_ratio:.1%})",
        )

        # Iso-score contour background showing AUC contribution landscape
        x_max = max(
            cp_results["mean_average_precision"].max(),
            live_results["mean_average_precision"].max(),
            0.5,
        )
        y_max = max(
            cp_results["-log10(p-value)"].max(),
            live_results["-log10(p-value)"].max(),
            2.0,
        )
        mAP_grid = np.linspace(0, x_max * 1.1, 80)
        nlp_grid = np.linspace(0, y_max * 1.1, 80)
        MAP_G, NLP_G = np.meshgrid(mAP_grid, nlp_grid)
        SCORE_G = MAP_G * (NLP_G / 6.0)
        ax.contourf(MAP_G, NLP_G, SCORE_G, levels=8, cmap="Greys", alpha=0.12, zorder=0)

        # AUC score annotations with delta % and color
        cp_auc = compute_auc_score(cp_results)
        live_auc = compute_auc_score(live_results)
        cp_sweep = compute_threshold_sweep_auc(cp_results)
        live_sweep = compute_threshold_sweep_auc(live_results)

        def _delta_str(cp_v, live_v):
            """Format delta as signed %, colored: green=CP wins, red=Live wins."""
            avg = (cp_v + live_v) / 2 if (cp_v + live_v) != 0 else 1
            pct = (cp_v - live_v) / avg * 100
            sign = "+" if pct >= 0 else ""
            return f"{sign}{pct:.0f}%", "#2d8a4e" if pct >= 0 else "#c0392b"

        auc_delta, auc_color = _delta_str(cp_auc, live_auc)
        sweep_delta, sweep_color = _delta_str(cp_sweep, live_sweep)

        # Score box
        ax.annotate(
            f"AUC: CP={cp_auc:.3f}, Live={live_auc:.3f}\n"
            f"Sweep AUC: CP={cp_sweep:.3f}, Live={live_sweep:.3f}",
            xy=(0.98, 0.02), xycoords="axes fraction",
            ha="right", va="bottom", fontsize=13,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
        )
        # Delta annotations (colored, right-aligned above the score box)
        ax.annotate(
            f"CP vs Live:  AUC {auc_delta}  |  Sweep {sweep_delta}",
            xy=(0.98, 0.15), xycoords="axes fraction",
            ha="right", va="bottom", fontsize=13, fontweight="bold",
            color=auc_color if auc_color == sweep_color else "black",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.85),
        )

        metric_title = metric_name.replace("_", " ").title()
        subtitle = f"Live {live_short}: {notes}" if notes else f"Live: {live_org_name} ({live_short})"
        cells_str = f" | {n_cells_used:,} cells" if n_cells_used > 0 else ""
        ax.axhline(-np.log10(0.05), color="black", linestyle="--", alpha=0.4, label="p=0.05")
        ax.set_xlabel("Mean Average Precision (mAP)", fontsize=12)
        ax.set_ylabel("-log10(corrected p-value)", fontsize=12)
        ax.set_title(
            f"{organelle_name.replace('_', ' ').title()} — {metric_title}{cells_str}\n{subtitle}",
            fontsize=13, fontweight="bold",
        )
        ax.legend(loc="upper left", fontsize=14, framealpha=0.9)

        plt.tight_layout()
        path = save_figure(fig, output_dir / f"cp_vs_{live_short}_{metric_name}_scatter.png")
        return path

    def _plot_active_gene_overlap(
        self,
        cp_activity: pd.DataFrame,
        live_activity: pd.DataFrame,
        organelle_name: str,
        live_experiment: str,
        output_dir: Path,
        live_short: str,
        notes: str = "",
        n_cells_used: int = 0,
    ) -> Optional[Path]:
        """Venn-style visualization of active gene overlap between CP and live-cell."""
        cp_active = set(
            cp_activity[cp_activity["below_corrected_p"]]["perturbation"]
        )
        live_active = set(
            live_activity[live_activity["below_corrected_p"]]["perturbation"]
        )

        cp_only = cp_active - live_active
        live_only = live_active - cp_active
        both = cp_active & live_active
        union = cp_active | live_active
        jaccard = len(both) / len(union) if union else 0

        fig, axes = plt.subplots(1, 2, figsize=(14, 5), gridspec_kw={"width_ratios": [1, 1.2]})

        # Left panel: stacked horizontal bars showing composition
        ax = axes[0]
        bar_height = 0.5
        y_positions = [1, 0]

        # CP bar
        ax.barh(y_positions[0], len(cp_only), bar_height, color="#4C72B0", edgecolor="black",
                linewidth=0.5, label="Unique")
        ax.barh(y_positions[0], len(both), bar_height, left=len(cp_only),
                color="#55A868", edgecolor="black", linewidth=0.5, label="Shared")

        # Live bar
        ax.barh(y_positions[1], len(live_only), bar_height, color="#DD8452", edgecolor="black",
                linewidth=0.5)
        ax.barh(y_positions[1], len(both), bar_height, left=len(live_only),
                color="#55A868", edgecolor="black", linewidth=0.5)

        # Labels on bars
        for y, unique_n, shared_n, total_n in [
            (y_positions[0], len(cp_only), len(both), len(cp_active)),
            (y_positions[1], len(live_only), len(both), len(live_active)),
        ]:
            if unique_n > 0:
                ax.text(unique_n / 2, y, str(unique_n), ha="center", va="center",
                        fontsize=11, fontweight="bold", color="white")
            if shared_n > 0:
                ax.text(unique_n + shared_n / 2, y, str(shared_n), ha="center", va="center",
                        fontsize=11, fontweight="bold", color="white")
            ax.text(total_n + 0.3, y, f"({total_n} total)", ha="left", va="center", fontsize=10)

        live_label = f"Live-Cell {live_short} — {notes}" if notes else f"Live-Cell ({live_short})"
        ax.set_yticks(y_positions)
        ax.set_yticklabels(["Cell Painting", live_label], fontsize=11)
        ax.set_xlabel("Number of active genes", fontsize=11)
        ax.legend(loc="lower right", fontsize=9)

        # Right panel: list top shared and unique genes
        ax2 = axes[1]
        ax2.axis("off")

        text_lines = [f"Jaccard similarity: {jaccard:.2f}"]
        text_lines.append(f"\nShared ({len(both)}): " + ", ".join(sorted(both)[:15]))
        if len(both) > 15:
            text_lines[-1] += f" ... (+{len(both) - 15} more)"
        text_lines.append(f"\nCP-only ({len(cp_only)}): " + ", ".join(sorted(cp_only)[:10]))
        if len(cp_only) > 10:
            text_lines[-1] += f" ... (+{len(cp_only) - 10} more)"
        text_lines.append(f"\nLive-only ({len(live_only)}): " + ", ".join(sorted(live_only)[:10]))
        if len(live_only) > 10:
            text_lines[-1] += f" ... (+{len(live_only) - 10} more)"

        ax2.text(0.05, 0.95, "\n".join(text_lines), transform=ax2.transAxes,
                 va="top", ha="left", fontsize=9, family="monospace",
                 bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8))

        subtitle = f"Live {live_short}: {notes}" if notes else f"CP vs {live_short}"
        cells_str = f" | {n_cells_used:,} cells" if n_cells_used > 0 else ""
        fig.suptitle(
            f"{organelle_name.replace('_', ' ').title()}: Active Gene Overlap{cells_str}\n{subtitle}",
            fontsize=13, fontweight="bold",
        )

        plt.tight_layout()
        path = save_figure(fig, output_dir / f"cp_vs_{live_short}_overlap.png")
        return path

    def _plot_overall_comparison(
        self, summary_df: pd.DataFrame, result: StageResult,
        subdir_prefix: str = "",
        title_extra: str = "",
    ) -> None:
        """Generate overall comparison plots across all organelles.

        Produces plots for 3 scoring metrics (ratio, weighted AUC, sweep AUC),
        each in its own subdirectory for easy navigation.

        Parameters
        ----------
        subdir_prefix : str
            Prefix for output subdirectory names (e.g. "control_" -> "control_ratio_plots/").
        title_extra : str
            Extra suffix appended to plot titles (e.g. " (Control)").
        """
        saved_output_dir = self._output_dir

        # Define the 3 metric scoring systems and their plot configs
        metric_configs = [
            {
                "subdir": f"{subdir_prefix}ratio_plots",
                "pairs": self._METRIC_PAIRS,
                "map_pairs": self._METRIC_MAP_PAIRS,
                "title": f" (Active Ratio){title_extra}",
                "y_label": "Ratio (%)", "x_label": "Ratio (%)",
                "scale": 100, "fmt": "{:.1f}%",
                "cbar_label": "Ratio (%)", "annot_fmt": ".1f",
                "scatter_x": "Cell Painting — mean mAP",
                "scatter_y": "Live-Cell — mean mAP",
            },
            {
                "subdir": f"{subdir_prefix}auc_plots",
                "pairs": self._METRIC_AUC_PAIRS,
                "map_pairs": self._METRIC_AUC_PAIRS,
                "title": f" (Weighted AUC){title_extra}",
                "y_label": "AUC Score", "x_label": "AUC Score",
                "scale": 1, "fmt": "{:.3f}",
                "cbar_label": "AUC Score", "annot_fmt": ".3f",
                "scatter_x": "Cell Painting — AUC",
                "scatter_y": "Live-Cell — AUC",
            },
            {
                "subdir": f"{subdir_prefix}sweep_auc_plots",
                "pairs": self._METRIC_SWEEP_AUC_PAIRS,
                "map_pairs": self._METRIC_SWEEP_AUC_PAIRS,
                "title": f" (Sweep AUC){title_extra}",
                "y_label": "Sweep AUC Score", "x_label": "Sweep AUC Score",
                "scale": 1, "fmt": "{:.3f}",
                "cbar_label": "Sweep AUC Score", "annot_fmt": ".3f",
                "scatter_x": "Cell Painting — Sweep AUC",
                "scatter_y": "Live-Cell — Sweep AUC",
            },
        ]

        for cfg in metric_configs:
            subdir = saved_output_dir / cfg["subdir"]
            subdir.mkdir(parents=True, exist_ok=True)
            self._output_dir = subdir

            self._plot_grouped_bar(
                summary_df, result,
                metric_pairs=cfg["pairs"], title_suffix=cfg["title"],
                filename="comparison_bar_chart.png",
                y_label=cfg["y_label"], scale=cfg["scale"], fmt=cfg["fmt"],
            )
            self._plot_heatmap(
                summary_df, result,
                metric_pairs=cfg["pairs"], title_suffix=cfg["title"],
                filename="comparison_heatmap.png",
                scale=cfg["scale"], annot_fmt=cfg["annot_fmt"],
                cbar_label=cfg["cbar_label"],
            )
            self._plot_winner_summary(
                summary_df, result,
                metric_pairs=cfg["pairs"], title_suffix=cfg["title"],
                filename="winner_summary.png",
                scale=cfg["scale"],
                y_label=f"CP - Live-Cell Difference ({cfg['y_label']})",
            )
            self._plot_radar_charts(
                summary_df, result,
                metric_pairs=cfg["pairs"], title_suffix=cfg["title"],
                filename="radar_charts.png", scale=cfg["scale"],
            )
            self._plot_radar_by_metric(
                summary_df, result,
                metric_pairs=cfg["pairs"], title_suffix=cfg["title"],
                filename="radar_by_metric.png", scale=cfg["scale"],
            )
            self._plot_paired_dots(
                summary_df, result,
                metric_pairs=cfg["pairs"], title_suffix=cfg["title"],
                filename="paired_dot_comparison.png",
                x_label=cfg["x_label"], scale=cfg["scale"],
            )
            self._plot_mean_map_scatter(
                summary_df, result,
                metric_pairs=cfg["map_pairs"], title_suffix=cfg["title"],
                filename="scatter_comparison.png",
                x_label=cfg["scatter_x"], y_label=cfg["scatter_y"],
            )

        self._output_dir = saved_output_dir

    @classmethod
    def _compute_normalized_summary(
        cls,
        summary_df: pd.DataFrame,
        control_summary_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """Batch-correct real metrics using control comparisons as reference.

        For each metric pair (cp_col, live_col), the control comparison
        (identical channels on both sides) reveals the per-experiment batch
        bias.  We compute the geometric mean of the two control values as
        a neutral reference, then scale each side toward that reference:

            geo = sqrt(ctrl_cp * ctrl_live)
            corrected_cp  = real_cp  * (geo / ctrl_cp)
            corrected_live = real_live * (geo / ctrl_live)

        This symmetrically removes the batch-driven asymmetry while
        preserving the overall signal magnitude and the paired CP/Live
        structure so all existing plot methods work unchanged.

        Returns a DataFrame with only those rows that had a matching control.
        """
        # Collect metric pairs: (cp_col, live_col) across all 4 scoring systems
        metric_pairs: list[tuple[str, str]] = []
        for pairs in [
            cls._METRIC_PAIRS,
            cls._METRIC_AUC_PAIRS,
            cls._METRIC_SWEEP_AUC_PAIRS,
            cls._METRIC_MAP_PAIRS,
        ]:
            for _, cp_col, live_col in pairs:
                metric_pairs.append((cp_col, live_col))

        all_metric_cols = [c for pair in metric_pairs for c in pair]

        # Keep only join key + metric columns from control (drop organelle metadata)
        ctrl_cols = ["live_experiment"] + [
            c for c in all_metric_cols if c in control_summary_df.columns
        ]
        ctrl = control_summary_df[ctrl_cols].drop_duplicates(
            subset=["live_experiment"], keep="first",
        )

        # Merge real ← control on live_experiment
        merged = summary_df.merge(
            ctrl, on="live_experiment", how="left", suffixes=("", "_ctrl"),
        )

        # Batch-correct each metric pair using geometric-mean reference
        MIN_CTRL = 0.005   # floor for control values to avoid division by ~0
        MAX_FACTOR = 5.0   # cap correction factor to prevent extreme swings

        for cp_col, live_col in metric_pairs:
            ctrl_cp_col = f"{cp_col}_ctrl"
            ctrl_live_col = f"{live_col}_ctrl"
            if ctrl_cp_col not in merged.columns or ctrl_live_col not in merged.columns:
                continue

            ctrl_cp = merged[ctrl_cp_col].abs().clip(lower=MIN_CTRL)
            ctrl_live = merged[ctrl_live_col].abs().clip(lower=MIN_CTRL)
            geo_mean = np.sqrt(ctrl_cp * ctrl_live)

            cp_factor = (geo_mean / ctrl_cp).clip(1 / MAX_FACTOR, MAX_FACTOR)
            live_factor = (geo_mean / ctrl_live).clip(1 / MAX_FACTOR, MAX_FACTOR)

            merged[cp_col] = merged[cp_col] * cp_factor
            merged[live_col] = merged[live_col] * live_factor

        # Drop _ctrl helper columns
        ctrl_drop = [c for c in merged.columns if c.endswith("_ctrl")]
        merged.drop(columns=ctrl_drop, inplace=True)

        # Drop rows where the merge found no control match (NaN metrics)
        merged.dropna(subset=[all_metric_cols[0]], inplace=True)

        # Strip any existing "(Batch-Corrected)" from notes — the plot title
        # already indicates batch correction via title_extra, so per-spoke
        # labels don't need it.
        if "notes" in merged.columns:
            merged["notes"] = merged["notes"].fillna("").astype(str)

        return merged.reset_index(drop=True)

    @staticmethod
    def _comparison_label(row: pd.Series, sep: str = "\n") -> str:
        """Build a consistent comparison label including experiment number and cell count."""
        live_short = row["live_experiment"].split("_")[0] if "_" in str(row["live_experiment"]) else row["live_experiment"]
        notes = row.get("notes", "")
        n_cells = row.get("n_cells_used", 0)
        cells_str = f"\n{n_cells:,} cells" if n_cells and n_cells > 0 else ""
        if notes:
            return f"{row['organelle_type']}{sep}vs {live_short}{sep}({notes}){cells_str}"
        return f"{row['organelle_type']}{sep}vs {live_short}{cells_str}"

    def _plot_grouped_bar(
        self, summary_df: pd.DataFrame, result: StageResult,
        metric_pairs: Optional[List] = None, title_suffix: str = "",
        filename: str = "comparison_bar_chart.png",
        y_label: str = "Ratio (%)", scale: float = 100, fmt: str = "{:.1f}%",
    ) -> None:
        """4-panel grouped bar chart: one row per metric, CP vs Live side-by-side."""
        if metric_pairs is None:
            metric_pairs = self._METRIC_PAIRS
        metric_pairs = [p for p in metric_pairs if p[1] in summary_df.columns]
        if not metric_pairs:
            return

        labels = [self._comparison_label(row) for _, row in summary_df.iterrows()]
        x = np.arange(len(labels))
        width = 0.35
        n_metrics = len(metric_pairs)

        fig, axes = plt.subplots(
            n_metrics, 1,
            figsize=(max(12, len(labels) * 1.5), 4.5 * n_metrics),
            sharex=True,
        )
        if n_metrics == 1:
            axes = [axes]

        for ax, (metric_label, cp_col, live_col) in zip(axes, metric_pairs):
            cp_vals = summary_df[cp_col].fillna(0) * scale
            live_vals = summary_df[live_col].fillna(0) * scale

            bars_cp = ax.bar(
                x - width / 2, cp_vals,
                width, label="Cell Painting", color="#4C72B0", edgecolor="black", linewidth=0.5,
            )
            bars_live = ax.bar(
                x + width / 2, live_vals,
                width, label="Live-Cell", color="#DD8452", edgecolor="black", linewidth=0.5,
            )

            # Value labels
            for bars in [bars_cp, bars_live]:
                for bar in bars:
                    h = bar.get_height()
                    if h > 0:
                        ax.text(
                            bar.get_x() + bar.get_width() / 2, h + 0.3,
                            fmt.format(h), ha="center", va="bottom", fontsize=8,
                        )

            max_val = max(cp_vals.max(), live_vals.max())
            ax.set_ylim(0, max_val * 1.15 + 3 if scale == 100 else max_val * 1.2 + 0.01)
            ax.set_ylabel(y_label, fontsize=10)
            ax.set_title(metric_label, fontsize=13, fontweight="bold")
            ax.legend(fontsize=9, loc="upper right")

        axes[-1].set_xticks(x)
        axes[-1].set_xticklabels(labels, fontsize=8)

        fig.suptitle(
            f"CP vs Live-Cell: All Metrics by Comparison{title_suffix}",
            fontsize=15, fontweight="bold", y=1.01,
        )
        plt.tight_layout()
        path = save_figure(fig, self.output_dir / filename)
        result.add_file(path)

    def _plot_heatmap(
        self, summary_df: pd.DataFrame, result: StageResult,
        metric_pairs: Optional[List] = None, title_suffix: str = "",
        filename: str = "comparison_heatmap.png",
        scale: float = 100, annot_fmt: str = ".1f", cbar_label: str = "Ratio (%)",
    ) -> None:
        """Heatmap of metric values by organelle comparison."""
        if metric_pairs is None:
            metric_pairs = self._METRIC_PAIRS
        labels = [self._comparison_label(row, sep=" ") for _, row in summary_df.iterrows()]

        col_labels = []
        data_cols = []
        for metric_label, cp_col, live_col in metric_pairs:
            if cp_col in summary_df.columns:
                cp_vals = summary_df[cp_col].values * scale
                live_vals = summary_df[live_col].values * scale
                data_cols.append(cp_vals)
                data_cols.append(live_vals)
                col_labels.append(f"CP\n{metric_label}")
                col_labels.append(f"Live\n{metric_label}")

        if not data_cols:
            return

        heatmap_data = pd.DataFrame(
            np.column_stack(data_cols), index=labels, columns=col_labels
        )

        fig, ax = plt.subplots(figsize=(max(10, len(col_labels) * 1.2), max(4, len(labels) * 0.6)))
        sns.heatmap(
            heatmap_data, annot=True, fmt=annot_fmt, cmap="RdYlGn",
            ax=ax, cbar_kws={"label": cbar_label},
            linewidths=0.5, linecolor="white",
        )
        ax.set_title(
            f"All Metrics: CP vs Live-Cell{title_suffix}",
            fontsize=13, fontweight="bold",
        )
        ax.set_xlabel("")
        ax.set_ylabel("")

        plt.tight_layout()
        path = save_figure(fig, self.output_dir / filename)
        result.add_file(path)

    def _plot_winner_summary(
        self, summary_df: pd.DataFrame, result: StageResult,
        metric_pairs: Optional[List] = None, title_suffix: str = "",
        filename: str = "winner_summary.png",
        scale: float = 100, y_label: str = "CP - Live-Cell Difference (%)",
    ) -> None:
        """Summary plot showing CP vs Live-Cell delta for all 4 metrics."""
        if metric_pairs is None:
            metric_pairs = [
                ("Activity", "cp_active_ratio", "live_active_ratio"),
                ("Distinctiveness", "cp_distinctive_ratio", "live_distinctive_ratio"),
                ("CORUM", "cp_consistency_corum_ratio", "live_consistency_corum_ratio"),
                ("CHAD", "cp_consistency_manual_ratio", "live_consistency_manual_ratio"),
            ]

        labels = [self._comparison_label(row) for _, row in summary_df.iterrows()]

        n_comparisons = len(labels)
        n_metrics = len(metric_pairs)
        x = np.arange(n_comparisons)
        bar_width = 0.8 / n_metrics
        metric_colors = ["#4C72B0", "#55A868", "#C44E52", "#8172B2"]

        fig, ax = plt.subplots(figsize=(max(12, n_comparisons * 2), 6))

        for i, (metric_label, cp_col, live_col) in enumerate(metric_pairs):
            if cp_col not in summary_df.columns:
                continue
            deltas = (summary_df[cp_col].fillna(0) - summary_df[live_col].fillna(0)) * scale
            offset = (i - n_metrics / 2 + 0.5) * bar_width
            ax.bar(
                x + offset, deltas, bar_width,
                label=metric_label, color=metric_colors[i % len(metric_colors)],
                edgecolor="black", linewidth=0.5, alpha=0.85,
            )

        ax.axhline(0, color="black", linewidth=1)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=9)
        ax.set_ylabel(y_label, fontsize=11)
        ax.set_title(
            f"CP vs Live-Cell: All Metrics (positive = CP wins){title_suffix}",
            fontsize=13, fontweight="bold",
        )
        ax.legend(fontsize=10, loc="best")

        plt.tight_layout()
        path = save_figure(fig, self.output_dir / filename)
        result.add_file(path)

    # -- Metric pairs used across multiple aggregate plots --
    _METRIC_PAIRS = [
        ("Activity", "cp_active_ratio", "live_active_ratio"),
        ("Distinctiveness", "cp_distinctive_ratio", "live_distinctive_ratio"),
        ("CORUM", "cp_consistency_corum_ratio", "live_consistency_corum_ratio"),
        ("CHAD", "cp_consistency_manual_ratio", "live_consistency_manual_ratio"),
    ]
    _METRIC_MAP_PAIRS = [
        ("Activity", "cp_mean_mAP_activity", "live_mean_mAP_activity"),
        ("Distinctiveness", "cp_mean_mAP_distinctiveness", "live_mean_mAP_distinctiveness"),
        ("CORUM", "cp_mean_mAP_corum", "live_mean_mAP_corum"),
        ("CHAD", "cp_mean_mAP_manual", "live_mean_mAP_manual"),
    ]
    _METRIC_AUC_PAIRS = [
        ("Activity", "cp_auc_activity", "live_auc_activity"),
        ("Distinctiveness", "cp_auc_distinctiveness", "live_auc_distinctiveness"),
        ("CORUM", "cp_auc_corum", "live_auc_corum"),
        ("CHAD", "cp_auc_manual", "live_auc_manual"),
    ]
    _METRIC_SWEEP_AUC_PAIRS = [
        ("Activity", "cp_sweep_auc_activity", "live_sweep_auc_activity"),
        ("Distinctiveness", "cp_sweep_auc_distinctiveness", "live_sweep_auc_distinctiveness"),
        ("CORUM", "cp_sweep_auc_corum", "live_sweep_auc_corum"),
        ("CHAD", "cp_sweep_auc_manual", "live_sweep_auc_manual"),
    ]

    def _plot_radar_charts(
        self, summary_df: pd.DataFrame, result: StageResult,
        metric_pairs: Optional[List] = None, title_suffix: str = "",
        filename: str = "radar_charts.png", scale: float = 100,
    ) -> None:
        """Radar chart per comparison: all 4 metrics overlaid for CP vs Live."""
        if metric_pairs is None:
            metric_pairs = self._METRIC_PAIRS
        metric_labels = []
        cp_cols = []
        live_cols = []
        for label, cp_col, live_col in metric_pairs:
            if cp_col in summary_df.columns:
                metric_labels.append(label)
                cp_cols.append(cp_col)
                live_cols.append(live_col)

        if len(metric_labels) < 3:
            return

        n_comparisons = len(summary_df)
        n_cols = min(n_comparisons, 3)
        n_rows = (n_comparisons + n_cols - 1) // n_cols

        fig, axes = plt.subplots(
            n_rows, n_cols, figsize=(5 * n_cols, 5 * n_rows),
            subplot_kw=dict(polar=True),
        )
        if n_comparisons == 1:
            axes = np.array([axes])
        axes = np.atleast_2d(axes)

        angles = np.linspace(0, 2 * np.pi, len(metric_labels), endpoint=False).tolist()
        angles += angles[:1]

        for idx, (_, row) in enumerate(summary_df.iterrows()):
            ax = axes[idx // n_cols, idx % n_cols]
            cp_vals = [row.get(c, 0) * scale for c in cp_cols] + [row.get(cp_cols[0], 0) * scale]
            live_vals = [row.get(c, 0) * scale for c in live_cols] + [row.get(live_cols[0], 0) * scale]

            ax.plot(angles, cp_vals, "o-", color="#4C72B0", linewidth=2, label="Cell Painting")
            ax.fill(angles, cp_vals, color="#4C72B0", alpha=0.15)
            ax.plot(angles, live_vals, "s-", color="#DD8452", linewidth=2, label="Live-Cell")
            ax.fill(angles, live_vals, color="#DD8452", alpha=0.15)

            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(metric_labels, fontsize=9)

            ax.set_title(
                self._comparison_label(row, sep="\n").replace("_", " ").title(),
                fontsize=11, fontweight="bold", pad=20,
            )
            ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=8)

        for idx in range(n_comparisons, n_rows * n_cols):
            axes[idx // n_cols, idx % n_cols].set_visible(False)

        fig.suptitle(
            f"CP vs Live-Cell: All Metrics Radar{title_suffix}",
            fontsize=14, fontweight="bold", y=1.02,
        )
        plt.tight_layout()
        path = save_figure(fig, self.output_dir / filename)
        result.add_file(path)

    def _plot_radar_by_metric(
        self, summary_df: pd.DataFrame, result: StageResult,
        metric_pairs: Optional[List] = None, title_suffix: str = "",
        filename: str = "radar_by_metric.png", scale: float = 100,
    ) -> None:
        """
        Radar chart per metric: spokes are comparisons, CP vs Live overlaid.

        Produces a 2x2 grid (Activity, Distinctiveness, CORUM, CHAD).
        Each radar has one spoke per organelle comparison.
        """
        if metric_pairs is None:
            metric_pairs = self._METRIC_PAIRS
        metric_pairs = [p for p in metric_pairs if p[1] in summary_df.columns]
        if len(metric_pairs) < 2:
            return

        spoke_labels = []
        for _, row in summary_df.iterrows():
            live_short = row["live_experiment"].split("_")[0] if "_" in str(row["live_experiment"]) else row["live_experiment"]
            notes = row.get("notes", "")
            n_cells = row.get("n_cells_used", 0)
            cells_str = f"\n{n_cells:,} cells" if n_cells and n_cells > 0 else ""
            if notes:
                spoke_labels.append(f"{row['organelle_type']}\n{live_short}\n{notes}{cells_str}")
            else:
                spoke_labels.append(f"{row['organelle_type']}\n({live_short}){cells_str}")

        n_spokes = len(spoke_labels)
        if n_spokes < 3:
            return

        angles = np.linspace(0, 2 * np.pi, n_spokes, endpoint=False).tolist()
        angles += angles[:1]

        n_metrics = len(metric_pairs)
        n_cols = min(n_metrics, 2)
        n_rows = (n_metrics + n_cols - 1) // n_cols

        fig, axes = plt.subplots(
            n_rows, n_cols, figsize=(7 * n_cols, 7 * n_rows),
            subplot_kw=dict(polar=True),
        )
        if n_metrics == 1:
            axes = np.array([[axes]])
        axes = np.atleast_2d(axes)

        for idx, (metric_label, cp_col, live_col) in enumerate(metric_pairs):
            ax = axes[idx // n_cols, idx % n_cols]

            cp_vals = (summary_df[cp_col].fillna(0) * scale).tolist()
            live_vals = (summary_df[live_col].fillna(0) * scale).tolist()
            cp_vals += cp_vals[:1]
            live_vals += live_vals[:1]

            ax.plot(angles, cp_vals, "o-", color="#4C72B0", linewidth=2, label="Cell Painting")
            ax.fill(angles, cp_vals, color="#4C72B0", alpha=0.15)
            ax.plot(angles, live_vals, "s-", color="#DD8452", linewidth=2, label="Live-Cell")
            ax.fill(angles, live_vals, color="#DD8452", alpha=0.15)

            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(spoke_labels, fontsize=8)
            ax.set_title(metric_label, fontsize=13, fontweight="bold", pad=20)
            ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.1), fontsize=9)

        for idx in range(n_metrics, n_rows * n_cols):
            axes[idx // n_cols, idx % n_cols].set_visible(False)

        fig.suptitle(
            f"CP vs Live-Cell: Comparisons by Metric{title_suffix}",
            fontsize=15, fontweight="bold", y=1.02,
        )
        plt.tight_layout()
        path = save_figure(fig, self.output_dir / filename)
        result.add_file(path)

    def _plot_paired_dots(
        self, summary_df: pd.DataFrame, result: StageResult,
        metric_pairs: Optional[List] = None, title_suffix: str = "",
        filename: str = "paired_dot_comparison.png",
        x_label: str = "Ratio (%)", scale: float = 100,
    ) -> None:
        """Paired dot (lollipop) plot: CP vs Live connected by lines for each metric."""
        if metric_pairs is None:
            metric_pairs = self._METRIC_PAIRS
        metric_pairs = [p for p in metric_pairs if p[1] in summary_df.columns]
        if not metric_pairs:
            return

        n_metrics = len(metric_pairs)
        fig, axes = plt.subplots(1, n_metrics, figsize=(4.5 * n_metrics, max(5, len(summary_df) * 0.6)))
        if n_metrics == 1:
            axes = [axes]

        for ax, (metric_label, cp_col, live_col) in zip(axes, metric_pairs):
            labels = [self._comparison_label(row) for _, row in summary_df.iterrows()]

            y = np.arange(len(labels))
            cp_vals = summary_df[cp_col].fillna(0).values * scale
            live_vals = summary_df[live_col].fillna(0).values * scale

            for i in range(len(y)):
                color = "#55A868" if cp_vals[i] >= live_vals[i] else "#C44E52"
                ax.plot([cp_vals[i], live_vals[i]], [y[i], y[i]],
                        color=color, linewidth=2, alpha=0.6, zorder=1)

            ax.scatter(cp_vals, y, s=80, color="#4C72B0", edgecolors="black",
                       linewidths=0.5, zorder=2, label="Cell Painting")
            ax.scatter(live_vals, y, s=80, color="#DD8452", edgecolors="black",
                       linewidths=0.5, zorder=2, marker="s", label="Live-Cell")

            ax.set_yticks(y)
            ax.set_yticklabels(labels, fontsize=9)
            ax.set_xlabel(x_label, fontsize=10)
            ax.set_title(metric_label, fontsize=12, fontweight="bold")
            ax.legend(fontsize=8, loc="lower right")
            ax.axvline(0, color="grey", linewidth=0.5, alpha=0.3)

        fig.suptitle(
            f"CP vs Live-Cell: Paired Comparison{title_suffix}",
            fontsize=14, fontweight="bold",
        )
        plt.tight_layout()
        path = save_figure(fig, self.output_dir / filename)
        result.add_file(path)

    def _plot_mean_map_scatter(
        self, summary_df: pd.DataFrame, result: StageResult,
        metric_pairs: Optional[List] = None, title_suffix: str = "",
        filename: str = "mean_map_scatter.png",
        x_label: str = "Cell Painting — mean mAP",
        y_label: str = "Live-Cell — mean mAP",
    ) -> None:
        """Scatter plot of CP vs Live scores for all 4 metrics."""
        if metric_pairs is None:
            metric_pairs = self._METRIC_MAP_PAIRS
        metric_pairs = [p for p in metric_pairs if p[1] in summary_df.columns]
        if not metric_pairs:
            return

        metric_colors = {"Activity": "#4C72B0", "Distinctiveness": "#55A868",
                         "CORUM": "#C44E52", "CHAD": "#8172B2"}
        metric_markers = {"Activity": "o", "Distinctiveness": "s",
                          "CORUM": "D", "CHAD": "^"}

        fig, ax = plt.subplots(figsize=(8, 8))

        all_vals = []
        for metric_label, cp_col, live_col in metric_pairs:
            cp_vals = summary_df[cp_col].dropna().values
            live_vals = summary_df[live_col].dropna().values
            if len(cp_vals) == 0:
                continue
            # Match lengths after dropna
            mask = summary_df[cp_col].notna() & summary_df[live_col].notna()
            cp_v = summary_df.loc[mask, cp_col].values
            live_v = summary_df.loc[mask, live_col].values
            all_vals.extend(cp_v)
            all_vals.extend(live_v)

            # Build labels for each point
            for i, (_, row) in enumerate(summary_df[mask].iterrows()):
                ax.annotate(
                    self._comparison_label(row, sep=" "), (cp_v[i], live_v[i]),
                    fontsize=7, alpha=0.7, textcoords="offset points",
                    xytext=(5, 5),
                )

            ax.scatter(
                cp_v, live_v, s=100,
                color=metric_colors.get(metric_label, "grey"),
                marker=metric_markers.get(metric_label, "o"),
                edgecolors="black", linewidths=0.5, alpha=0.85,
                label=metric_label, zorder=3,
            )

        # Diagonal parity line
        if all_vals:
            lo = min(all_vals) * 0.9
            hi = max(all_vals) * 1.1
            ax.plot([lo, hi], [lo, hi], "k--", alpha=0.4, linewidth=1, label="Parity")
            ax.fill_between([lo, hi], [lo, hi], [hi, hi],
                            color="#DD8452", alpha=0.04)
            ax.fill_between([lo, hi], [lo, lo], [lo, hi],
                            color="#4C72B0", alpha=0.04)
            # Region labels: above diagonal = Live wins (higher y), below = CP wins (higher x)
            ax.text(lo + (hi - lo) * 0.15, hi - (hi - lo) * 0.08,
                    "Live wins", fontsize=10, color="#DD8452", alpha=0.6, fontstyle="italic")
            ax.text(hi - (hi - lo) * 0.25, lo + (hi - lo) * 0.03,
                    "CP wins", fontsize=10, color="#4C72B0", alpha=0.6, fontstyle="italic")

        ax.set_xlabel(x_label, fontsize=12)
        ax.set_ylabel(y_label, fontsize=12)
        ax.set_title(
            f"CP vs Live-Cell (all metrics, all comparisons){title_suffix}",
            fontsize=13, fontweight="bold",
        )
        ax.legend(fontsize=10, loc="best")
        ax.set_aspect("equal", adjustable="datalim")

        plt.tight_layout()
        path = save_figure(fig, self.output_dir / filename)
        result.add_file(path)

    def _plot_metric_umap(
        self,
        adata: ad.AnnData,
        metric_map: pd.DataFrame,
        metric_name: str,
        side_label: str,
        organelle_name: str,
        output_dir: Path,
        live_short: str,
        notes: str = "",
    ) -> Optional[Path]:
        """
        Compute UMAP on the adata and visualize mAP metric scores.

        Adapted from alex_map_og.py metric_umap. Creates a 2-panel figure:
          - Top: UMAP colored by mAP (significant points only, others grey)
          - Bottom: UMAP colored by -log10(p-value)

        Parameters
        ----------
        adata : AnnData
            Guide- or gene-level AnnData (features in .X, 'perturbation' in .obs).
        metric_map : DataFrame
            Output from copairs mAP functions with columns:
            'perturbation', 'mean_average_precision', 'corrected_p_value',
            'below_corrected_p'.
        metric_name : str
            Short name like "activity", "distinctiveness", "corum", "chad".
        side_label : str
            "CP" or "Live" for title labeling.
        organelle_name : str
            Organelle name for title.
        output_dir : Path
            Directory to save figure.
        live_short : str
            Short live experiment name for filename.
        notes : str
            Optional annotation for subtitle.

        Returns
        -------
        Path or None
        """
        import scanpy as sc

        if adata.n_obs < 10:
            logger.warning(f"  Too few observations ({adata.n_obs}) for UMAP, skipping")
            return None

        # Compute UMAP (copy to avoid modifying the original)
        adata_umap = adata.copy()
        try:
            sc.pp.neighbors(adata_umap, n_neighbors=min(15, adata_umap.n_obs - 1), use_rep="X")
            sc.tl.umap(adata_umap)
        except Exception as e:
            logger.warning(f"  UMAP computation failed: {e}")
            return None

        umap_coords = adata_umap.obsm["X_umap"]

        # Add -log10(p-value) to metric_map if missing
        if "-log10(p-value)" not in metric_map.columns:
            metric_map = metric_map.copy()
            metric_map["-log10(p-value)"] = -metric_map["corrected_p_value"].apply(np.log10)

        # Map metric values to adata.obs
        metric_dict = metric_map.set_index("perturbation")[
            ["mean_average_precision", "-log10(p-value)", "below_corrected_p"]
        ].to_dict("index")

        obs = adata_umap.obs
        obs["mAP"] = obs["perturbation"].map(
            lambda x: metric_dict.get(x, {}).get("mean_average_precision", np.nan)
        )
        obs["log10p"] = obs["perturbation"].map(
            lambda x: metric_dict.get(x, {}).get("-log10(p-value)", np.nan)
        )
        obs["significant"] = obs["perturbation"].map(
            lambda x: metric_dict.get(x, {}).get("below_corrected_p", False)
        )
        obs["is_NTC"] = obs["perturbation"] == "NTC"

        ntc_mask = obs["is_NTC"].values
        significant_mask = (obs["significant"] == True).values & ~ntc_mask
        nonsig_mask = ~significant_mask & ~ntc_mask

        metric_title = metric_name.replace("_", " ").title()
        subtitle = f"{notes}" if notes else f"{live_short}"
        n_ntc = ntc_mask.sum()

        fig, axes = plt.subplots(1, 2, figsize=(18, 7))

        for panel_idx, (ax, color_col, cmap_name, cbar_label, panel_title_suffix) in enumerate([
            (axes[0], "mAP", "viridis", "Mean Average Precision", "mAP"),
            (axes[1], "log10p", "plasma", "-log10(p-value)", "-log10(p)"),
        ]):
            # Layer 1: non-significant (grey, behind)
            if nonsig_mask.any():
                ax.scatter(
                    umap_coords[nonsig_mask, 0], umap_coords[nonsig_mask, 1],
                    c="lightgrey", s=20, alpha=0.5, label="Not significant",
                )
            # Layer 2: significant (colored)
            if significant_mask.any():
                sc = ax.scatter(
                    umap_coords[significant_mask, 0], umap_coords[significant_mask, 1],
                    c=obs.loc[significant_mask, color_col], s=30, alpha=0.8,
                    cmap=cmap_name, edgecolors="black", linewidths=0.3,
                    label="Significant",
                )
                plt.colorbar(sc, ax=ax, label=cbar_label, shrink=0.8)
            # Layer 3: NTC (red diamonds, on top)
            if ntc_mask.any():
                ax.scatter(
                    umap_coords[ntc_mask, 0], umap_coords[ntc_mask, 1],
                    c="#E03030", s=50, alpha=0.9, marker="D",
                    edgecolors="black", linewidths=0.5,
                    label=f"NTC ({n_ntc})", zorder=5,
                )
            ax.set_xlabel("UMAP 1", fontsize=11)
            ax.set_ylabel("UMAP 2", fontsize=11)
            ax.set_title(f"{side_label} {organelle_name} — {metric_title}: {panel_title_suffix}",
                         fontsize=12, fontweight="bold")
            ax.legend(fontsize=9, loc="best")

        n_sig = significant_mask.sum()
        n_total = len(obs) - obs["is_NTC"].sum()
        fig.suptitle(
            f"{metric_title} UMAP — {subtitle}\n"
            f"{n_sig}/{n_total} significant perturbations ({100*n_sig/max(n_total,1):.1f}%)",
            fontsize=13, fontweight="bold", y=1.02,
        )
        plt.tight_layout()

        fname = f"{side_label.lower()}_{live_short}_{metric_name}_umap.png"
        path = save_figure(fig, output_dir / fname)
        return path


def _print_summary_table(summary_df: pd.DataFrame, norm_label: str, use_logger: bool = True) -> None:
    """
    Print a prettytable summary of CP Challenge results.

    Shows activity AND distinctiveness metrics side-by-side with a winner column.
    Works for both local (logger.info) and SLURM callback (print) output.
    """
    from prettytable import PrettyTable

    log = logger.info if use_logger else print

    table = PrettyTable()
    table.field_names = [
        "Organelle", "vs Exp", "Notes",
        "CP Act%", "Live Act%", "Act W",
        "CP AUC", "Live AUC", "AUC W",
        "CP SwAUC", "Live SwAUC", "Sw W",
        "CP Dist%", "Live Dist%", "Dist W",
    ]
    table.align = "l"
    for col in ["CP Act%", "Live Act%", "CP AUC", "Live AUC",
                "CP SwAUC", "Live SwAUC", "CP Dist%", "Live Dist%"]:
        table.align[col] = "r"

    def _fmt_pair(row, cp_key, live_key, fmt_str="{:.1%}", na="N/A"):
        cp_v = row.get(cp_key, float("nan"))
        live_v = row.get(live_key, float("nan"))
        if pd.notna(cp_v) and pd.notna(live_v):
            w = "CP" if cp_v > live_v else "Live"
            return fmt_str.format(cp_v), fmt_str.format(live_v), w
        return na, na, "N/A"

    for _, row in summary_df.iterrows():
        live_short = row.get("live_experiment", "?")
        if "_" in str(live_short):
            live_short = live_short.split("_")[0]

        cp_act_s, live_act_s, act_w = _fmt_pair(row, "cp_active_ratio", "live_active_ratio")
        cp_auc_s, live_auc_s, auc_w = _fmt_pair(row, "cp_auc_activity", "live_auc_activity", "{:.3f}")
        cp_sw_s, live_sw_s, sw_w = _fmt_pair(row, "cp_sweep_auc_activity", "live_sweep_auc_activity", "{:.3f}")
        cp_dist_s, live_dist_s, dist_w = _fmt_pair(row, "cp_distinctive_ratio", "live_distinctive_ratio")

        table.add_row([
            row.get("organelle_type", "?"),
            live_short,
            row.get("notes", "")[:20],
            cp_act_s, live_act_s, act_w,
            cp_auc_s, live_auc_s, auc_w,
            cp_sw_s, live_sw_s, sw_w,
            cp_dist_s, live_dist_s, dist_w,
        ])

    # Count wins
    n = len(summary_df)
    def _count_wins(cp_key, live_key):
        return sum(
            1 for _, r in summary_df.iterrows()
            if pd.notna(r.get(cp_key)) and r.get(cp_key, 0) > r.get(live_key, 0)
        )

    cp_act_wins = _count_wins("cp_active_ratio", "live_active_ratio")
    cp_auc_wins = _count_wins("cp_auc_activity", "live_auc_activity")
    cp_sw_wins = _count_wins("cp_sweep_auc_activity", "live_sweep_auc_activity")
    cp_dist_wins = _count_wins("cp_distinctive_ratio", "live_distinctive_ratio")

    log(f"\n  CP CHALLENGE SUMMARY — {norm_label.upper()}")
    log(f"  {n} comparisons | Act: CP {cp_act_wins}/{n} | AUC: CP {cp_auc_wins}/{n} | SwAUC: CP {cp_sw_wins}/{n} | Dist: CP {cp_dist_wins}/{n}")
    for line in table.get_string().split("\n"):
        log(f"  {line}")


# ---------------------------------------------------------------------------
# Top-level function for SLURM job submission (must be pickle-able)
# ---------------------------------------------------------------------------

def run_single_comparison_job(
    cp_experiment: str,
    live_experiment: str,
    organelle_name: str,
    cp_pattern: str,
    live_pattern: str,
    notes: str,
    output_dir: str,
    norm_method: str = "global",
    is_control: bool = False,
) -> str:
    """
    Run a single CP vs live-cell comparison as a standalone SLURM job.

    This is a top-level function (not a method) so submitit can pickle it.
    Each SLURM job loads data independently, runs the comparison, and saves
    per-comparison CSVs + plots to output_dir.

    Returns a status string.
    """
    import traceback
    from types import SimpleNamespace

    _check_copairs_installed()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    logging.getLogger("copairs").setLevel(logging.WARNING)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    subdir = "per_organelle_control" if is_control else "per_organelle"
    org_output_dir = output_dir / subdir / organelle_name
    org_output_dir.mkdir(parents=True, exist_ok=True)

    try:
        from ops_utils.data.filesystem import resolve_experiment_name
        from ops_utils.data.experiment import OpsDataset

        cp_resolved = resolve_experiment_name(cp_experiment, allow_interactive=False, autoselect=True)
        dataset = OpsDataset(cp_resolved)

        # Create minimal data context shim
        data_shim = SimpleNamespace(
            experiment=cp_resolved,
            graph_output_path=output_dir,
        )
        config_shim = SimpleNamespace(experiment=cp_resolved)

        stage = CPChallengeStage(
            data_context=data_shim,
            config=config_shim,
            level="guide",
        )
        # Override output_dir to avoid BaseStage double-nesting
        stage._output_dir = output_dir

        # Load CP cell-level data
        cp_cell_adata = stage._load_cell_adata(cp_resolved)
        if cp_cell_adata is None:
            return f"[ERROR] Could not load CP cell data for {cp_resolved}"

        # Discover CP organelles and match
        cp_organelles = stage._discover_organelles(cp_cell_adata)
        cp_org_name, cp_feat_cols = stage._match_organelle(cp_organelles, cp_pattern)
        if cp_org_name is None:
            return f"[ERROR] No CP organelle matching '{cp_pattern}' in {cp_resolved}"

        logger.info(f"CP organelle: {cp_org_name} ({len(cp_feat_cols)} features)")

        # Run comparison
        comp_result = stage._run_single_comparison(
            cp_cell_adata=cp_cell_adata,
            cp_org_name=cp_org_name,
            cp_feat_cols=cp_feat_cols,
            live_experiment=live_experiment,
            live_pattern=live_pattern,
            organelle_name=organelle_name,
            org_output_dir=org_output_dir,
            notes=notes,
            norm_method=norm_method,
        )

        comparison = comp_result.get("comparison")
        if comparison is None:
            errors = comp_result.get("errors", [])
            return f"[ERROR] {organelle_name} vs {live_experiment}: {'; '.join(errors)}"

        # Add notes to comparison
        comparison["notes"] = notes

        # Save per-comparison summary CSV
        import pandas as pd
        summary_df = pd.DataFrame([comparison])
        live_short = live_experiment.split("_")[0] if "_" in live_experiment else live_experiment
        summary_path = org_output_dir / f"comparison_summary_{live_short}.csv"
        summary_df.to_csv(summary_path, index=False)

        winner = comparison.get("winner_activity", "N/A")
        cp_ratio = comparison.get("cp_active_ratio", 0)
        live_ratio = comparison.get("live_active_ratio", 0)
        return (
            f"[OK] {organelle_name} vs {live_experiment} ({notes}): "
            f"CP={cp_ratio:.1%} Live={live_ratio:.1%} -> {winner}"
        )

    except Exception as e:
        return f"[ERROR] {organelle_name} vs {live_experiment}: {e}\n{traceback.format_exc()}"


def _aggregate_slurm_results(submitted_jobs: list, experiment: str) -> None:
    """
    Post-completion callback: aggregate per-comparison CSVs into overall summary.

    Called by submit_parallel_jobs after all SLURM jobs finish.
    Aggregates separately for each normalization subdirectory (global_norm, ntc_norm).
    """
    import pandas as pd

    logger.info("Aggregating SLURM comparison results...")

    # Collect unique output dirs (one per norm method)
    output_dirs = set()
    for job_info in submitted_jobs:
        od = job_info.get("metadata", {}).get("output_dir")
        if od:
            output_dirs.add(Path(od))

    if not output_dirs:
        print("Warning: Could not determine output directories for aggregation")
        return

    for output_dir in sorted(output_dirs):
        norm_label = output_dir.name  # e.g. "global_norm" or "ntc_norm"
        print(f"\n--- Aggregating {norm_label} ---")

        # Find all per-comparison summary CSVs
        summary_csvs = sorted(output_dir.glob("per_organelle/*/comparison_summary_*.csv"))
        if not summary_csvs:
            print(f"  No per-comparison summary CSVs found in {output_dir}")
            continue

        # Combine into overall summary
        all_dfs = []
        for csv_path in summary_csvs:
            try:
                df = pd.read_csv(csv_path)
                all_dfs.append(df)
            except Exception as e:
                print(f"  Warning: Could not read {csv_path}: {e}")

        if not all_dfs:
            continue

        summary_df = pd.concat(all_dfs, ignore_index=True)
        summary_path = output_dir / "cp_challenge_summary.csv"
        summary_df.to_csv(summary_path, index=False)
        print(f"  Aggregated summary ({len(summary_df)} comparisons): {summary_path}")

        # Log summary with prettytable
        _print_summary_table(summary_df, norm_label, use_logger=False)

        # Generate aggregation plots (bar chart, heatmap, radar, etc.)
        try:
            from types import SimpleNamespace

            data_shim = SimpleNamespace(
                experiment=experiment,
                graph_output_path=output_dir,
            )
            config_shim = SimpleNamespace(experiment=experiment)
            stage = CPChallengeStage(
                data_context=data_shim,
                config=config_shim,
                level="guide",
            )
            stage._output_dir = output_dir

            # StageResult shim — just needs add_file / add_metric
            class _ResultShim:
                def __init__(self):
                    self.output_files = []
                    self.data = {}
                def add_file(self, p):
                    self.output_files.append(p)
                def add_metric(self, k, v):
                    self.data[k] = v

            result_shim = _ResultShim()
            stage._plot_overall_comparison(summary_df, result_shim)
            print(f"  Generated {len(result_shim.output_files)} aggregation plots in {output_dir}")
        except Exception as e:
            print(f"  Warning: Aggregation plot generation failed: {e}")

        # --- Aggregate CONTROL results ---
        control_csvs = sorted(output_dir.glob("per_organelle_control/*/comparison_summary_*.csv"))
        if control_csvs:
            ctrl_dfs = []
            for csv_path in control_csvs:
                try:
                    df = pd.read_csv(csv_path)
                    ctrl_dfs.append(df)
                except Exception as e:
                    print(f"  Warning: Could not read control CSV {csv_path}: {e}")

            if ctrl_dfs:
                control_summary_df = pd.concat(ctrl_dfs, ignore_index=True)
                # Deduplicate — old SLURM runs may have produced duplicate control rows
                n_before = len(control_summary_df)
                control_summary_df = control_summary_df.drop_duplicates(
                    subset=["live_experiment"], keep="first",
                ).reset_index(drop=True)
                if len(control_summary_df) < n_before:
                    print(f"  Deduplicated controls: {n_before} -> {len(control_summary_df)}")
                control_summary_path = output_dir / "cp_challenge_control_summary.csv"
                control_summary_df.to_csv(control_summary_path, index=False)
                print(f"  Aggregated CONTROL summary ({len(control_summary_df)} comparisons): {control_summary_path}")

                _print_summary_table(control_summary_df, f"{norm_label} CONTROL", use_logger=False)

                try:
                    ctrl_result_shim = _ResultShim()
                    stage._plot_overall_comparison(
                        control_summary_df, ctrl_result_shim,
                        subdir_prefix="control_", title_extra=" (Control)",
                    )
                    print(f"  Generated {len(ctrl_result_shim.output_files)} control aggregation plots")
                except Exception as e:
                    print(f"  Warning: Control aggregation plot generation failed: {e}")

                # --- Generate BATCH-CORRECTED plots ---
                try:
                    normalized_df = stage._compute_normalized_summary(summary_df, control_summary_df)
                    if not normalized_df.empty:
                        normalized_df.to_csv(output_dir / "cp_challenge_normalized_summary.csv", index=False)
                        _print_summary_table(normalized_df, f"{norm_label} BATCH-CORRECTED", use_logger=False)

                        norm_result_shim = _ResultShim()
                        stage._plot_overall_comparison(
                            normalized_df, norm_result_shim,
                            subdir_prefix="normalized_", title_extra=" (Batch-Corrected)",
                        )
                        print(f"  Generated {len(norm_result_shim.output_files)} normalized aggregation plots")
                except Exception as e:
                    print(f"  Warning: Normalized aggregation plot generation failed: {e}")


# ---------------------------------------------------------------------------
# Standalone CLI entry point (no orchestrator needed)
# ---------------------------------------------------------------------------

def _check_copairs_installed():
    """Verify copairs is importable; raise immediately if missing."""
    try:
        import copairs  # noqa: F401
    except ImportError:
        raise ImportError(
            "The 'copairs' package is required but not installed in the current environment. "
            "Install it with:  pip install copairs"
        )


def main():
    """Run CP challenge as a standalone script, bypassing the orchestrator."""
    _check_copairs_installed()

    import argparse
    from types import SimpleNamespace

    parser = argparse.ArgumentParser(
        description="Cell Painting Challenge: CP vs Live-Cell mAP comparison"
    )
    parser.add_argument("-e", "--experiment", default="ops0094_20251217",
                        help="CP experiment name or shorthand (default: ops0094_20251217)")
    parser.add_argument("-o", "--output-dir", default=None,
                        help="Output directory (default: <fast_ops>/<experiment>/.../graphs)")

    # SLURM options
    slurm_group = parser.add_argument_group("SLURM options")
    slurm_group.add_argument("--slurm", action="store_true",
                             help="Submit each comparison as a separate SLURM job")
    slurm_group.add_argument("--no-wait", action="store_true",
                             help="Don't wait for SLURM jobs to complete")
    slurm_group.add_argument("--yes", "-y", action="store_true",
                             help="Skip confirmation prompt")
    slurm_group.add_argument("--quiet", "-q", action="store_true",
                             help="Reduce output verbosity")
    slurm_group.add_argument("--slurm-memory", type=str, default="128GB",
                             help="Memory per SLURM job (default: 128GB)")
    slurm_group.add_argument("--slurm-time", type=int, default=60,
                             help="Time limit per SLURM job in minutes (default: 60)")
    slurm_group.add_argument("--slurm-cpus", type=int, default=16,
                             help="CPUs per SLURM job (default: 16)")

    parser.add_argument("--aggregate", action="store_true",
                        help="Only run aggregation (summary table + plots) on existing per-comparison CSVs")

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    from ops_utils.data.filesystem import resolve_experiment_name
    from ops_utils.data.experiment import OpsDataset

    experiment = resolve_experiment_name(args.experiment, allow_interactive=False, autoselect=True)
    dataset = OpsDataset(experiment)

    # Determine output path
    if args.output_dir:
        graph_output = Path(args.output_dir)
    else:
        graph_output = dataset.results_fast / "feature_extraction" / "graphs"
    graph_output.mkdir(parents=True, exist_ok=True)

    # --- Aggregate-only mode: re-run summary tables + plots from existing CSVs ---
    if args.aggregate:
        cp_challenge_dir = graph_output / "2_guide_level" / "12_cp_challenge"
        _run_aggregate_only(experiment, cp_challenge_dir)
        return

    # --- SLURM mode: submit each comparison as a separate job ---
    if args.slurm:
        _run_slurm_mode(args, experiment, graph_output)
        return

    # --- Local mode: run all comparisons sequentially ---
    # Create minimal data context shim (only needs experiment + graph_output_path)
    data_shim = SimpleNamespace(
        experiment=experiment,
        graph_output_path=graph_output,
    )
    config_shim = SimpleNamespace(experiment=experiment)

    # Instantiate and run
    stage = CPChallengeStage(
        data_context=data_shim,
        config=config_shim,
        level="guide",
    )
    result = stage.run()

    # Report
    print(f"\nOutput: {graph_output}")
    print(f"Files: {len(result.output_files)}")
    if result.errors:
        print(f"Errors: {len(result.errors)}")
        for err in result.errors:
            print(f"  - {err}")


def _run_slurm_mode(args, experiment: str, graph_output: Path) -> None:
    """Submit each comparison as a separate SLURM job."""
    from ops_utils.hpc.slurm_batch_utils import submit_parallel_jobs

    # Load config
    if not DEFAULT_CONFIG_PATH.exists():
        print(f"Config not found: {DEFAULT_CONFIG_PATH}")
        return

    with open(DEFAULT_CONFIG_PATH) as f:
        config = yaml.safe_load(f)

    cp_experiment = config["cp_experiment"]
    comparisons = config.get("comparisons", {})

    # Build job list: one job per (comparison x norm_method)
    norm_methods = ["global", "ntc"]
    jobs_to_submit = []
    control_seen = set()  # deduplicate controls by (live_experiment, norm_method)
    for norm_method in norm_methods:
        norm_label = f"{norm_method}_norm"
        for organelle_name, organelle_config in comparisons.items():
            cp_pattern = organelle_config["cp_organelle_pattern"]

            for live_entry in organelle_config.get("live_cell", []):
                live_exp = live_entry["experiment"]
                live_pattern = live_entry["organelle_pattern"]
                notes = live_entry.get("notes", "")
                live_short = live_exp.split("_")[0] if "_" in live_exp else live_exp
                out_dir = str(graph_output / f"2_guide_level/12_cp_challenge/{norm_label}")

                # Main organelle comparison job
                jobs_to_submit.append({
                    "name": f"{norm_label}_{organelle_name}_vs_{live_short}",
                    "func": run_single_comparison_job,
                    "kwargs": {
                        "cp_experiment": cp_experiment,
                        "live_experiment": live_exp,
                        "organelle_name": organelle_name,
                        "cp_pattern": cp_pattern,
                        "live_pattern": live_pattern,
                        "notes": notes,
                        "output_dir": out_dir,
                        "norm_method": norm_method,
                    },
                    "metadata": {
                        "organelle": organelle_name,
                        "live_experiment": live_exp,
                        "notes": notes,
                        "norm_method": norm_method,
                        "output_dir": out_dir,
                        "is_control": False,
                    },
                })

                # Control comparison job (phase2d_tubular on both sides)
                # Deduplicate: control only depends on live experiment, not organelle
                ctrl_key = (live_exp, norm_method)
                if ctrl_key not in control_seen:
                    control_seen.add(ctrl_key)
                    jobs_to_submit.append({
                        "name": f"CTRL_{norm_label}_{live_short}",
                        "func": run_single_comparison_job,
                        "kwargs": {
                            "cp_experiment": cp_experiment,
                            "live_experiment": live_exp,
                            "organelle_name": live_short,
                            "cp_pattern": CONTROL_PATTERN,
                            "live_pattern": CONTROL_PATTERN,
                            "notes": f"CONTROL: {CONTROL_PATTERN}",
                            "output_dir": out_dir,
                            "norm_method": norm_method,
                            "is_control": True,
                        },
                        "metadata": {
                            "organelle": live_short,
                            "live_experiment": live_exp,
                            "notes": f"CONTROL: {CONTROL_PATTERN}",
                            "norm_method": norm_method,
                            "output_dir": out_dir,
                            "is_control": True,
                        },
                    })

    if not jobs_to_submit:
        print("No comparisons found in config!")
        return

    # SLURM parameters
    slurm_params = {
        "timeout_min": args.slurm_time,
        "mem": args.slurm_memory,
        "cpus_per_task": args.slurm_cpus,
        "slurm_partition": "cpu",
    }

    n_main = len([j for j in jobs_to_submit if not j["metadata"].get("is_control")])
    n_ctrl = len([j for j in jobs_to_submit if j["metadata"].get("is_control")])

    # Print plan
    print(f"\n{'='*60}")
    print(f"CP Challenge SLURM Submission")
    print(f"{'='*60}")
    print(f"CP experiment: {cp_experiment}")
    print(f"Jobs:          {n_main} comparisons + {n_ctrl} controls = {len(jobs_to_submit)} total")
    print(f"Output:        {graph_output}")
    print(f"\nSLURM Resources (per job):")
    print(f"  Timeout:   {slurm_params['timeout_min']} min")
    print(f"  Memory:    {slurm_params['mem']}")
    print(f"  CPUs:      {slurm_params['cpus_per_task']}")
    print(f"  Partition: {slurm_params['slurm_partition']}")
    print(f"\nJobs ({len(jobs_to_submit)} total):")
    for norm_method in norm_methods:
        print(f"\n  [{norm_method.upper()} normalization]")
        norm_jobs = [j for j in jobs_to_submit
                     if j["metadata"]["norm_method"] == norm_method
                     and not j["metadata"].get("is_control")]
        for i, job in enumerate(norm_jobs, 1):
            meta = job["metadata"]
            print(f"    {i:2d}. {meta['organelle']} vs {meta['live_experiment']} ({meta['notes']})")
        ctrl_jobs = [j for j in jobs_to_submit
                     if j["metadata"]["norm_method"] == norm_method
                     and j["metadata"].get("is_control")]
        if ctrl_jobs:
            print(f"\n  [{norm_method.upper()} CONTROL ({CONTROL_PATTERN})]")
            for i, job in enumerate(ctrl_jobs, 1):
                meta = job["metadata"]
                print(f"    {i:2d}. {meta['organelle']} vs {meta['live_experiment']}")
    print(f"{'='*60}")

    # Confirmation
    if not args.yes:
        try:
            response = input(f"\nSubmit {len(jobs_to_submit)} jobs to SLURM? [y/N]: ").strip().lower()
            if response not in ['y', 'yes']:
                print("\nCancelled. No jobs submitted.\n")
                return
        except (KeyboardInterrupt, EOFError):
            print("\n\nCancelled. No jobs submitted.\n")
            return
        print()
    else:
        print("\nProceeding with submission (--yes flag provided)...\n")

    # Submit
    result = submit_parallel_jobs(
        jobs_to_submit=jobs_to_submit,
        experiment=f"cp_challenge_{experiment}",
        slurm_params=slurm_params,
        log_dir=f"slurm_cp_challenge_logs/{experiment}",
        manifest_prefix="cp_challenge",
        dry_run=False,
        wait_for_completion=not args.no_wait,
        verbose=False,
        post_completion_callback=_aggregate_slurm_results,
        print_resource_summary=not args.quiet,
    )

    if result.get("success"):
        if result.get("all_completed"):
            print("\nAll comparisons completed successfully!")
        elif result.get("failed"):
            print(f"\n{len(result['failed'])} comparison(s) failed")
    else:
        print(f"\nJob submission failed: {result.get('error', 'unknown')}")


def _run_aggregate_only(experiment: str, cp_challenge_dir: Path) -> None:
    """
    Re-run aggregation (summary tables + plots) from existing per-comparison CSVs.

    Looks for global_norm/ and ntc_norm/ subdirectories inside cp_challenge_dir,
    collects comparison_summary_*.csv files, and generates the summary table + all
    aggregation plots.
    """
    from types import SimpleNamespace
    import pandas as pd

    norm_dirs = sorted(d for d in cp_challenge_dir.iterdir()
                       if d.is_dir() and d.name.endswith("_norm"))

    if not norm_dirs:
        # Fallback: maybe no norm subdirs, check cp_challenge_dir directly
        print(f"No *_norm subdirectories found in {cp_challenge_dir}")
        print(f"Contents: {[p.name for p in cp_challenge_dir.iterdir()]}")
        return

    print(f"\nAggregate-only mode: {cp_challenge_dir}")
    print(f"Found norm directories: {[d.name for d in norm_dirs]}\n")

    for norm_dir in norm_dirs:
        norm_label = norm_dir.name
        summary_csvs = sorted(norm_dir.glob("per_organelle/*/comparison_summary_*.csv"))
        if not summary_csvs:
            print(f"  [{norm_label}] No per-comparison CSVs found, skipping")
            continue

        all_dfs = []
        for csv_path in summary_csvs:
            try:
                df = pd.read_csv(csv_path)
                all_dfs.append(df)
            except Exception as e:
                print(f"  Warning: Could not read {csv_path}: {e}")

        if not all_dfs:
            continue

        summary_df = pd.concat(all_dfs, ignore_index=True)
        summary_path = norm_dir / "cp_challenge_summary.csv"
        summary_df.to_csv(summary_path, index=False)
        print(f"  [{norm_label}] Aggregated {len(summary_df)} comparisons -> {summary_path}")

        # Print prettytable
        _print_summary_table(summary_df, norm_label, use_logger=False)

        # Generate plots
        try:
            data_shim = SimpleNamespace(
                experiment=experiment,
                graph_output_path=norm_dir,
            )
            config_shim = SimpleNamespace(experiment=experiment)
            stage = CPChallengeStage(
                data_context=data_shim,
                config=config_shim,
                level="guide",
            )
            stage._output_dir = norm_dir

            class _ResultShim:
                def __init__(self):
                    self.output_files = []
                    self.data = {}
                def add_file(self, p):
                    self.output_files.append(p)
                def add_metric(self, k, v):
                    self.data[k] = v

            result_shim = _ResultShim()
            stage._plot_overall_comparison(summary_df, result_shim)
            print(f"  [{norm_label}] Generated {len(result_shim.output_files)} plots")
            for p in result_shim.output_files:
                print(f"    {p}")
        except Exception as e:
            print(f"  [{norm_label}] Plot generation failed: {e}")
            import traceback
            traceback.print_exc()

        # --- Aggregate CONTROL results ---
        control_csvs = sorted(norm_dir.glob("per_organelle_control/*/comparison_summary_*.csv"))
        if control_csvs:
            ctrl_dfs = []
            for csv_path in control_csvs:
                try:
                    df = pd.read_csv(csv_path)
                    ctrl_dfs.append(df)
                except Exception as e:
                    print(f"  Warning: Could not read control CSV {csv_path}: {e}")

            if ctrl_dfs:
                control_summary_df = pd.concat(ctrl_dfs, ignore_index=True)
                # Deduplicate — old SLURM runs may have produced duplicate control rows
                n_before = len(control_summary_df)
                control_summary_df = control_summary_df.drop_duplicates(
                    subset=["live_experiment"], keep="first",
                ).reset_index(drop=True)
                if len(control_summary_df) < n_before:
                    print(f"  [{norm_label}] Deduplicated controls: {n_before} -> {len(control_summary_df)}")
                control_summary_path = norm_dir / "cp_challenge_control_summary.csv"
                control_summary_df.to_csv(control_summary_path, index=False)
                print(f"  [{norm_label}] Aggregated CONTROL {len(control_summary_df)} comparisons -> {control_summary_path}")

                _print_summary_table(control_summary_df, f"{norm_label} CONTROL", use_logger=False)

                try:
                    ctrl_result_shim = _ResultShim()
                    stage._plot_overall_comparison(
                        control_summary_df, ctrl_result_shim,
                        subdir_prefix="control_", title_extra=" (Control)",
                    )
                    print(f"  [{norm_label}] Generated {len(ctrl_result_shim.output_files)} control plots")
                    for p in ctrl_result_shim.output_files:
                        print(f"    {p}")
                except Exception as e:
                    print(f"  [{norm_label}] Control plot generation failed: {e}")
                    import traceback
                    traceback.print_exc()

                # --- Generate BATCH-CORRECTED plots ---
                try:
                    normalized_df = stage._compute_normalized_summary(summary_df, control_summary_df)
                    if not normalized_df.empty:
                        normalized_df.to_csv(norm_dir / "cp_challenge_normalized_summary.csv", index=False)
                        _print_summary_table(normalized_df, f"{norm_label} BATCH-CORRECTED", use_logger=False)

                        norm_result_shim = _ResultShim()
                        stage._plot_overall_comparison(
                            normalized_df, norm_result_shim,
                            subdir_prefix="normalized_", title_extra=" (Batch-Corrected)",
                        )
                        print(f"  [{norm_label}] Generated {len(norm_result_shim.output_files)} normalized plots")
                        for p in norm_result_shim.output_files:
                            print(f"    {p}")
                except Exception as e:
                    print(f"  [{norm_label}] Normalized plot generation failed: {e}")
                    import traceback
                    traceback.print_exc()

    print(f"\nDone. Results in {cp_challenge_dir}")


if __name__ == "__main__":
    main()
