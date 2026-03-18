"""Reporter Radar Stage: Per-reporter mAP biological profiling via radar/spider plots.

Answers: **what biology does each reporter see?**

Two analysis levels:
  - Level 1 (per reporter): Each reporter evaluated on its own features
  - Level 2 (per reporter-type): Reporters sharing an organelle type combined

Gene categorization (--sources):
  - chad: CHAD v5 clusters only (8 categories, ~19% coverage)
  - chad_boosted: CHAD + keyword/regex/Harmonizome (8 categories, ~98% coverage)
  - reactome_toplevel: Reactome top-level pathways (29 categories, 78%, multi-mapped)
  Default: chad_boosted. Results saved under sources_{name}/ subdirs.
"""

import time
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import yaml
import logging

from ops_utils.analysis.map_scores import (
    phenotypic_activity_assesment,
    phenotypic_distinctivness,
    compute_auc_score,
)
from ops_utils.analysis.normalization import zscore_normalize
from ops_utils.analysis.gene_supercategories import (
    ALL_SOURCES,
    build_gene_supercategory_map,
    build_reactome_toplevel_map,
    assign_genes_to_categories,
    assign_genes_to_categories_multi,
    is_reactome_toplevel_mode,
    parse_sources,
    sources_label,
)
from ops_utils.data.feature_discovery import (
    discover_dino_experiments,
    resolve_channel_label,
    build_signal_groups,
    get_channel_maps_path,
    get_storage_roots,
    sanitize_signal_filename,
)

try:
    from .fe_graphs_stage_base import BaseStage, StageResult
    from ..plotting.fe_graphs_utils import save_figure
except ImportError:
    from organelle_profiler.fe_graphs.stages.fe_graphs_stage_base import BaseStage, StageResult
    from organelle_profiler.fe_graphs.plotting.fe_graphs_utils import save_figure

logger = logging.getLogger(__name__)
logging.getLogger("copairs").setLevel(logging.WARNING)

DEFAULT_CONFIG_PATH = Path(__file__).parents[4] / "configs" / "reporter_radar_config.yaml"
DEFAULT_SUPERCATEGORY_PATH = Path(__file__).parents[4] / "configs" / "gene_supercategory_mapping.yaml"


# ---------------------------------------------------------------------------
# Reporter-type grouping
# ---------------------------------------------------------------------------

def group_reporters_by_type(
    signal_map: Dict[str, List[Tuple[str, str]]],
) -> Dict[str, List[str]]:
    """Group reporter labels by organelle type (part before comma).

    Examples:
      "lysosome, LAMP1" → type "lysosome"
      "mitochondria, TOMM20" → type "mitochondria"
      "Phase" → type "Phase"

    Returns dict: type_name → [reporter_label, ...]
    """
    type_to_reporters: Dict[str, List[str]] = defaultdict(list)
    for label in signal_map:
        if "," in label:
            rtype = label.split(",", 1)[0].strip()
        else:
            rtype = label
        type_to_reporters[rtype].append(label)
    return dict(type_to_reporters)


# ---------------------------------------------------------------------------
# Stage class
# ---------------------------------------------------------------------------

class ReporterRadarStage(BaseStage):
    """Per-reporter mAP biological profiling with radar/spider plots."""

    STAGE_NUMBER = 14
    STAGE_NAME = "reporter_radar"

    def __init__(
        self,
        data_context,
        config,
        level: str = "guide",
        norm_method: str = "ntc",
        config_path: Optional[Path] = None,
        supercategory_path: Optional[Path] = None,
        radar_metric: str = "fraction_active",
        analysis_level: str = "both",
        reporter_filter: Optional[List[str]] = None,
        sources: Optional[frozenset] = None,
        **kwargs,
    ):
        super().__init__(data_context, config, level, **kwargs)
        self.norm_method = norm_method
        self.config_path = config_path or DEFAULT_CONFIG_PATH
        self.supercategory_path = supercategory_path or DEFAULT_SUPERCATEGORY_PATH
        self.radar_metric = radar_metric
        self.analysis_level = analysis_level  # "individual", "type", or "both"
        self.reporter_filter = reporter_filter
        self.sources = sources or frozenset({"chad_boosted"})

        # Load configs
        self.stage_config = _load_yaml(self.config_path)
        self.supercategory_config = _load_yaml(self.supercategory_path)

        self._storage_roots = get_storage_roots(self.stage_config)
        self._feature_dir = self.stage_config.get("feature_dir", "dino_features")
        self._feature_type = self.stage_config.get("feature_type", "dinov3")
        self._join = self.stage_config.get("join", "inner")
        self._null_size = self.stage_config.get("null_size", 1_000_000)
        self._min_perturbations = self.stage_config.get("min_perturbations", 50)
        self._min_genes_per_category = self.stage_config.get("min_genes_per_category", 3)

    # ------------------------------------------------------------------
    # Main run
    # ------------------------------------------------------------------

    def run(self) -> StageResult:
        result = StageResult()
        t0 = time.time()
        self.log_start("Reporter Radar: Per-Reporter mAP Biological Profiling")

        # Step 1: Discover experiments and build signal map
        logger.info("Step 1: Discovering experiments...")
        pairs = discover_dino_experiments(self._storage_roots, self._feature_dir)
        if len(pairs) < 2:
            result.add_error(f"Need at least 2 experiment/channel pairs, found {len(pairs)}")
            return result

        from ops_utils.data.feature_metadata import FeatureMetadata
        fm = FeatureMetadata(metadata_path=get_channel_maps_path())
        signal_map = build_signal_groups(pairs, fm)

        # Apply reporter filter if specified
        if self.reporter_filter:
            signal_map = {k: v for k, v in signal_map.items() if k in self.reporter_filter}
            if not signal_map:
                result.add_error(f"No reporters matched filter: {self.reporter_filter}")
                return result

        logger.info(f"  {len(signal_map)} reporters to process")

        # Step 2: Build gene super-category mapping
        src_lbl = sources_label(self.sources)
        self._multi_mapping = is_reactome_toplevel_mode(self.sources)
        logger.info(
            f"Step 2: Building gene super-category mapping "
            f"(sources={src_lbl}, mode={'multi-mapping' if self._multi_mapping else 'single-mapping'})..."
        )

        if self._multi_mapping:
            self._gene_to_cats = build_reactome_toplevel_map()
            gene_to_cat = {}
        else:
            boosted = "chad_boosted" in self.sources
            gene_to_cat = build_gene_supercategory_map(
                self.supercategory_config,
                boosted=boosted,
            )
            self._gene_to_cats = {}
        self._boosted = "chad_boosted" in self.sources

        # Nest outputs under sources subdir
        method_dir = self.output_dir / f"sources_{src_lbl}"
        method_dir.mkdir(parents=True, exist_ok=True)

        # Save gene assignment CSV
        cat_csv_path = method_dir / "gene_supercategory_assignment.csv"
        if self._multi_mapping:
            rows = []
            for g, cats in sorted(self._gene_to_cats.items()):
                for cat in cats:
                    rows.append({"gene": g, "category": cat})
            pd.DataFrame(rows).to_csv(cat_csv_path, index=False)
        else:
            pd.DataFrame([
                {"gene": g, "category": c} for g, c in sorted(gene_to_cat.items())
            ]).to_csv(cat_csv_path, index=False)
        result.add_file(cat_csv_path)

        # Step 3: Per-reporter mAP (Level 1)
        reporter_results: Dict[str, Dict[str, Any]] = {}
        if self.analysis_level in ("individual", "both"):
            logger.info("Step 3: Per-reporter mAP scoring...")
            reporter_results = self._run_all_reporters(signal_map, gene_to_cat, result)

        # Step 4: Per-reporter-type mAP (Level 2)
        type_results: Dict[str, Dict[str, Any]] = {}
        if self.analysis_level in ("type", "both"):
            logger.info("Step 4: Per-reporter-type mAP scoring...")
            type_groups = group_reporters_by_type(signal_map)
            type_results = self._run_all_reporter_types(
                type_groups, signal_map, gene_to_cat, result
            )

        # Step 5: Build radar matrices and generate plots
        logger.info("Step 5: Generating radar plots and summaries...")
        if reporter_results:
            self._generate_level_outputs(
                reporter_results, gene_to_cat, "per_reporter", result,
                base_dir=method_dir,
            )
        if type_results:
            self._generate_level_outputs(
                type_results, gene_to_cat, "per_reporter_type", result,
                base_dir=method_dir,
            )

        elapsed = time.time() - t0
        logger.info(f"\nReporter radar complete in {elapsed:.0f}s")
        logger.info(f"Output: {self.output_dir}")
        self.log_complete(result)
        return result

    # ------------------------------------------------------------------
    # Per-reporter scoring (Level 1)
    # ------------------------------------------------------------------

    def _run_all_reporters(
        self,
        signal_map: Dict[str, List[Tuple[str, str]]],
        gene_to_cat: Dict[str, str],
        result: StageResult,
    ) -> Dict[str, Dict[str, Any]]:
        """Run mAP for each individual reporter."""
        all_results: Dict[str, Dict[str, Any]] = {}
        n_total = len(signal_map)

        for i, (label, pairs) in enumerate(sorted(signal_map.items()), 1):
            logger.info(f"  [{i}/{n_total}] Reporter: {label} ({len(pairs)} pairs)")
            r = self._score_reporter(label, pairs)
            if r is not None:
                all_results[label] = r
            else:
                logger.warning(f"    Skipped {label}")

        return all_results

    def _run_all_reporter_types(
        self,
        type_groups: Dict[str, List[str]],
        signal_map: Dict[str, List[Tuple[str, str]]],
        gene_to_cat: Dict[str, str],
        result: StageResult,
    ) -> Dict[str, Dict[str, Any]]:
        """Run mAP for each reporter-type (combined reporters)."""
        all_results: Dict[str, Dict[str, Any]] = {}
        n_total = len(type_groups)

        for i, (type_name, reporter_labels) in enumerate(sorted(type_groups.items()), 1):
            # Collect all pairs for this type
            all_pairs = []
            for rl in reporter_labels:
                all_pairs.extend(signal_map.get(rl, []))

            members_str = ", ".join(reporter_labels)
            logger.info(
                f"  [{i}/{n_total}] Type: {type_name} "
                f"({len(reporter_labels)} reporters, {len(all_pairs)} pairs) "
                f"[{members_str}]"
            )
            r = self._score_reporter(type_name, all_pairs)
            if r is not None:
                all_results[type_name] = r
            else:
                logger.warning(f"    Skipped type {type_name}")

        return all_results

    def _score_reporter(
        self,
        label: str,
        pairs: List[Tuple[str, str]],
    ) -> Optional[Dict[str, Any]]:
        """Load a reporter's data, normalize, and run mAP activity + distinctiveness.

        Returns dict with activity_map, distinct_map, and scalar summaries,
        or None if data is insufficient.
        """
        try:
            from ops_model.features.anndata_utils import (
                concatenate_experiments_comprehensive,
                aggregate_to_level,
            )

            maps_path = get_channel_maps_path()

            # Build a single-reporter signal_map for the combiner
            sr_signal_map = {label: pairs}

            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message=".*names are not unique.*")
                adata_guide, adata_gene = concatenate_experiments_comprehensive(
                    experiments_channels=pairs,
                    feature_type=self._feature_type,
                    base_dir=str(self._storage_roots[0]),
                    feature_dir=self._feature_dir,
                    recompute_embeddings=False,
                    compute_pca=False,
                    compute_umap=False,
                    compute_phate=False,
                    normalize_on_pooling=False,
                    normalize_on_controls=False,
                    join=self._join,
                    verbose=False,
                    search_dirs=self._storage_roots,
                    use_preaggregated=False,
                    metadata_path=maps_path,
                    signal_map=sr_signal_map,
                )

            if adata_guide is None or adata_guide.n_obs < self._min_perturbations:
                logger.warning(
                    f"    {label}: only {adata_guide.n_obs if adata_guide else 0} "
                    f"perturbations (min {self._min_perturbations}), skipping"
                )
                return None

            # Normalize
            feature_cols = list(adata_guide.var_names)
            df = pd.DataFrame(adata_guide.X, columns=feature_cols)
            for col in adata_guide.obs.columns:
                df[col] = adata_guide.obs[col].values
            df = zscore_normalize(
                df, feature_cols, method=self.norm_method,
                perturbation_col="perturbation",
            )
            adata_guide.X = df[feature_cols].values.astype(np.float32)

            # Re-aggregate to gene level from normalized guides
            adata_gene = aggregate_to_level(
                adata_guide, "gene",
                preserve_batch_info=False,
                subsample_controls=False,
            )

            # Activity
            t1 = time.time()
            activity_map, active_ratio = phenotypic_activity_assesment(
                adata_guide, plot_results=False, null_size=self._null_size,
            )
            activity_auc = compute_auc_score(activity_map)
            logger.info(
                f"    Activity ({time.time()-t1:.1f}s): "
                f"{active_ratio:.2%} active, AUC={activity_auc:.4f}"
            )

            # Distinctiveness
            t2 = time.time()
            distinct_map, distinctive_ratio = phenotypic_distinctivness(
                adata_guide, activity_map, plot_results=False, null_size=self._null_size,
            )
            distinct_auc = compute_auc_score(distinct_map)
            logger.info(
                f"    Distinctiveness ({time.time()-t2:.1f}s): "
                f"{distinctive_ratio:.2%} distinctive, AUC={distinct_auc:.4f}"
            )

            return {
                "activity_map": activity_map,
                "distinct_map": distinct_map,
                "active_ratio": active_ratio,
                "distinctive_ratio": distinctive_ratio,
                "activity_auc": activity_auc,
                "distinct_auc": distinct_auc,
                "n_perturbations": adata_guide.n_obs,
                "n_features": adata_guide.n_vars,
            }

        except Exception as e:
            logger.error(f"    Failed for {label}: {e}")
            import traceback
            traceback.print_exc()
            return None

    # ------------------------------------------------------------------
    # Output generation (shared for both levels)
    # ------------------------------------------------------------------

    def _generate_level_outputs(
        self,
        all_results: Dict[str, Dict[str, Any]],
        gene_to_cat: Dict[str, str],
        subdir: str,
        result: StageResult,
        base_dir: Optional[Path] = None,
    ) -> None:
        """Generate CSVs, radar plots, and heatmaps for a given analysis level."""
        out_dir = (base_dir or self.output_dir) / subdir
        out_dir.mkdir(parents=True, exist_ok=True)

        # 1. Per-reporter mAP CSVs
        self._save_map_csvs(all_results, out_dir, result)

        # 2. Summary CSV
        summary_df = self._build_summary(all_results)
        summary_path = out_dir / "summary.csv"
        summary_df.to_csv(summary_path, index=False)
        result.add_file(summary_path)

        # 3. Radar matrices
        for metric_type in ("activity", "distinctiveness"):
            radar_df = self._compute_radar_matrix(
                all_results, gene_to_cat, metric_type
            )
            if radar_df is None or radar_df.empty:
                continue

            csv_path = out_dir / f"radar_matrix_{metric_type}.csv"
            radar_df.to_csv(csv_path)
            result.add_file(csv_path)

            # Radar plots
            self._plot_radar_grid(radar_df, metric_type, out_dir, result)
            self._plot_radar_overlay(radar_df, metric_type, out_dir, result)
            self._plot_heatmap(radar_df, metric_type, out_dir, result)

    def _save_map_csvs(
        self,
        all_results: Dict[str, Dict[str, Any]],
        out_dir: Path,
        result: StageResult,
    ) -> None:
        """Save stacked per-gene mAP results across all reporters."""
        for metric_type, key in [("activity", "activity_map"), ("distinctiveness", "distinct_map")]:
            frames = []
            for label, r in sorted(all_results.items()):
                df = r[key].copy()
                df["reporter"] = label
                frames.append(df)
            if frames:
                stacked = pd.concat(frames, ignore_index=True)
                path = out_dir / f"per_reporter_{metric_type}.csv"
                stacked.to_csv(path, index=False)
                result.add_file(path)

    def _build_summary(self, all_results: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
        """Build per-reporter summary table."""
        rows = []
        for label, r in sorted(all_results.items()):
            rows.append({
                "reporter": label,
                "n_perturbations": r["n_perturbations"],
                "n_features": r["n_features"],
                "n_active": int(r["activity_map"]["below_corrected_p"].sum()),
                "n_distinctive": int(r["distinct_map"]["below_corrected_p"].sum()),
                "active_ratio": r["active_ratio"],
                "distinctive_ratio": r["distinctive_ratio"],
                "activity_auc": r["activity_auc"],
                "distinct_auc": r["distinct_auc"],
            })
        return pd.DataFrame(rows)

    def _compute_radar_matrix(
        self,
        all_results: Dict[str, Dict[str, Any]],
        gene_to_cat: Dict[str, str],
        metric_type: str,  # "activity" or "distinctiveness"
    ) -> Optional[pd.DataFrame]:
        """Build reporters × categories radar matrix.

        Handles both single-mapping (curated sources) and multi-mapping
        (reactome_toplevel). In multi-mapping mode, a gene contributes to
        every category it belongs to.

        The value per cell depends on ``self.radar_metric``:
          - fraction_active: fraction of genes in category that are significant
          - mean_map: mean mAP of genes in that category
          - auc_score: significance-weighted AUC for genes in that category
        """
        map_key = "activity_map" if metric_type == "activity" else "distinct_map"
        rows: Dict[str, Dict[str, float]] = {}

        for label, r in all_results.items():
            map_df = r[map_key]
            genes = map_df["perturbation"].tolist()

            if self._multi_mapping:
                # Multi-mapping: gene → [cat1, cat2, ...]
                gene_cats_multi = assign_genes_to_categories_multi(
                    genes, self._gene_to_cats,
                )
                # Collect all categories
                all_cats = set()
                for cats in gene_cats_multi.values():
                    all_cats.update(cats)
                all_cats.discard("Other")

                row: Dict[str, float] = {}
                for cat in sorted(all_cats):
                    # Genes belonging to this category
                    cat_genes = {g for g, cats in gene_cats_multi.items() if cat in cats}
                    cat_df = map_df[map_df["perturbation"].isin(cat_genes)]
                    if len(cat_df) < self._min_genes_per_category:
                        row[cat] = 0.0
                        continue
                    row[cat] = self._score_category(cat_df)
                rows[label] = row
            else:
                # Single-mapping: gene → cat
                gene_cats = assign_genes_to_categories(
                    genes, gene_to_cat, self.supercategory_config,
                    boosted=self._boosted,
                )
                map_df = map_df.copy()
                map_df["category"] = map_df["perturbation"].map(gene_cats)

                row = {}
                for cat in sorted(set(gene_cats.values())):
                    if cat == "Other":
                        continue
                    cat_df = map_df[map_df["category"] == cat]
                    if len(cat_df) < self._min_genes_per_category:
                        row[cat] = 0.0
                        continue
                    row[cat] = self._score_category(cat_df)
                rows[label] = row

        if not rows:
            return None

        df = pd.DataFrame.from_dict(rows, orient="index")
        df = df.fillna(0.0)
        # Drop columns with all zeros
        df = df.loc[:, (df != 0).any(axis=0)]
        return df

    def _score_category(self, cat_df: pd.DataFrame) -> float:
        """Compute the radar metric for a category subset."""
        if self.radar_metric == "fraction_active":
            return float(cat_df["below_corrected_p"].mean())
        elif self.radar_metric == "mean_map":
            return float(cat_df["mean_average_precision"].mean())
        elif self.radar_metric == "auc_score":
            return float(compute_auc_score(cat_df))
        return float(cat_df["below_corrected_p"].mean())

    # ------------------------------------------------------------------
    # Radar / spider plots
    # ------------------------------------------------------------------

    def _plot_radar_single(
        self, ax, values: np.ndarray, categories: List[str],
        color: str, label: str, alpha: float = 0.2,
    ) -> None:
        """Draw one radar trace on a polar axes."""
        n = len(categories)
        angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
        angles += angles[:1]
        vals = values.tolist() + [values[0]]

        ax.plot(angles, vals, "o-", color=color, linewidth=2, markersize=5, label=label)
        ax.fill(angles, vals, alpha=alpha, color=color)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories, fontsize=8)

    def _plot_radar_grid(
        self,
        radar_df: pd.DataFrame,
        metric_type: str,
        out_dir: Path,
        result: StageResult,
    ) -> None:
        """Small multiples grid: one radar per reporter."""
        reporters = list(radar_df.index)
        categories = list(radar_df.columns)
        n = len(reporters)
        if n == 0 or len(categories) < 3:
            return

        ncols = min(4, n)
        nrows = (n + ncols - 1) // ncols
        fig_w = ncols * 4.5
        fig_h = nrows * 4.5

        fig, axes = plt.subplots(
            nrows, ncols, figsize=(fig_w, fig_h),
            subplot_kw={"projection": "polar"},
        )
        if n == 1:
            axes = np.array([axes])
        axes = axes.flatten()

        max_val = max(radar_df.values.max(), 0.01)
        cmap = plt.get_cmap("tab20")

        for i, reporter in enumerate(reporters):
            ax = axes[i]
            values = radar_df.loc[reporter].values
            color = cmap(i / max(n - 1, 1))
            self._plot_radar_single(ax, values, categories, color, reporter)
            ax.set_ylim(0, min(max_val * 1.15, 1.0))
            ax.set_title(_wrap_label(reporter, 25), fontsize=10, fontweight="bold", pad=20)

        # Hide unused axes
        for j in range(n, len(axes)):
            axes[j].set_visible(False)

        metric_label = "Activity" if metric_type == "activity" else "Distinctiveness"
        fig.suptitle(
            f"Reporter Radar — {metric_label} ({self.radar_metric})",
            fontsize=14, fontweight="bold", y=1.02,
        )
        plt.tight_layout()

        path = save_figure(fig, out_dir / f"radar_grid_{metric_type}.png")
        result.add_file(path)
        logger.info(f"  Saved: {path}")

    def _plot_radar_overlay(
        self,
        radar_df: pd.DataFrame,
        metric_type: str,
        out_dir: Path,
        result: StageResult,
    ) -> None:
        """All reporters overlaid on one radar."""
        categories = list(radar_df.columns)
        reporters = list(radar_df.index)
        if len(reporters) == 0 or len(categories) < 3:
            return

        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw={"projection": "polar"})
        cmap = plt.get_cmap("tab20")
        max_val = max(radar_df.values.max(), 0.01)

        for i, reporter in enumerate(reporters):
            values = radar_df.loc[reporter].values
            color = cmap(i / max(len(reporters) - 1, 1))
            self._plot_radar_single(ax, values, categories, color, reporter, alpha=0.08)

        ax.set_ylim(0, min(max_val * 1.15, 1.0))
        ax.legend(
            loc="upper right", bbox_to_anchor=(1.3, 1.1),
            fontsize=8, framealpha=0.9,
        )

        metric_label = "Activity" if metric_type == "activity" else "Distinctiveness"
        ax.set_title(
            f"Reporter Overlay — {metric_label} ({self.radar_metric})",
            fontsize=13, fontweight="bold", pad=30,
        )

        path = save_figure(fig, out_dir / f"radar_overlay_{metric_type}.png")
        result.add_file(path)
        logger.info(f"  Saved: {path}")

    def _plot_heatmap(
        self,
        radar_df: pd.DataFrame,
        metric_type: str,
        out_dir: Path,
        result: StageResult,
    ) -> None:
        """Heatmap view: reporters × categories."""
        if radar_df.empty or radar_df.shape[1] < 2:
            return

        fig_h = max(6, len(radar_df) * 0.4)
        fig, ax = plt.subplots(figsize=(12, fig_h))
        sns.heatmap(
            radar_df, cmap="YlOrRd", annot=True, fmt=".2f",
            ax=ax, cbar_kws={"label": self.radar_metric},
            linewidths=0.5,
        )
        metric_label = "Activity" if metric_type == "activity" else "Distinctiveness"
        ax.set_title(
            f"Reporter × Category — {metric_label} ({self.radar_metric})",
            fontsize=13, fontweight="bold",
        )
        ax.set_ylabel("Reporter")
        ax.set_xlabel("Biology Category")
        plt.xticks(rotation=30, ha="right")
        plt.yticks(rotation=0)
        plt.tight_layout()

        path = save_figure(fig, out_dir / f"heatmap_{metric_type}.png")
        result.add_file(path)
        logger.info(f"  Saved: {path}")

    # ------------------------------------------------------------------
    # Dry-run
    # ------------------------------------------------------------------

    def dry_run(self) -> None:
        """Discover reporters and print summary without loading data."""
        from ops_utils.data.feature_metadata import FeatureMetadata

        print("\n" + "=" * 80)
        print("  REPORTER RADAR — DRY RUN")
        print("=" * 80)

        pairs = discover_dino_experiments(self._storage_roots, self._feature_dir)
        if not pairs:
            print("\n  No experiment/channel pairs found!")
            return

        fm = FeatureMetadata(metadata_path=get_channel_maps_path())
        signal_map = build_signal_groups(pairs, fm)

        # Reporter-type groups
        type_groups = group_reporters_by_type(signal_map)

        print(f"\n  Sources:         {sources_label(self.sources)}")
        print(f"  Storage roots:   {[str(r) for r in self._storage_roots]}")
        print(f"  Total pairs:     {len(pairs)}")
        print(f"  Reporters:       {len(signal_map)}")
        print(f"  Reporter types:  {len(type_groups)}")

        print(f"\n{'─' * 80}")
        print(f"  {'REPORTER':<45} {'PAIRS':>6}  {'EXPERIMENTS':>11}")
        print(f"{'─' * 80}")
        for label in sorted(signal_map.keys()):
            rp = signal_map[label]
            n_exp = len(set(e for e, _ in rp))
            print(f"  {label:<45} {len(rp):>6}  {n_exp:>11}")

        print(f"\n{'─' * 80}")
        print(f"  {'REPORTER TYPE':<25} {'REPORTERS':>10}  {'TOTAL PAIRS':>12}  MEMBERS")
        print(f"{'─' * 80}")
        for tname in sorted(type_groups.keys()):
            members = type_groups[tname]
            total_pairs = sum(len(signal_map.get(m, [])) for m in members)
            members_str = ", ".join(members)
            print(f"  {tname:<25} {len(members):>10}  {total_pairs:>12}  {members_str}")

        # Gene super-category summary
        multi = is_reactome_toplevel_mode(self.sources)
        if multi:
            gene_to_cats = build_reactome_toplevel_map()
            cat_counts: Dict[str, int] = defaultdict(int)
            for cats in gene_to_cats.values():
                for cat in cats:
                    cat_counts[cat] += 1
            mode_label = f"reactome_toplevel (29 categories, {len(gene_to_cats)} genes, multi-mapped)"
        else:
            boosted = "chad_boosted" in self.sources
            gene_to_cat = build_gene_supercategory_map(
                self.supercategory_config,
                boosted=boosted,
            )
            cat_counts = defaultdict(int)
            for cat in gene_to_cat.values():
                cat_counts[cat] += 1
            src = "chad_boosted" if boosted else "chad"
            mode_label = f"{src} ({len(cat_counts)} categories, {len(gene_to_cat)} genes)"

        print(f"\n{'─' * 80}")
        print(f"  GENE SUPER-CATEGORIES — {mode_label}")
        print(f"{'─' * 80}")
        for cat, n in sorted(cat_counts.items(), key=lambda x: -x[1]):
            print(f"  {cat:<40} {n:>4} genes")

        n_runs_l1 = len(signal_map)
        n_runs_l2 = len(type_groups)
        print(f"\n  Level 1 (individual) mAP runs: {n_runs_l1}")
        print(f"  Level 2 (type) mAP runs:       {n_runs_l2}")
        print(f"  Total mAP runs:                {n_runs_l1 + n_runs_l2}")
        print(f"\n{'=' * 80}\n")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_yaml(path: Path) -> dict:
    if path.exists():
        with open(path) as f:
            return yaml.safe_load(f) or {}
    logger.warning(f"Config not found: {path}, using defaults")
    return {}


def _wrap_label(text: str, max_chars: int = 20) -> str:
    """Wrap a long label with newlines for plot readability."""
    if len(text) <= max_chars:
        return text
    if ", " in text:
        parts = text.split(", ", 1)
        return parts[0] + ",\n" + parts[1]
    mid = len(text) // 2
    best = text.rfind(" ", 0, mid + 5)
    if best == -1:
        best = text.find(" ", mid)
    if best == -1:
        return text
    return text[:best] + "\n" + text[best + 1:]


# ---------------------------------------------------------------------------
# Top-level SLURM job function (must be picklable)
# ---------------------------------------------------------------------------

def run_reporter_radar_job(
    output_dir: str,
    config_path: str,
    supercategory_path: str,
    norm_method: str = "ntc",
    radar_metric: str = "fraction_active",
    analysis_level: str = "both",
    reporter_filter: Optional[List[str]] = None,
    sources_str: str = "chad,reactome,regex,harmonizome",
) -> str:
    """Run reporter radar as a standalone SLURM job."""
    import traceback
    from types import SimpleNamespace

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    logging.getLogger("copairs").setLevel(logging.WARNING)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        data_shim = SimpleNamespace(
            experiment="cross_experiment",
            graph_output_path=output_dir,
        )
        config_shim = SimpleNamespace(experiment="cross_experiment")

        stage = ReporterRadarStage(
            data_context=data_shim,
            config=config_shim,
            level="guide",
            norm_method=norm_method,
            config_path=Path(config_path),
            supercategory_path=Path(supercategory_path),
            radar_metric=radar_metric,
            analysis_level=analysis_level,
            reporter_filter=reporter_filter,
            sources=parse_sources(sources_str),
        )
        stage._output_dir = output_dir / "14_reporter_radar"
        stage._output_dir.mkdir(parents=True, exist_ok=True)

        result = stage.run()

        if result.errors:
            return f"FAILED: {'; '.join(result.errors)}"
        return f"OK: {len(result.output_files)} files generated"

    except Exception as e:
        traceback.print_exc()
        return f"ERROR: {e}"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    """Run reporter radar as a standalone script."""
    import argparse
    from types import SimpleNamespace

    parser = argparse.ArgumentParser(
        description="Reporter Radar: Per-reporter mAP biological profiling with radar/spider plots"
    )
    parser.add_argument("-o", "--output-dir", default=None,
                        help="Output directory (default: auto-generated)")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH),
                        help=f"Config YAML path (default: {DEFAULT_CONFIG_PATH})")
    parser.add_argument("--supercategory-config", default=str(DEFAULT_SUPERCATEGORY_PATH),
                        help="Gene super-category mapping YAML")
    parser.add_argument("--norm-method", default="ntc", choices=["global", "ntc"],
                        help="Normalization method (default: ntc)")
    parser.add_argument("--radar-metric", default="fraction_active",
                        choices=["fraction_active", "mean_map", "auc_score"],
                        help="Radar plot value metric (default: fraction_active)")
    parser.add_argument("--level", default="both",
                        choices=["individual", "type", "both"],
                        help="Analysis level (default: both)")
    parser.add_argument("--reporters", default=None,
                        help="Comma-separated reporter labels to process (default: all)")
    parser.add_argument("--sources", default="chad_boosted",
                        help="Gene categorization source. "
                             "chad: CHAD only (8 cats, ~19%%). "
                             "chad_boosted: CHAD+keywords+regex+harmonizome (8 cats, ~98%%). "
                             "reactome_toplevel: Reactome 29 cats, multi-mapped (78%%). "
                             "(default: chad_boosted)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Discover reporters and print summary")

    slurm_group = parser.add_argument_group("SLURM options")
    slurm_group.add_argument("--slurm", action="store_true",
                             help="Submit as a SLURM job")
    slurm_group.add_argument("--no-wait", action="store_true",
                             help="Don't wait for SLURM job to complete")
    slurm_group.add_argument("--yes", "-y", action="store_true",
                             help="Skip confirmation prompt")
    slurm_group.add_argument("--slurm-memory", type=str, default="256GB",
                             help="Memory (default: 256GB)")
    slurm_group.add_argument("--slurm-time", type=int, default=480,
                             help="Time limit in minutes (default: 480)")
    slurm_group.add_argument("--slurm-cpus", type=int, default=16,
                             help="CPUs (default: 16)")

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    # Output dir
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = Path("/hpc/projects/icd.fast.ops/reporter_radar")
    output_dir.mkdir(parents=True, exist_ok=True)

    config_path = Path(args.config)
    supercategory_path = Path(args.supercategory_config)

    reporter_filter = None
    if args.reporters:
        reporter_filter = [r.strip() for r in args.reporters.split(",")]

    sources = parse_sources(args.sources)

    # Dry-run
    if args.dry_run:
        data_shim = SimpleNamespace(
            experiment="cross_experiment",
            graph_output_path=output_dir,
        )
        config_shim = SimpleNamespace(experiment="cross_experiment")
        stage = ReporterRadarStage(
            data_context=data_shim,
            config=config_shim,
            level="guide",
            config_path=config_path,
            supercategory_path=supercategory_path,
            sources=sources,
        )
        stage.dry_run()
        return

    # SLURM mode
    if args.slurm:
        from ops_utils.hpc.slurm_batch_utils import submit_parallel_jobs

        slurm_params = {
            "timeout_min": args.slurm_time,
            "mem": args.slurm_memory,
            "cpus_per_task": args.slurm_cpus,
            "slurm_partition": "cpu,gpu",
        }

        jobs = [{
            "name": "reporter_radar",
            "func": run_reporter_radar_job,
            "kwargs": {
                "output_dir": str(output_dir),
                "config_path": str(config_path),
                "supercategory_path": str(supercategory_path),
                "norm_method": args.norm_method,
                "radar_metric": args.radar_metric,
                "analysis_level": args.level,
                "reporter_filter": reporter_filter,
                "sources_str": args.sources,
            },
        }]

        if not args.yes:
            print(f"\nReporter Radar SLURM Job:")
            print(f"  Output:       {output_dir}")
            print(f"  Sources:      {sources_label(sources)}")
            print(f"  Norm:         {args.norm_method}")
            print(f"  Radar metric: {args.radar_metric}")
            print(f"  Level:        {args.level}")
            print(f"  Memory:       {args.slurm_memory}")
            print(f"  Time:         {args.slurm_time} min")
            print(f"  CPUs:         {args.slurm_cpus}")
            confirm = input("\nSubmit? [y/N] ").strip().lower()
            if confirm != "y":
                print("Cancelled.")
                return

        submit_result = submit_parallel_jobs(
            jobs_to_submit=jobs,
            experiment="reporter_radar",
            slurm_params=slurm_params,
            log_dir="reporter_radar",
            manifest_prefix="reporter_radar",
            wait_for_completion=not args.no_wait,
        )

        if submit_result.get("success"):
            print(f"\nJob submitted: {submit_result.get('base_job_id')}")
        else:
            print("\nJob submission failed!")
        return

    # Local mode
    data_shim = SimpleNamespace(
        experiment="cross_experiment",
        graph_output_path=output_dir,
    )
    config_shim = SimpleNamespace(experiment="cross_experiment")

    stage = ReporterRadarStage(
        data_context=data_shim,
        config=config_shim,
        level="guide",
        norm_method=args.norm_method,
        config_path=config_path,
        supercategory_path=supercategory_path,
        radar_metric=args.radar_metric,
        analysis_level=args.level,
        reporter_filter=reporter_filter,
        sources=sources,
    )
    stage._output_dir = output_dir / "14_reporter_radar"
    stage._output_dir.mkdir(parents=True, exist_ok=True)

    result = stage.run()

    print(f"\nOutput: {stage.output_dir}")
    print(f"Files: {len(result.output_files)}")
    if result.errors:
        print(f"Errors: {len(result.errors)}")
        for err in result.errors:
            print(f"  - {err}")


if __name__ == "__main__":
    main()
