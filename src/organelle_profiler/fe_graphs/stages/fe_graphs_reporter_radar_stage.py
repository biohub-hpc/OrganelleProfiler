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
    phenotypic_consistency_corum,
    phenotypic_consistency_manual_annotation,
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
    reporter_labels: List[str],
) -> Dict[str, List[str]]:
    """Group reporter labels by organelle type (part before comma).

    Examples:
      "lysosome, LAMP1" → type "lysosome"
      "mitochondria, TOMM20" → type "mitochondria"
      "Phase" → type "Phase"

    Returns dict: type_name → [reporter_label, ...]
    """
    type_to_reporters: Dict[str, List[str]] = defaultdict(list)
    for label in reporter_labels:
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

    DEFAULT_PCA_OPTIMIZED_DIR = "/hpc/projects/icd.fast.ops/organelle_attribution/pca_optimized_v2/dino/all"

    VALID_SOURCES  = ("chad", "chad_boosted", "reactome_toplevel")
    VALID_SCORES   = ("ratio", "mean_map")
    VALID_LEVELS   = ("individual", "type")
    VALID_METRICS  = ("activity", "distinctiveness", "corum", "chad")

    def __init__(
        self,
        data_context,
        config,
        level: str = "guide",
        norm_method: str = "ntc",
        config_path: Optional[Path] = None,
        supercategory_path: Optional[Path] = None,
        analysis_level: str = "individual",  # "individual" or "type" — one per job
        metric: str = "activity",           # which mAP metric to compute per job
        source: str = "chad_boosted",       # single ontology source per job
        reporter_filter: Optional[List[str]] = None,
        pca_optimized_dir: Optional[str] = None,
        downsampled: bool = False,
        **kwargs,
    ):
        super().__init__(data_context, config, level, **kwargs)
        self.norm_method = norm_method
        self.config_path = config_path or DEFAULT_CONFIG_PATH
        self.supercategory_path = supercategory_path or DEFAULT_SUPERCATEGORY_PATH
        self.analysis_level = analysis_level
        self.metric = metric
        self.source = source
        self.reporter_filter = reporter_filter
        self.downsampled = downsampled

        # Resolve PCA-optimized dir (swap all→downsampled when requested)
        base_pca = pca_optimized_dir or self.DEFAULT_PCA_OPTIMIZED_DIR
        if downsampled:
            base_pca = str(Path(base_pca).parent / "downsampled")
        self.pca_optimized_dir = Path(base_pca)

        # Load configs
        self.stage_config = _load_yaml(self.config_path)
        self.supercategory_config = _load_yaml(self.supercategory_path)

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

        # Step 1: Load PCA-optimized coembedding (guide level only needed for mAP)
        import anndata as ad
        logger.info(f"Step 1: Loading PCA-optimized data from {self.pca_optimized_dir}...")
        guide_path = self.pca_optimized_dir / "guide_pca_optimized.h5ad"
        if not guide_path.exists():
            result.add_error(f"PCA-optimized guide file not found: {guide_path}")
            return result

        gene_path = self.pca_optimized_dir / "gene_pca_optimized.h5ad"
        if not gene_path.exists():
            result.add_error(f"PCA-optimized gene file not found: {gene_path}")
            return result

        adata_guide_full = ad.read_h5ad(guide_path)
        adata_gene_full  = ad.read_h5ad(gene_path)
        logger.info(
            f"  Loaded guide: {adata_guide_full.n_obs} obs × {adata_guide_full.n_vars} features | "
            f"gene: {adata_gene_full.n_obs} obs × {adata_gene_full.n_vars} features"
        )

        # Build label→feature-columns map from var_name prefixes (label_PCN convention)
        import re as _re
        _pc_re = _re.compile(r'^(.+)_PC\d+$')
        label_to_cols: Dict[str, List[str]] = {}
        for v in adata_guide_full.var_names:
            m = _pc_re.match(v)
            prefix = m.group(1) if m else v
            label_to_cols.setdefault(prefix, []).append(v)

        reporter_labels = sorted(label_to_cols.keys())

        # Apply reporter filter if specified
        if self.reporter_filter:
            reporter_labels = [l for l in reporter_labels if l in self.reporter_filter]
            if not reporter_labels:
                result.add_error(f"No reporters matched filter: {self.reporter_filter}")
                return result

        logger.info(f"  {len(reporter_labels)} reporters to process")

        # Step 2: Build gene super-category mapping for this source
        logger.info(
            f"Step 2: Building gene super-category mapping (source={self.source})..."
        )
        sources_fs = frozenset({self.source})
        self._multi_mapping = is_reactome_toplevel_mode(sources_fs)
        if self._multi_mapping:
            self._gene_to_cats = build_reactome_toplevel_map()
            gene_to_cat = {}
        else:
            gene_to_cat = build_gene_supercategory_map(
                self.supercategory_config,
                boosted=(self.source == "chad_boosted"),
            )
            self._gene_to_cats = {}
        self._boosted = (self.source == "chad_boosted")

        # Output structure: 14_.../all/{level}/{metric}/{source}/
        data_subdir = "downsampled" if self.downsampled else "all"
        out_dir = self.output_dir / data_subdir / self.analysis_level / self.metric / self.source
        out_dir.mkdir(parents=True, exist_ok=True)

        # Save gene assignment CSV
        cat_csv_path = out_dir / "gene_supercategory_assignment.csv"
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

        # Step 3/4: Score reporters at the configured analysis level
        logger.info(
            f"Step 3: mAP scoring "
            f"(level={self.analysis_level}, metric={self.metric}, source={self.source})..."
        )
        if self.analysis_level == "individual":
            scored_results = self._run_all_reporters(
                reporter_labels, adata_guide_full, adata_gene_full, label_to_cols, gene_to_cat, result
            )
        else:  # "type"
            type_groups = group_reporters_by_type(reporter_labels)
            scored_results = self._run_all_reporter_types(
                type_groups, adata_guide_full, adata_gene_full, label_to_cols, gene_to_cat, result
            )

        # Step 5: Build radar matrices and generate plots
        logger.info("Step 4: Generating radar plots and summaries...")
        if scored_results:
            self._generate_level_outputs(scored_results, gene_to_cat, result, out_dir)

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
        reporter_labels: List[str],
        adata_guide_full,
        adata_gene_full,
        label_to_cols: Dict[str, List[str]],
        gene_to_cat: Dict[str, str],
        result: StageResult,
    ) -> Dict[str, Dict[str, Any]]:
        """Run mAP for each individual reporter."""
        all_results: Dict[str, Dict[str, Any]] = {}
        n_total = len(reporter_labels)

        for i, label in enumerate(sorted(reporter_labels), 1):
            n_cols = len(label_to_cols.get(label, []))
            logger.info(f"  [{i}/{n_total}] Reporter: {label} ({n_cols} features)")
            r = self._score_reporter([label], adata_guide_full, adata_gene_full, label_to_cols)
            if r is not None:
                all_results[label] = r
            else:
                logger.warning(f"    Skipped {label}")

        return all_results

    def _run_all_reporter_types(
        self,
        type_groups: Dict[str, List[str]],
        adata_guide_full,
        adata_gene_full,
        label_to_cols: Dict[str, List[str]],
        gene_to_cat: Dict[str, str],
        result: StageResult,
    ) -> Dict[str, Dict[str, Any]]:
        """Run mAP for each reporter-type (union of member reporter features)."""
        all_results: Dict[str, Dict[str, Any]] = {}
        n_total = len(type_groups)

        for i, (type_name, members) in enumerate(sorted(type_groups.items()), 1):
            n_cols = sum(len(label_to_cols.get(m, [])) for m in members)
            members_str = ", ".join(members)
            logger.info(
                f"  [{i}/{n_total}] Type: {type_name} "
                f"({len(members)} reporters, {n_cols} features) [{members_str}]"
            )
            r = self._score_reporter(members, adata_guide_full, adata_gene_full, label_to_cols)
            if r is not None:
                all_results[type_name] = r
            else:
                logger.warning(f"    Skipped type {type_name}")

        return all_results

    def _score_reporter(
        self,
        labels: List[str],
        adata_guide_full,
        adata_gene_full,
        label_to_cols: Dict[str, List[str]],
    ) -> Optional[Dict[str, Any]]:
        """Subset the PCA-optimized coembedding to the given reporter labels and run all 4 mAP metrics.

        Data is already normalized — no loading or normalization needed.

        DESIGN CHOICE: distinctiveness, CORUM, and CHAD are computed on ALL geneKOs
        (not filtered to active ones), matching the attribution stage behaviour. This
        keeps the perturbation set stable and comparable across reporters.
        """
        label_str = ", ".join(labels)
        try:
            # Subset guide and gene data to this reporter's features
            keep_cols = set()
            for lbl in labels:
                keep_cols.update(label_to_cols.get(lbl, []))

            col_mask_guide = np.array([v in keep_cols for v in adata_guide_full.var_names])
            col_mask_gene  = np.array([v in keep_cols for v in adata_gene_full.var_names])
            adata_guide = adata_guide_full[:, col_mask_guide].copy()
            adata_gene  = adata_gene_full[:,  col_mask_gene].copy()

            if adata_guide.n_obs < self._min_perturbations:
                logger.warning(
                    f"    {label_str}: only {adata_guide.n_obs} perturbations "
                    f"(min {self._min_perturbations}), skipping"
                )
                return None

            metric = self.metric

            # 1. Activity (always computed — fast and used as reference)
            t0 = time.time()
            activity_map, active_ratio = phenotypic_activity_assesment(
                adata_guide, plot_results=False, null_size=self._null_size,
            )
            activity_auc = compute_auc_score(activity_map)
            logger.info(
                f"    Activity ({time.time()-t0:.1f}s): "
                f"{active_ratio:.2%} active, AUC={activity_auc:.4f}"
            )

            # All-active map for non-activity metrics (all geneKOs, not just significant)
            _all_active_guide = pd.DataFrame({
                "perturbation": adata_guide.obs["perturbation"].unique(),
                "below_corrected_p": True,
            })
            _all_active_gene = pd.DataFrame({
                "perturbation": adata_gene.obs["perturbation"].unique(),
                "below_corrected_p": True,
            })

            distinct_map, distinctive_ratio, distinct_auc = None, 0.0, 0.0
            corum_map,    corum_ratio,        corum_auc    = None, 0.0, 0.0
            chad_map,     chad_ratio,         chad_auc     = None, 0.0, 0.0

            if metric == "distinctiveness":
                t1 = time.time()
                distinct_map, distinctive_ratio = phenotypic_distinctivness(
                    adata_guide, _all_active_guide, plot_results=False, null_size=self._null_size,
                )
                distinct_auc = compute_auc_score(distinct_map)
                logger.info(
                    f"    Distinctiveness ({time.time()-t1:.1f}s): "
                    f"{distinctive_ratio:.2%}, AUC={distinct_auc:.4f}"
                )

            elif metric == "corum":
                t2 = time.time()
                corum_map, corum_ratio = phenotypic_consistency_corum(
                    adata_gene, _all_active_gene, plot_results=False,
                    null_size=self._null_size, cache_similarity=True,
                )
                corum_auc = compute_auc_score(corum_map)
                logger.info(
                    f"    CORUM ({time.time()-t2:.1f}s): "
                    f"{corum_ratio:.2%}, AUC={corum_auc:.4f}"
                )

            elif metric == "chad":
                t3 = time.time()
                chad_map, chad_ratio = phenotypic_consistency_manual_annotation(
                    adata_gene, _all_active_gene, plot_results=False,
                    null_size=self._null_size, cache_similarity=True,
                )
                chad_auc = compute_auc_score(chad_map)
                logger.info(
                    f"    CHAD ({time.time()-t3:.1f}s): "
                    f"{chad_ratio:.2%}, AUC={chad_auc:.4f}"
                )

            return {
                "activity_map":      activity_map,
                "distinct_map":      distinct_map,
                "corum_map":         corum_map,
                "chad_map":          chad_map,
                "active_ratio":      active_ratio,
                "distinctive_ratio": distinctive_ratio,
                "corum_ratio":       corum_ratio,
                "chad_ratio":        chad_ratio,
                "activity_auc":      activity_auc,
                "distinct_auc":      distinct_auc,
                "corum_auc":         corum_auc,
                "chad_auc":          chad_auc,
                "n_perturbations":   adata_guide.n_obs,
                "n_features":        adata_guide.n_vars,
            }

        except Exception as e:
            logger.error(f"    Failed for {label_str}: {e}")
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
        result: StageResult,
        out_dir: Path,
    ) -> None:
        """Generate CSVs, radar plots, and heatmaps. One source/score/level per call."""
        # 1. Per-reporter mAP CSVs
        self._save_map_csvs(all_results, out_dir, result)

        # 2. Summary CSV
        summary_df = self._build_summary(all_results)
        summary_path = out_dir / "summary.csv"
        summary_df.to_csv(summary_path, index=False)
        result.add_file(summary_path)

        # 3. Radar matrices — both scores for this job's metric
        for score in self.VALID_SCORES:
            radar_df = self._compute_radar_matrix(all_results, gene_to_cat, self.metric, score)
            if radar_df is None or radar_df.empty:
                continue

            csv_path = out_dir / f"radar_matrix_{score}.csv"
            radar_df.to_csv(csv_path)
            result.add_file(csv_path)

            self._plot_radar_grid(radar_df, f"{self.metric}_{score}", out_dir, result)
            self._plot_radar_overlay(radar_df, f"{self.metric}_{score}", out_dir, result)
            self._plot_heatmap(radar_df, f"{self.metric}_{score}", out_dir, result)

    def replot_from_csvs(self, out_dir: Path) -> "StageResult":
        """Regenerate all plots from existing radar_matrix_*.csv files.

        Skips all data loading and mAP computation. Reads CSVs written by a
        previous run and re-runs the three plot methods for each score.
        """
        result = StageResult()
        found = 0
        for score in self.VALID_SCORES:
            csv_path = out_dir / f"radar_matrix_{score}.csv"
            if not csv_path.exists():
                logger.info(f"  Skipping {score}: {csv_path} not found")
                continue
            radar_df = pd.read_csv(csv_path, index_col=0)
            if radar_df.empty:
                continue
            metric_type = f"{self.metric}_{score}"
            self._plot_radar_grid(radar_df, metric_type, out_dir, result)
            self._plot_radar_overlay(radar_df, metric_type, out_dir, result)
            self._plot_heatmap(radar_df, metric_type, out_dir, result)
            found += 1
            logger.info(f"  Replotted {score}: {len(result.output_files)} files so far")
        logger.info(f"replot_from_csvs: {found} scores replotted -> {len(result.output_files)} files")
        return result

    def _save_map_csvs(
        self,
        all_results: Dict[str, Dict[str, Any]],
        out_dir: Path,
        result: StageResult,
    ) -> None:
        """Save stacked mAP results across all reporters for all 4 metrics."""
        for metric_type, key in [
            ("activity",        "activity_map"),
            ("distinctiveness", "distinct_map"),
            ("corum",           "corum_map"),
            ("chad",            "chad_map"),
        ]:
            frames = []
            for label, r in sorted(all_results.items()):
                if r.get(key) is not None:
                    df = r[key].copy()
                    df["reporter"] = label
                    frames.append(df)
            if frames:
                stacked = pd.concat(frames, ignore_index=True)
                path = out_dir / f"per_reporter_{metric_type}.csv"
                stacked.to_csv(path, index=False)
                result.add_file(path)

    @staticmethod
    def _mean_map(map_df) -> float:
        """Return mean mAP from a map DataFrame, or NaN if not computed."""
        if map_df is None:
            return float("nan")
        return float(map_df["mean_average_precision"].mean())

    def _build_summary(self, all_results: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
        """Build per-reporter summary table with all 4 mAP metrics × 2 scores."""
        rows = []
        for label, r in sorted(all_results.items()):
            rows.append({
                "reporter":          label,
                "n_perturbations":   r["n_perturbations"],
                "n_features":        r["n_features"],
                # ratio (% above threshold) — NaN when metric not computed this job
                "activity_ratio":    r["active_ratio"],
                "distinct_ratio":    r["distinctive_ratio"],
                "corum_ratio":       r["corum_ratio"],
                "chad_ratio":        r["chad_ratio"],
                # mean_map (unweighted mean mAP)
                "activity_mean_map": self._mean_map(r["activity_map"]),
                "distinct_mean_map": self._mean_map(r["distinct_map"]),
                "corum_mean_map":    self._mean_map(r["corum_map"]),
                "chad_mean_map":     self._mean_map(r["chad_map"]),
                # AUC (significance-weighted, stored for reference)
                "activity_auc":      r["activity_auc"],
                "distinct_auc":      r["distinct_auc"],
                "corum_auc":         r["corum_auc"],
                "chad_auc":          r["chad_auc"],
            })
        return pd.DataFrame(rows)

    def _compute_radar_matrix(
        self,
        all_results: Dict[str, Dict[str, Any]],
        gene_to_cat: Dict[str, str],
        metric_type: str,
        score: str,
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
        _map_keys = {
            "activity": "activity_map", "distinctiveness": "distinct_map",
            "corum": "corum_map",       "chad": "chad_map",
        }
        map_key = _map_keys[metric_type]
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
                    row[cat] = self._score_category(cat_df, score)
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
                    row[cat] = self._score_category(cat_df, score)
                rows[label] = row

        if not rows:
            return None

        df = pd.DataFrame.from_dict(rows, orient="index")
        df = df.fillna(0.0)
        # Drop columns with all zeros
        df = df.loc[:, (df != 0).any(axis=0)]
        return df

    def _score_category(self, cat_df: pd.DataFrame, score: str) -> float:
        """Aggregate a gene category's mAP rows into a single radar cell value."""
        if score == "mean_map":
            return float(cat_df["mean_average_precision"].mean())
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
            f"Reporter Radar — {metric_label}",
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
            f"Reporter Overlay — {metric_label}",
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
            ax=ax, cbar_kws={"label": metric_type},
            linewidths=0.5,
        )
        metric_label = "Activity" if metric_type == "activity" else "Distinctiveness"
        ax.set_title(
            f"Reporter × Category — {metric_label}",
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
    analysis_level: str = "individual",
    metric: str = "activity",
    source: str = "chad_boosted",
    reporter_filter: Optional[List[str]] = None,
    pca_optimized_dir: Optional[str] = None,
    downsampled: bool = False,
) -> str:
    """Run reporter radar as a standalone SLURM job (one source/score/level per job)."""
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
            analysis_level=analysis_level,
            metric=metric,
            source=source,
            reporter_filter=reporter_filter,
            pca_optimized_dir=pca_optimized_dir,
            downsampled=downsampled,
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
                        help="Output directory (default: /hpc/projects/icd.fast.ops/reporter_radar)")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH),
                        help=f"Config YAML path (default: {DEFAULT_CONFIG_PATH})")
    parser.add_argument("--supercategory-config", default=str(DEFAULT_SUPERCATEGORY_PATH),
                        help="Gene super-category mapping YAML")
    parser.add_argument("--norm-method", default="ntc", choices=["global", "ntc"],
                        help="Normalization method (default: ntc)")
    parser.add_argument("--metric", default="all",
                        choices=["all"] + list(ReporterRadarStage.VALID_METRICS),
                        help="mAP metric to compute. Default: all (submits one job per metric).")
    parser.add_argument("--source", default="all",
                        choices=["all"] + list(ReporterRadarStage.VALID_SOURCES),
                        help="Ontology source for gene categorization. Default: all (submits one job per source).")
    parser.add_argument("--level", default="all",
                        choices=["all"] + list(ReporterRadarStage.VALID_LEVELS),
                        help="individual (per reporter) or type (per organelle type). "
                             "Default: all (submits one job per level).")
    parser.add_argument("--reporters", default=None,
                        help="Comma-separated reporter labels to process (default: all)")
    parser.add_argument("--pca-optimized", type=str,
                        default=ReporterRadarStage.DEFAULT_PCA_OPTIMIZED_DIR,
                        help="Path to dir with guide_pca_optimized.h5ad "
                             "(default: pca_optimized_v2/dino/all)")
    parser.add_argument("--downsampled", action="store_true",
                        help="Use downsampled PCA data; outputs under .../downsampled/ instead of .../all/")
    parser.add_argument("--dry-run", action="store_true",
                        help="Discover reporters and print summary")
    parser.add_argument("--plot-only", action="store_true",
                        help="Skip computation; regenerate plots from existing radar_matrix_*.csv files")

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

    pca_path = args.pca_optimized
    if args.downsampled:
        pca_path = str(Path(pca_path).parent / "downsampled")

    # Resolve dimension lists
    sources  = list(ReporterRadarStage.VALID_SOURCES)  if args.source == "all" else [args.source]
    metrics  = list(ReporterRadarStage.VALID_METRICS)  if args.metric == "all" else [args.metric]
    levels   = list(ReporterRadarStage.VALID_LEVELS)   if args.level  == "all" else [args.level]

    # Dry-run
    if args.dry_run:
        data_shim = SimpleNamespace(experiment="cross_experiment", graph_output_path=output_dir)
        config_shim = SimpleNamespace(experiment="cross_experiment")
        stage = ReporterRadarStage(
            data_context=data_shim, config=config_shim, level="guide",
            config_path=config_path, supercategory_path=supercategory_path,
        )
        stage.dry_run()
        return

    # Plot-only — read existing CSVs and regenerate plots, no SLURM/computation
    if args.plot_only:
        data_shim = SimpleNamespace(experiment="cross_experiment", graph_output_path=output_dir)
        config_shim = SimpleNamespace(experiment="cross_experiment")
        data_subdir = "downsampled" if args.downsampled else "all"
        n_total = len(levels) * len(metrics) * len(sources)
        print(f"--plot-only: regenerating plots for {n_total} combinations from {output_dir}")
        for level in levels:
            for metric in metrics:
                for source in sources:
                    stage = ReporterRadarStage(
                        data_context=data_shim, config=config_shim, level="guide",
                        config_path=config_path, supercategory_path=supercategory_path,
                        analysis_level=level, metric=metric, source=source,
                        downsampled=args.downsampled,
                    )
                    stage._output_dir = output_dir / "14_reporter_radar"
                    out_dir = stage._output_dir / data_subdir / level / metric / source
                    logger.info(f"Replotting {level}/{metric}/{source} from {out_dir}")
                    stage.replot_from_csvs(out_dir)
        return

    common_kwargs = {
        "output_dir": str(output_dir),
        "config_path": str(config_path),
        "supercategory_path": str(supercategory_path),
        "norm_method": args.norm_method,
        "reporter_filter": reporter_filter,
        "pca_optimized_dir": pca_path,
        "downsampled": args.downsampled,
    }

    # SLURM mode — one job per (level, score, source)
    if args.slurm:
        from ops_utils.hpc.slurm_batch_utils import submit_parallel_jobs

        slurm_params = {
            "timeout_min": args.slurm_time,
            "mem": args.slurm_memory,
            "cpus_per_task": args.slurm_cpus,
            "slurm_partition": "cpu,gpu",
        }

        jobs = []
        for level in levels:
            for metric in metrics:
                for source in sources:
                    jobs.append({
                        "name": f"reporter_radar_{level}_{metric}_{source}",
                        "func": run_reporter_radar_job,
                        "kwargs": {**common_kwargs, "analysis_level": level, "metric": metric, "source": source},
                    })

        job_desc = f"{len(levels)} levels × {len(metrics)} metrics × {len(sources)} sources = {len(jobs)} jobs"

        if not args.yes:
            print(f"\nReporter Radar SLURM Job(s):")
            print(f"  Output:   {output_dir}")
            print(f"  PCA:      {pca_path}")
            print(f"  Data:     {'downsampled' if args.downsampled else 'all'}")
            print(f"  Levels:   {', '.join(levels)}")
            print(f"  Metrics:  {', '.join(metrics)}")
            print(f"  Sources:  {', '.join(sources)}")
            print(f"  Jobs:     {job_desc}")
            print(f"  Memory:   {args.slurm_memory}")
            print(f"  Time:     {args.slurm_time} min")
            print(f"  CPUs:     {args.slurm_cpus}")
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
            print(f"\nJob(s) submitted: {submit_result.get('base_job_id')}")
            print(f"  Jobs: {len(jobs)}")
        else:
            print("\nJob submission failed!")
        return

    # Local mode — iterate over all combinations
    for level in levels:
        for metric in metrics:
            for source in sources:
              print(f"\n--- level={level} | metric={metric} | source={source} ---")
              data_shim = SimpleNamespace(experiment="cross_experiment", graph_output_path=output_dir)
              config_shim = SimpleNamespace(experiment="cross_experiment")
              stage = ReporterRadarStage(
                  data_context=data_shim,
                  config=config_shim,
                  level="guide",
                  norm_method=args.norm_method,
                  config_path=config_path,
                  supercategory_path=supercategory_path,
                  analysis_level=level,
                  metric=metric,
                  source=source,
                  reporter_filter=reporter_filter,
                  pca_optimized_dir=pca_path,
                  downsampled=args.downsampled,
              )
              stage._output_dir = output_dir / "14_reporter_radar"
              stage._output_dir.mkdir(parents=True, exist_ok=True)

              result = stage.run()

              print(f"  Output: {stage.output_dir}")
              print(f"  Files: {len(result.output_files)}")
              if result.errors:
                  for err in result.errors:
                      print(f"  ERROR: {err}")


if __name__ == "__main__":
    main()
