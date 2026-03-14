"""
Organelle Attribution Stage: Leave-One-Out mAP Analysis on Cross-Experiment Dino Coembedding.

Answers: **which organelle channels drive phenotypic discrimination of gene perturbations?**

Workflow:
1. Discover all experiments with dino guide_bulked_*.h5ad files, filter bad experiments
2. Build biology-aware coembedding via concatenate_experiments_comprehensive
3. Run the 4 copairs mAP metrics (activity, distinctiveness, CORUM, CHAD) on full features
4. Greedy backward elimination (knock-out): cumulatively remove least important channel
5. Greedy forward selection (knock-in): cumulatively add most impactful channel
6. Compute deltas to reveal each organelle's contribution + which perturbations depend on it

Usage:
  # Local mode:
  python -m organelle_profiler.fe_graphs.stages.fe_graphs_organelle_attribution_stage -o /path/to/output

  # SLURM mode:
  python -m organelle_profiler.fe_graphs.stages.fe_graphs_organelle_attribution_stage --slurm
  python -m organelle_profiler.fe_graphs.stages.fe_graphs_organelle_attribution_stage --slurm --slurm-memory 500GB

CLI arguments:
  -o, --output-dir      Output directory (default: auto-generated)
  --config              Path to config YAML (default: organelle_attribution_config.yaml)
  --norm-method         Normalization method(s): global, ntc, or both (default: both)
  --fast                Post-aggregation PCA reduction for faster iteration
  --variance-threshold  Cumulative explained variance for PCA (default: 0.95)
  --pca-optimized       Path to pre-reduced PCA-optimized h5ad dir (from pca_optimization_stage)
  --dry-run             Discover experiments/channels and print summary without loading data

SLURM options:
  --slurm               Submit as a single SLURM job
  --slurm-memory        Memory (default: 500GB)
  --slurm-time          Time limit in minutes (default: 120)
  --slurm-cpus          CPUs (default: 16)
"""

import time
from collections import defaultdict

import numpy as np
import pandas as pd
import anndata as ad
import yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Optional, Dict, List, Tuple, Any
import logging

from ops_utils.analysis.map_scores import (
    phenotypic_activity_assesment,
    phenotypic_distinctivness,
    phenotypic_consistency_corum,
    phenotypic_consistency_manual_annotation,
    compute_auc_score,
)
from ops_utils.analysis.normalization import zscore_normalize, df_to_adata

try:
    from .fe_graphs_stage_base import BaseStage, StageResult
    from ..plotting.fe_graphs_utils import save_figure
except ImportError:
    from organelle_profiler.fe_graphs.stages.fe_graphs_stage_base import BaseStage, StageResult
    from organelle_profiler.fe_graphs.plotting.fe_graphs_utils import save_figure

logger = logging.getLogger(__name__)

# Suppress copairs internal INFO logging
logging.getLogger("copairs").setLevel(logging.WARNING)

import multiprocessing as _mp

# Default config path
DEFAULT_CONFIG_PATH = Path(__file__).parents[4] / "configs" / "organelle_attribution_config.yaml"

# ---------------------------------------------------------------------------
# Process-pool infrastructure for greedy elimination/selection
# ---------------------------------------------------------------------------
# Shared state populated before forking; children inherit via COW.
_POOL_SHARED: Dict[str, Any] = {}


def _pool_worker_init():
    """Set per-thread limits in forked workers to prevent oversubscription."""
    import os
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS",
                "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[var] = "1"


def _ensure_picklable():
    """Register pool functions in __main__ so multiprocessing can pickle them.

    When this module is loaded by submitit (SLURM), __main__ is submitit's
    wrapper, not our module.  multiprocessing.Pool pickles function references
    by qualified name, so it needs to find them in __main__.
    """
    import __main__
    for fn in (_pool_worker_init, _eval_candidate_process):
        if not hasattr(__main__, fn.__name__):
            setattr(__main__, fn.__name__, fn)


def _eval_candidate_process(args: tuple):
    """Evaluate a single candidate in a forked process (lightweight mode).

    Reads immutable shared data from ``_POOL_SHARED`` (inherited via fork COW).
    Args tuple: (candidate, keep_labels_list, run_label)
    Returns: (candidate, score, ratio) — only scalars to avoid pipe deadlock
        from serialising large DataFrames back to the parent process.
    """
    candidate, keep_labels_list, run_label = args
    keep_labels = set(keep_labels_list)

    sd = _POOL_SHARED
    adata_guide = sd["adata_guide"]
    adata_gene = sd["adata_gene"]
    label_to_idx_guide = sd["label_to_idx_guide"]
    label_to_idx_gene = sd["label_to_idx_gene"]
    stage = sd["stage"]

    # Fast integer-indexed subset
    idx_guide = np.concatenate([label_to_idx_guide[l] for l in sorted(keep_labels)])
    idx_gene = np.concatenate([label_to_idx_gene[l] for l in sorted(keep_labels)])
    idx_guide.sort()
    idx_gene.sort()

    subset_guide = adata_guide[:, idx_guide].copy()
    subset_gene = adata_gene[:, idx_gene].copy()

    if subset_guide.n_vars == 0:
        return candidate, None, None

    if stage.fast_mode and stage.pca_optimized_dir is None:
        subset_guide, subset_gene, _, _ = stage._pca_reduce_combined(
            subset_guide, subset_gene, label=run_label
        )

    r = stage._run_map_battery(subset_guide, subset_gene, run_label, lightweight=True)
    if r is None:
        return candidate, None, None
    # Return only scalars — avoids serialising large DataFrames through the pipe
    return candidate, stage._get_metric_score(r), stage._get_metric_ratio(r)

# Channel maps path (prefer fast_ops partition which has the most up-to-date entries)
CHANNEL_MAPS_PATH = Path("/hpc/projects/icd.fast.ops/configs/ops_channel_maps.yaml")
CHANNEL_MAPS_PATH_FALLBACK = Path("/hpc/projects/intracellular_dashboard/ops/configs/ops_channel_maps.yaml")

# Storage roots to search for dino features
DEFAULT_STORAGE_ROOTS = [
    Path("/hpc/projects/icd.fast.ops"),
]


def _check_copairs_installed():
    """Check copairs is available."""
    try:
        import copairs  # noqa: F401
    except ImportError:
        raise ImportError(
            "copairs is required for mAP scoring. Install with: pip install copairs"
        )


class OrganelleAttributionStage(BaseStage):
    """
    Leave-one-out mAP attribution analysis across all dino channels/experiments.

    For each channel label from ops_channel_maps.yaml, removes those features
    and measures how the 4 mAP metrics drop, revealing organelle-level contributions.

    Parameters
    ----------
    data_context : DataContext or SimpleNamespace
        Minimal context with graph_output_path.
    config : GraphConfig or SimpleNamespace
        Configuration.
    level : str
        Analysis level (default "guide").
    norm_methods : list of str
        Normalization methods to run. Default ["ntc", "global"] runs both.
    config_path : Path, optional
        Path to organelle_attribution_config.yaml.
    """

    STAGE_NUMBER = 13
    STAGE_NAME = "organelle_attribution"
    VALID_METRICS = ("activity", "distinctiveness", "corum", "chad")

    def __init__(
        self,
        data_context,
        config,
        level: str = "guide",
        norm_methods: Optional[List[str]] = None,
        config_path: Optional[Path] = None,
        fast_mode: bool = False,
        variance_threshold: float = 0.95,
        pca_optimized_dir: Optional[str] = None,
        **kwargs,
    ):
        self.mode = kwargs.pop("mode", "all")  # "all", "knockout", or "knockin"
        self.metric = kwargs.pop("metric", "activity")  # which metric drives scoring
        self.variance_sweep = kwargs.pop("variance_sweep", None)  # list of thresholds for baseline sweep
        self.agg_funcs = kwargs.pop("agg_funcs", None)  # multi-stat aggregation e.g. ["mean","std","min","max","median","sum"]
        self.enforce_phase = kwargs.pop("enforce_phase", False)  # force Phase as first channel in forward selection
        super().__init__(data_context, config, level, **kwargs)
        self.norm_methods = norm_methods or ["ntc"]
        self.config_path = config_path or DEFAULT_CONFIG_PATH
        self.fast_mode = fast_mode
        self.variance_threshold = variance_threshold
        self.pca_optimized_dir = Path(pca_optimized_dir) if pca_optimized_dir else None

        # Load attribution config
        if self.config_path.exists():
            with open(self.config_path) as f:
                self.attr_config = yaml.safe_load(f)
        else:
            logger.warning(f"Config not found: {self.config_path}, using defaults")
            self.attr_config = {}

        self._storage_roots = [
            Path(p) for p in self.attr_config.get("storage_roots", [str(p) for p in DEFAULT_STORAGE_ROOTS])
        ]
        self._feature_dir = self.attr_config.get("feature_dir", "dino_features")
        self._feature_type = self.attr_config.get("feature_type", "dinov3")
        self._join = self.attr_config.get("join", "inner")

    # -------------------------------------------------------------------------
    # Main run
    # -------------------------------------------------------------------------

    def run(self) -> StageResult:
        """Execute the full organelle attribution pipeline.

        Loads data once (steps 1-3), then runs the full mAP analysis
        (normalize → baseline → leave-one-out → attribution) separately
        for each requested normalization method.  Results are saved under
        ``<output_dir>/ntc_norm/`` and ``<output_dir>/global_norm/``.
        """
        result = StageResult()
        t0 = time.time()
        self.log_start("Organelle Attribution: Leave-One-Out mAP Analysis")

        # Step 1: Discover all dino experiment/channel pairs
        logger.info("Step 1: Discovering dino experiments...")
        exp_channel_pairs = self._discover_dino_experiments()
        if len(exp_channel_pairs) < 2:
            result.add_error(f"Need at least 2 experiment/channel pairs, found {len(exp_channel_pairs)}")
            return result
        logger.info(f"  Found {len(exp_channel_pairs)} experiment/channel pairs")

        # Step 1b: Build signal_map from resolved channel labels so the combiner
        #          groups by the same labels as dry-run (bypasses FeatureMetadata)
        import io, contextlib, warnings as _warnings
        from ops_utils.data.feature_metadata import FeatureMetadata
        fm = FeatureMetadata(metadata_path=str(CHANNEL_MAPS_PATH) if CHANNEL_MAPS_PATH.exists() else str(CHANNEL_MAPS_PATH_FALLBACK))
        signal_map: Dict[str, List[Tuple[str, str]]] = defaultdict(list)
        # Suppress FeatureMetadata's bare print() warnings during reverse lookup
        with contextlib.redirect_stdout(io.StringIO()):
            for exp, ch in exp_channel_pairs:
                resolved = self._resolve_channel_label(fm, exp, ch)
                signal_map[resolved["label"]].append((exp, ch))
        signal_map = dict(signal_map)
        logger.info(f"  Resolved {len(signal_map)} biological signal groups")

        # Step 2: Build biology-aware coembedding (done once for all norm methods)
        if self.pca_optimized_dir is not None:
            # Load pre-reduced PCA-optimized data
            logger.info(f"Step 2: Loading PCA-optimized data from {self.pca_optimized_dir}...")
            guide_path = self.pca_optimized_dir / "guide_pca_optimized.h5ad"
            gene_path = self.pca_optimized_dir / "gene_pca_optimized.h5ad"
            if not guide_path.exists() or not gene_path.exists():
                result.add_error(f"PCA-optimized files not found in {self.pca_optimized_dir}")
                return result
            adata_guide_raw = ad.read_h5ad(guide_path)
            adata_gene_raw = ad.read_h5ad(gene_path)
            logger.info(f"  Loaded pre-reduced data (already NTC-normalized, PCA-optimized)")
        else:
            logger.info("Step 2: Building biology-aware coembedding...")
            with _warnings.catch_warnings():
                _warnings.filterwarnings("ignore", message=".*names are not unique.*")
                adata_guide_raw, adata_gene_raw = self._load_and_combine(exp_channel_pairs, signal_map=signal_map)
        if adata_guide_raw is None:
            result.add_error("Failed to build coembedding")
            return result
        logger.info(
            f"  Guide-level: {adata_guide_raw.n_obs} perturbations, {adata_guide_raw.n_vars} features"
        )
        logger.info(
            f"  Gene-level: {adata_gene_raw.n_obs} perturbations, {adata_gene_raw.n_vars} features"
        )
        logger.info(
            f"  Config: agg={'multi-stat' if self.agg_funcs else 'mean-only'}, "
            f"PCA={'pca-optimized' if self.pca_optimized_dir else ('on (' + str(self.variance_threshold) + ')' if self.fast_mode else 'off')}, "
            f"norm={self.norm_methods}"
        )

        # Step 3: Build channel label -> feature column mapping (shared)
        logger.info("Step 3: Building channel label -> feature column mapping...")
        label_to_cols = self._build_label_to_columns(adata_guide_raw, exp_channel_pairs, signal_map=signal_map)
        logger.info(f"  {len(label_to_cols)} unique channel labels")
        for label, cols in sorted(label_to_cols.items()):
            logger.info(f"    {label}: {len(cols)} features")

        if len(label_to_cols) < 2:
            result.add_error(f"Need at least 2 channel labels, found {len(label_to_cols)}")
            return result

        # Steps 4-7: Run for each normalization method
        base_output_dir = self.output_dir
        for norm_method in self.norm_methods:
            logger.info(f"\n{'='*70}")
            logger.info(f"  NORMALIZATION: {norm_method} | METRIC: {self.metric} | MODE: {self.mode}")
            logger.info(f"{'='*70}")

            # Each norm method + metric gets its own output subdirectory
            norm_dir = base_output_dir / f"{norm_method}_norm" / self.metric
            norm_dir.mkdir(parents=True, exist_ok=True)
            self._output_dir = norm_dir

            # Copy raw data (normalization writes in-place)
            t_copy = time.time()
            adata_guide = adata_guide_raw.copy()
            adata_gene = adata_gene_raw.copy()
            logger.info(f"  Data copy: {time.time()-t_copy:.1f}s")

            # Step 4: Z-score normalize (skip if already normalized in PCA-optimized data)
            if self.pca_optimized_dir is not None:
                logger.info(f"Step 4: Skipping normalization (PCA-optimized data already normalized)")
            else:
                t_norm = time.time()
                logger.info(f"Step 4: Z-score normalizing (method={norm_method})...")
                adata_guide, adata_gene = self._normalize_adata(
                    adata_guide, adata_gene, norm_method
                )
                logger.info(f"  Normalization done: {time.time()-t_norm:.1f}s")

            # Step 4b/5: Variance sweep mode (baseline-only with multiple thresholds)
            if self.mode == "baseline" and self.variance_sweep:
                sweep_rows = []
                logger.info(
                    f"Step 4b/5: Variance sweep — testing {len(self.variance_sweep)} "
                    f"thresholds on {adata_guide.n_vars} features..."
                )
                # Pre-fit full PCA once, then slice to different n_keep values
                from sklearn.decomposition import PCA as _PCA
                X_guide = np.asarray(adata_guide.X, dtype=np.float64)
                X_gene = np.asarray(adata_gene.X, dtype=np.float64)
                X_guide = np.nan_to_num(X_guide, nan=0.0, posinf=0.0, neginf=0.0)
                X_gene = np.nan_to_num(X_gene, nan=0.0, posinf=0.0, neginf=0.0)
                max_components = min(adata_guide.n_vars, adata_guide.n_obs - 1)
                t_pca_fit = time.time()
                logger.info(f"  Fitting PCA with {max_components} components...")
                pca_full = _PCA(n_components=max_components)
                guide_transformed = pca_full.fit_transform(X_guide)
                gene_transformed = pca_full.transform(X_gene)
                cumvar = np.cumsum(pca_full.explained_variance_ratio_)
                logger.info(f"  PCA fit done in {time.time()-t_pca_fit:.1f}s")

                for thresh in self.variance_sweep:
                    t_step = time.time()
                    n_keep = int(np.searchsorted(cumvar, thresh) + 1)
                    n_keep = max(n_keep, 1)
                    n_keep = min(n_keep, max_components)
                    actual_var = float(cumvar[n_keep - 1])
                    logger.info(
                        f"  Threshold {thresh:.1%}: {n_keep} PCs "
                        f"(actual {actual_var:.2%} variance)"
                    )

                    pc_names = [f"PC{j}" for j in range(n_keep)]
                    sweep_guide = ad.AnnData(
                        X=guide_transformed[:, :n_keep].astype(np.float32),
                        obs=adata_guide.obs.copy(),
                        var=pd.DataFrame(index=pc_names),
                    )
                    sweep_gene = ad.AnnData(
                        X=gene_transformed[:, :n_keep].astype(np.float32),
                        obs=adata_gene.obs.copy(),
                        var=pd.DataFrame(index=pc_names),
                    )

                    sweep_result = self._run_map_battery(
                        sweep_guide, sweep_gene, f"sweep_{thresh:.3f}"
                    )
                    if sweep_result is not None:
                        auc = self._get_metric_score(sweep_result)
                        ratio = self._get_metric_ratio(sweep_result)
                        sweep_rows.append({
                            "variance_threshold": thresh,
                            "n_pcs": n_keep,
                            "actual_variance": actual_var,
                            "auc": auc,
                            "active_ratio": ratio,
                            "time_s": time.time() - t_step,
                        })
                        logger.info(
                            f"    AUC={auc:.4f}, ratio={ratio:.2%} "
                            f"({time.time()-t_step:.1f}s)"
                        )

                # Free large arrays
                del guide_transformed, gene_transformed, pca_full

                # Save sweep results
                if sweep_rows:
                    sweep_df = pd.DataFrame(sweep_rows)
                    sweep_path = self.output_dir / "variance_sweep_results.csv"
                    sweep_df.to_csv(sweep_path, index=False)
                    result.add_file(sweep_path)
                    logger.info(f"\n{'='*60}")
                    logger.info("VARIANCE SWEEP RESULTS:")
                    logger.info(f"{'='*60}")
                    for row in sweep_rows:
                        logger.info(
                            f"  {row['variance_threshold']:>6.1%} → "
                            f"{row['n_pcs']:>4d} PCs | "
                            f"AUC={row['auc']:.4f} | "
                            f"active={row['active_ratio']:.2%}"
                        )
                    logger.info(f"{'='*60}")
                    logger.info(f"Saved to {sweep_path}")
                continue  # skip normal baseline/elimination flow

            # Step 4b: Combined PCA reduction for baseline (fast mode)
            # Skip if using PCA-optimized data (already reduced at cell level)
            if self.fast_mode and self.pca_optimized_dir is None:
                t_pca = time.time()
                logger.info(
                    f"Step 4b: Combined PCA reduction "
                    f"(variance threshold={self.variance_threshold:.0%})..."
                )
                baseline_guide, baseline_gene, n_pcs, expl_var = (
                    self._pca_reduce_combined(adata_guide, adata_gene, label="baseline")
                )
                logger.info(
                    f"  PCA: {adata_guide.n_vars} → {n_pcs} PCs "
                    f"({expl_var:.1%} variance, {time.time()-t_pca:.1f}s)"
                )
                # Save reduction report
                report_df = pd.DataFrame([{
                    "original_features": adata_guide.n_vars,
                    "kept_components": n_pcs,
                    "explained_variance": expl_var,
                }])
                report_df.to_csv(self.output_dir / "pca_reduction_report.csv", index=False)
            else:
                baseline_guide, baseline_gene = adata_guide, adata_gene

            # Step 5: Baseline mAP
            logger.info("Step 5: Running baseline mAP battery...")
            baseline = self._run_map_battery(baseline_guide, baseline_gene, "baseline")
            if baseline is None:
                result.add_error(f"Baseline mAP computation failed ({norm_method} norm)")
                continue

            self._save_baseline(baseline, result)
            logger.info(
                f"  Baseline ({norm_method}, {self.metric}): "
                f"AUC={self._get_metric_score(baseline):.4f}, "
                f"ratio={self._get_metric_ratio(baseline):.2%}"
            )

            # Step 6: Greedy backward elimination (cumulative knock-out)
            # Pass RAW normalized data + label_to_cols; PCA is redone per candidate
            if self.mode in ("all", "knockout"):
                logger.info("Step 6: Greedy backward elimination (knock-out)...")
                ablation_results = self._greedy_backward_elimination(
                    adata_guide, adata_gene, label_to_cols, baseline,
                    norm_method, result,
                )

                # Step 7: Compute attribution and generate outputs
                logger.info("Step 7: Computing attribution and generating outputs...")
                self._compute_and_save_attribution(baseline, ablation_results, label_to_cols, result)
            else:
                logger.info("Skipping knock-out (mode=%s)", self.mode)

            # Step 8: Greedy forward selection (cumulative knock-in)
            # Pass RAW normalized data + label_to_cols; PCA is redone per candidate
            if self.mode in ("all", "knockin"):
                logger.info("Step 8: Greedy forward selection (knock-in)...")
                self._greedy_minimal_set(
                    adata_guide, adata_gene, label_to_cols, baseline, norm_method, result
                )
            else:
                logger.info("Skipping knock-in (mode=%s)", self.mode)

        # Restore base output dir
        self._output_dir = base_output_dir

        elapsed = time.time() - t0
        logger.info(f"\nOrganelle attribution complete in {elapsed:.0f}s")
        logger.info(f"Norm methods: {', '.join(self.norm_methods)}")
        logger.info(f"Output: {base_output_dir}")
        self.log_complete(result)
        return result

    # -------------------------------------------------------------------------
    # Step 1: Discover experiments
    # -------------------------------------------------------------------------

    def _discover_dino_experiments(self) -> List[Tuple[str, str]]:
        """
        Discover all experiments with dino guide_bulked_*.h5ad files.

        Filters out bad experiments via is_excluded().
        Returns list of (experiment_name, channel) pairs.
        """
        from ops_utils.data.bad_experiments import is_excluded

        pairs = []
        seen = set()

        for root in self._storage_roots:
            if not root.exists():
                continue

            # Scan for experiment directories
            try:
                exp_dirs = sorted(root.iterdir())
            except PermissionError:
                logger.warning(f"  Permission denied: {root}")
                continue

            for exp_dir in exp_dirs:
                if not exp_dir.is_dir():
                    continue

                exp_name = exp_dir.name

                # Skip hidden directories (.Trash, .snapshot, etc.)
                if exp_name.startswith("."):
                    continue

                exp_short = exp_name.split("_")[0]

                # Skip if already found or excluded
                if exp_short in seen:
                    continue
                if is_excluded(exp_short):
                    continue

                # Check for dino anndata objects
                anndata_dir = exp_dir / "3-assembly" / self._feature_dir / "anndata_objects"
                try:
                    if not anndata_dir.exists():
                        continue
                except PermissionError:
                    continue

                # Find guide_bulked_*.h5ad files (skip umap_ embeddings and Cell Painting channels)
                for h5ad in sorted(anndata_dir.glob("guide_bulked_*.h5ad")):
                    channel = h5ad.stem.replace("guide_bulked_", "")
                    if channel.startswith("umap_"):
                        continue
                    if channel.startswith(("CP1_", "CP2_")):
                        continue
                    pairs.append((exp_name, channel))

                if any(exp_name == p[0] for p in pairs):
                    seen.add(exp_short)

        logger.info(f"  Discovered {len(pairs)} (experiment, channel) pairs "
                     f"across {len(seen)} experiments")
        return pairs

    # -------------------------------------------------------------------------
    # Channel label resolution (handles both channel-named and reporter-named h5ad files)
    # -------------------------------------------------------------------------

    @staticmethod
    def _resolve_channel_label(fm, experiment: str, file_channel: str) -> Dict[str, str]:
        """
        Resolve a channel name from an h5ad filename to its biological label.

        h5ad files are sometimes named by microscope channel (GFP, mCherry, Phase)
        and sometimes by reporter protein (EEA1, TOMM70A, LAMP1), and sometimes
        by live-cell dye name (LysoTracker_live-cell_dye).  This tries:

        1. Direct lookup: fm.get_channel_info(exp, file_channel)
           Handles: GFP, mCherry, Phase/Phase2D→BF, CP1_*/CP2_*
        2. Reverse lookup (exact): scan YAML channels, match marker == file_channel
           Handles: EEA1 → finds GFP channel with label "early endosome, EEA1"
        3. Reverse lookup (fuzzy): normalize underscores→spaces and compare
           Handles: LysoTracker_live-cell_dye → "LysoTracker live-cell dye"
                    H2B_(TOMM20-combo) → "H2B (TOMM20-combo)"
        4. Cell Painting channels (CP1_organelle_marker)

        Special rules:
        - Phase2D is grouped with Phase (both are label-free brightfield)
        - Bare microscope channels (GFP, mCherry) that resolve to label-free
          via direct lookup are fine; if not in YAML, they stay unresolved.

        Returns dict with keys: label, short, yaml_channel, method
        """
        exp_short = experiment.split("_")[0]

        # Phase2D → treat as Phase (same label-free brightfield)
        _PHASE_ALIASES = {"Phase2D", "Phase3D"}
        if file_channel in _PHASE_ALIASES:
            file_channel = "Phase"

        # --- Attempt 1: direct lookup (channel name or Phase alias) ---
        info = fm.get_channel_info(experiment, file_channel)
        if info.get("label") != "unknown":
            label = info["label"]
            # "no label" means autofluorescence — tag with the microscope channel
            if label == "no label":
                yaml_ch = info["channel_name"]
                label = f"autofluorescence, {yaml_ch.lower()}"
                return {
                    "label": label,
                    "short": f"autofluorescence_{yaml_ch.lower()}",
                    "yaml_channel": yaml_ch,
                    "method": "direct_autofluorescence",
                }
            return {
                "label": label,
                "short": fm.get_short_label(experiment, file_channel),
                "yaml_channel": info["channel_name"],
                "method": "direct",
            }

        # --- Attempt 2: reverse lookup — file_channel might be a reporter name ---
        # Normalize helper: underscores/hyphens→spaces, case-insensitive
        def _norm(s: str) -> str:
            return s.replace("_", " ").replace("-", " ").lower()

        # Collapsed helper: strip ALL separators for CellProfiler names
        # (CP filenames like "BODIPYlivecelldye" vs YAML "BODIPY_live_cell_dye")
        def _collapse(s: str) -> str:
            return s.replace("_", "").replace("-", "").replace(" ", "").lower()

        file_norm = _norm(file_channel)
        file_collapsed = _collapse(file_channel)

        if exp_short in fm.metadata:
            for ch_entry in fm.metadata[exp_short]:
                if not isinstance(ch_entry, dict) or "label" not in ch_entry:
                    continue
                label = ch_entry["label"]
                # 2a. Exact marker match
                if "," in label:
                    marker = label.split(",", 1)[1].strip()
                    if marker == file_channel:
                        return {
                            "label": label,
                            "short": marker,
                            "yaml_channel": ch_entry.get("channel_name", file_channel),
                            "method": "reverse_marker",
                        }
                # 2b. Exact whole-label match (e.g., "5xUPRE")
                if label == file_channel:
                    return {
                        "label": label,
                        "short": file_channel,
                        "yaml_channel": ch_entry.get("channel_name", file_channel),
                        "method": "reverse_label",
                    }
                # 2c. Fuzzy marker match (normalize both sides: _→space, case-insensitive)
                if "," in label:
                    marker = label.split(",", 1)[1].strip()
                    if _norm(marker) == file_norm:
                        return {
                            "label": label,
                            "short": marker,
                            "yaml_channel": ch_entry.get("channel_name", file_channel),
                            "method": "reverse_marker_fuzzy",
                        }
                # 2d. Fuzzy whole-label match
                if _norm(label) == file_norm:
                    return {
                        "label": label,
                        "short": label,
                        "yaml_channel": ch_entry.get("channel_name", file_channel),
                        "method": "reverse_label_fuzzy",
                    }
                # 2e. Collapsed marker match (all separators stripped — for CellProfiler filenames)
                if "," in label:
                    marker = label.split(",", 1)[1].strip()
                    if _collapse(marker) == file_collapsed:
                        return {
                            "label": label,
                            "short": marker,
                            "yaml_channel": ch_entry.get("channel_name", file_channel),
                            "method": "reverse_marker_collapsed",
                        }
                # 2f. Collapsed whole-label match
                if _collapse(label) == file_collapsed:
                    return {
                        "label": label,
                        "short": label,
                        "yaml_channel": ch_entry.get("channel_name", file_channel),
                        "method": "reverse_label_collapsed",
                    }

        # --- Attempt 3: Cell Painting channels (CP1_organelle_marker) ---
        if file_channel.startswith(("CP1_", "CP2_")):
            parts = file_channel.split("_", 2)
            if len(parts) == 3:
                return {
                    "label": f"{parts[1]}, {parts[2]}",
                    "short": parts[2],
                    "yaml_channel": file_channel,
                    "method": "cellpainting",
                }

        # --- Attempt 4: hardcoded fixes for known naming mismatches ---
        _HARDCODED = {
            # File says "emission" but YAML says "excitation"
            "ChromaLive_488_emission": "ChromaLIVE 488 excitation",
            "ChromaLive488emission": "ChromaLIVE 488 excitation",  # CellProfiler variant
            # CellProfiler truncated names that can't be fuzzy-matched
            "FastAct": "actin filament, FastAct_SPY555 Live Cell Dye",
            "unlabeled": "no label",
        }
        if file_channel in _HARDCODED:
            hardcoded_label = _HARDCODED[file_channel]
            return {
                "label": hardcoded_label,
                "short": hardcoded_label,
                "yaml_channel": file_channel,
                "method": "hardcoded",
            }

        # --- Unresolved ---
        return {
            "label": f"(unmapped: {file_channel})",
            "short": file_channel,
            "yaml_channel": file_channel,
            "method": "unresolved",
        }

    # -------------------------------------------------------------------------
    # Dry-run: discover only, no data loading
    # -------------------------------------------------------------------------

    def dry_run(self) -> None:
        """Discover experiments/channels and print summary without loading data."""
        from ops_utils.data.feature_metadata import FeatureMetadata
        from collections import defaultdict

        print("\n" + "=" * 80)
        print("  ORGANELLE ATTRIBUTION — DRY RUN")
        print("=" * 80)

        # Step 1: Discover experiment/channel pairs
        exp_channel_pairs = self._discover_dino_experiments()

        if not exp_channel_pairs:
            print("\n  No experiment/channel pairs found!")
            print(f"  Storage roots searched: {[str(r) for r in self._storage_roots]}")
            print(f"  Feature dir: {self._feature_dir}")
            return

        # Group by experiment
        exp_to_channels: Dict[str, List[str]] = defaultdict(list)
        for exp, ch in exp_channel_pairs:
            exp_to_channels[exp].append(ch)

        # Resolve channel labels via FeatureMetadata (handles both naming conventions)
        fm = FeatureMetadata(metadata_path=str(CHANNEL_MAPS_PATH) if CHANNEL_MAPS_PATH.exists() else str(CHANNEL_MAPS_PATH_FALLBACK))
        label_to_exp_channels: Dict[str, List[Tuple[str, str]]] = defaultdict(list)
        exp_channel_to_info: Dict[Tuple[str, str], Dict] = {}
        unresolved = []

        for exp, ch in exp_channel_pairs:
            resolved = self._resolve_channel_label(fm, exp, ch)
            exp_channel_to_info[(exp, ch)] = resolved
            label = resolved["label"]
            label_to_exp_channels[label].append((exp, ch))
            if resolved["method"] == "unresolved":
                unresolved.append((exp, ch))

        # --- Print experiment summary ---
        print(f"\n  Storage roots: {[str(r) for r in self._storage_roots]}")
        print(f"  Feature dir:   {self._feature_dir}")
        print(f"  Feature type:  {self._feature_type}")
        print(f"  Join:          {self._join}")
        print(f"\n  Experiments:   {len(exp_to_channels)}")
        print(f"  Total pairs:   {len(exp_channel_pairs)}")
        print(f"  Unique labels: {len(label_to_exp_channels)}")
        if unresolved:
            print(f"  Unresolved:    {len(unresolved)} (no YAML mapping)")

        # --- Per-experiment table ---
        print(f"\n{'─' * 100}")
        print(f"  {'EXPERIMENT':<30} {'CH':>3}  {'FILE CHANNEL':<25} {'RESOLVED LABEL':<30} {'VIA'}")
        print(f"{'─' * 100}")
        for exp in sorted(exp_to_channels.keys()):
            channels = sorted(exp_to_channels[exp])
            exp_short = exp.split("_")[0]
            for i, ch in enumerate(channels):
                info = exp_channel_to_info[(exp, ch)]
                exp_col = f"  {exp:<30}" if i == 0 else f"  {'':<30}"
                ch_count = f"{len(channels):>3}" if i == 0 else f"{'':>3}"
                method_tag = info["method"]
                label_display = info["label"]
                if len(label_display) > 28:
                    label_display = label_display[:25] + "..."
                print(f"{exp_col} {ch_count}  {ch:<25} {label_display:<30} {method_tag}")

        # --- Per-label table ---
        print(f"\n{'─' * 80}")
        print(f"  {'CHANNEL LABEL':<45} {'EXP×CH PAIRS':>12}   {'UNIQUE EXPS':>11}")
        print(f"{'─' * 80}")
        for label in sorted(label_to_exp_channels.keys()):
            pairs = label_to_exp_channels[label]
            unique_exps = len(set(e for e, _ in pairs))
            print(f"  {label:<45} {len(pairs):>12}   {unique_exps:>11}")

        # --- Per-label detail (which exp has which channel) ---
        print(f"\n{'─' * 80}")
        print("  LABEL → EXPERIMENT / CHANNEL DETAIL")
        print(f"{'─' * 80}")
        for label in sorted(label_to_exp_channels.keys()):
            pairs = label_to_exp_channels[label]
            print(f"\n  {label}:")
            for exp, ch in sorted(pairs):
                exp_short = exp.split("_")[0]
                info = exp_channel_to_info[(exp, ch)]
                via = f" (via {info['yaml_channel']})" if info["yaml_channel"] != ch else ""
                print(f"    {exp_short:<15} {ch}{via}")

        # --- What the ablation would look like ---
        print(f"\n{'─' * 80}")
        print("  ABLATION PLAN (leave-one-out)")
        print(f"{'─' * 80}")
        total_pairs = len(exp_channel_pairs)
        for label in sorted(label_to_exp_channels.keys()):
            n_remove = len(label_to_exp_channels[label])
            n_remain = total_pairs - n_remove
            print(f"  Remove '{label}': {n_remove} pairs removed, ~{n_remain} remaining")

        n_labels = len(label_to_exp_channels)
        n_fwd_runs = n_labels * (n_labels + 1) // 2
        print(f"\n  Leave-one-out runs:      {n_labels}")
        print(f"  Forward selection runs:  ~{n_fwd_runs} (greedy, upper bound)")
        print(f"  Total mAP battery runs:  ~{n_labels + n_fwd_runs + 1} (incl. baseline)")
        print(f"\n  Norm methods: {', '.join(self.norm_methods)}")
        print(f"  (above counts × {len(self.norm_methods)} norm methods)")
        print(f"\n{'=' * 80}\n")

    # -------------------------------------------------------------------------
    # Step 2: Load and combine
    # -------------------------------------------------------------------------

    def _load_and_combine(
        self,
        exp_channel_pairs: List[Tuple[str, str]],
        signal_map: Optional[Dict[str, List[Tuple[str, str]]]] = None,
    ) -> Tuple[Optional[ad.AnnData], Optional[ad.AnnData]]:
        """
        Build biology-aware coembedding using concatenate_experiments_comprehensive.

        Parameters
        ----------
        signal_map : dict, optional
            Pre-resolved biological signal grouping from _resolve_channel_label.
            When provided, the combiner skips its own FeatureMetadata lookup.

        Returns (adata_guide, adata_gene) or (None, None) on failure.
        """
        try:
            from ops_model.features.anndata_utils import concatenate_experiments_comprehensive

            # Use the fast_ops YAML so FeatureMetadata can resolve all experiments
            _maps = str(CHANNEL_MAPS_PATH) if CHANNEL_MAPS_PATH.exists() else str(CHANNEL_MAPS_PATH_FALLBACK)

            adata_guide, adata_gene = concatenate_experiments_comprehensive(
                experiments_channels=exp_channel_pairs,
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
                metadata_path=_maps,
                signal_map=signal_map,
                agg_funcs=self.agg_funcs,
            )
            return adata_guide, adata_gene

        except Exception as e:
            logger.error(f"Failed to build coembedding: {e}")
            import traceback
            traceback.print_exc()
            return None, None

    # -------------------------------------------------------------------------
    # Step 3: Build label -> column mapping
    # -------------------------------------------------------------------------

    def _build_label_to_columns(
        self,
        adata: ad.AnnData,
        exp_channel_pairs: List[Tuple[str, str]],
        signal_map: Optional[Dict[str, List[Tuple[str, str]]]] = None,
    ) -> Dict[str, List[str]]:
        """
        Map each channel label to the corresponding feature columns in the
        combined adata.

        When *signal_map* is provided (the same dict passed to the combiner),
        the biological signal group names are the var-name prefixes, so we
        match directly — no FeatureMetadata lookup needed.
        """
        all_var_names = list(adata.var_names)
        label_to_cols: Dict[str, List[str]] = {}

        if signal_map is not None:
            # signal_map keys = biological signal group names = var_name prefixes
            for label in signal_map:
                matching = [v for v in all_var_names if v.startswith(f"{label}_")]
                if matching:
                    label_to_cols[label] = matching
        else:
            # Fallback: group by prefix (everything before the last _digit)
            prefix_groups: Dict[str, List[str]] = {}
            for v in all_var_names:
                parts = v.rsplit("_", 1)
                prefix = parts[0] if len(parts) == 2 and parts[1].isdigit() else v
                if prefix not in prefix_groups:
                    prefix_groups[prefix] = []
                prefix_groups[prefix].append(v)
            label_to_cols = prefix_groups

        # Check for unmatched columns
        matched = set()
        for cols in label_to_cols.values():
            matched.update(cols)
        unmatched = [v for v in all_var_names if v not in matched]
        if unmatched:
            prefix_groups_um: Dict[str, List[str]] = {}
            for v in unmatched:
                parts = v.rsplit("_", 1)
                prefix = parts[0] if len(parts) == 2 and parts[1].isdigit() else v
                if prefix not in prefix_groups_um:
                    prefix_groups_um[prefix] = []
                prefix_groups_um[prefix].append(v)
            for prefix, cols in prefix_groups_um.items():
                label = f"unknown ({prefix})"
                label_to_cols[label] = cols
                logger.warning(f"  {len(cols)} features with unmatched prefix '{prefix}'")

        return label_to_cols

    # -------------------------------------------------------------------------
    # Step 4: Normalize
    # -------------------------------------------------------------------------

    def _normalize_adata(
        self,
        adata_guide: ad.AnnData,
        adata_gene: ad.AnnData,
        norm_method: str = "ntc",
    ) -> Tuple[ad.AnnData, ad.AnnData]:
        """Z-score normalize guide-level, then re-aggregate to gene-level.

        Normalizing at guide-level first ensures proper NTC normalization
        (210 NTC guides available). Gene-level is then derived by averaging
        the already-normalized guide features per gene, avoiding the problem
        of only 1 NTC row at gene-level.
        """
        from ops_model.features.anndata_utils import aggregate_to_level

        # 1. Normalize guide-level (has full NTC guide population)
        feature_cols = list(adata_guide.var_names)
        df = pd.DataFrame(adata_guide.X, columns=feature_cols)
        for col in adata_guide.obs.columns:
            df[col] = adata_guide.obs[col].values

        df = zscore_normalize(
            df, feature_cols,
            method=norm_method,
            perturbation_col="perturbation",
        )
        adata_guide.X = df[feature_cols].values.astype(np.float32)
        logger.info(f"  Normalized guide-level ({norm_method}): {adata_guide.n_obs} obs, {adata_guide.n_vars} features")

        # 2. Re-aggregate normalized guide data to gene-level
        min_guides = self.attr_config.get("min_guides_per_perturbation", 2)
        adata_gene = aggregate_to_level(
            adata_guide, "gene",
            preserve_batch_info=False,
            subsample_controls=False,
        )
        logger.info(
            f"  Re-aggregated to gene-level from normalized guides: "
            f"{adata_gene.n_obs} genes, {adata_gene.n_vars} features"
        )

        return adata_guide, adata_gene

    # -------------------------------------------------------------------------
    # Step 4b: Per-channel PCA reduction (fast mode)
    # -------------------------------------------------------------------------

    # Minimum features per channel group to attempt PCA (smaller groups kept as-is)
    _MIN_FEATURES_FOR_PCA = 10

    def _pca_reduce_combined(
        self,
        adata_guide: ad.AnnData,
        adata_gene: ad.AnnData,
        label: str = "",
    ) -> Tuple[ad.AnnData, ad.AnnData, int, float]:
        """
        Combined PCA reduction across ALL features in the given AnnData objects.

        Fits PCA on the guide-level data (typically more observations) and keeps
        the minimum number of components to reach ``self.variance_threshold``
        cumulative explained variance.  The same transform is applied to gene-level.

        Parameters
        ----------
        adata_guide, adata_gene : AnnData
            Input data (raw concatenated features from all included channels).
        label : str
            Label for logging (e.g. "baseline", "bwd_without_X").

        Returns
        -------
        reduced_guide, reduced_gene : AnnData
            PCA-reduced AnnData objects with PC0..PCn as var_names.
        n_components : int
            Number of PCs kept.
        explained_variance : float
            Cumulative explained variance of kept components.
        """
        from sklearn.decomposition import PCA

        n_features = adata_guide.n_vars

        X_guide = np.asarray(adata_guide.X, dtype=np.float64)
        X_gene = np.asarray(adata_gene.X, dtype=np.float64)

        # Replace NaN/inf
        X_guide = np.nan_to_num(X_guide, nan=0.0, posinf=0.0, neginf=0.0)
        X_gene = np.nan_to_num(X_gene, nan=0.0, posinf=0.0, neginf=0.0)

        max_components = min(n_features, adata_guide.n_obs - 1)
        pca = PCA(n_components=max_components)
        guide_transformed = pca.fit_transform(X_guide)

        # Find n_keep: minimum components to reach variance_threshold
        cumvar = np.cumsum(pca.explained_variance_ratio_)
        n_keep = int(np.searchsorted(cumvar, self.variance_threshold) + 1)
        n_keep = max(n_keep, 1)
        n_keep = min(n_keep, max_components)

        explained = float(cumvar[n_keep - 1])

        guide_reduced = guide_transformed[:, :n_keep].astype(np.float32)
        gene_reduced = pca.transform(X_gene)[:, :n_keep].astype(np.float32)

        pc_names = [f"PC{j}" for j in range(n_keep)]

        reduced_guide = ad.AnnData(
            X=guide_reduced,
            obs=adata_guide.obs.copy(),
            var=pd.DataFrame(index=pc_names),
        )
        reduced_gene = ad.AnnData(
            X=gene_reduced,
            obs=adata_gene.obs.copy(),
            var=pd.DataFrame(index=pc_names),
        )

        return reduced_guide, reduced_gene, n_keep, explained

    def _subset_channels(
        self,
        adata_guide: ad.AnnData,
        adata_gene: ad.AnnData,
        label_to_cols: Dict[str, List[str]],
        keep_labels: set,
    ) -> Tuple[ad.AnnData, ad.AnnData]:
        """Subset AnnData to only features belonging to keep_labels channels."""
        keep_cols = set()
        for lbl in keep_labels:
            keep_cols.update(label_to_cols[lbl])

        all_guide = list(adata_guide.var_names)
        all_gene = list(adata_gene.var_names)
        mask_guide = np.array([v in keep_cols for v in all_guide])
        mask_gene = np.array([v in keep_cols for v in all_gene])

        return adata_guide[:, mask_guide].copy(), adata_gene[:, mask_gene].copy()

    @staticmethod
    def _precompute_label_indices(
        adata: ad.AnnData,
        label_to_cols: Dict[str, List[str]],
    ) -> Dict[str, np.ndarray]:
        """Pre-compute label → integer index arrays for fast subsetting.

        Called once before the greedy loop so that per-candidate subsetting
        uses cheap integer indexing instead of repeated string set membership.
        """
        var_names = list(adata.var_names)
        name_to_idx = {v: i for i, v in enumerate(var_names)}
        label_to_idx: Dict[str, np.ndarray] = {}
        for label, cols in label_to_cols.items():
            indices = [name_to_idx[c] for c in cols if c in name_to_idx]
            label_to_idx[label] = np.array(indices, dtype=np.intp)
        return label_to_idx

    def _reduce_features_pca(
        self,
        adata_guide: ad.AnnData,
        adata_gene: ad.AnnData,
        label_to_cols: Dict[str, List[str]],
    ) -> Tuple[ad.AnnData, ad.AnnData, Dict[str, List[str]]]:
        """
        Combined PCA reduction across all features.

        Fits PCA on the full concatenated feature space (all channels together)
        and keeps the minimum number of components to reach
        ``self.variance_threshold`` cumulative explained variance.

        This is used for the baseline computation.  Backward/forward elimination
        steps redo PCA on each candidate subset independently via
        ``_pca_reduce_combined``.

        Returns
        -------
        new_adata_guide, new_adata_gene : AnnData
            PCA-reduced AnnData objects.
        label_to_cols : dict
            Original label_to_cols (unchanged — elimination steps use raw features).
        """
        reduced_guide, reduced_gene, n_keep, explained = self._pca_reduce_combined(
            adata_guide, adata_gene, label="baseline"
        )

        n_original = adata_guide.n_vars
        logger.info(
            f"  Combined PCA: {n_original} → {n_keep} PCs "
            f"({explained:.1%} variance explained)"
        )

        # Save reduction report
        report_df = pd.DataFrame([{
            "original_features": n_original,
            "kept_components": n_keep,
            "explained_variance": explained,
        }])
        report_path = self.output_dir / "pca_reduction_report.csv"
        report_df.to_csv(report_path, index=False)

        # Return original label_to_cols — elimination steps need raw column mapping
        return reduced_guide, reduced_gene, label_to_cols

    # -------------------------------------------------------------------------
    # Step 5: mAP battery
    # -------------------------------------------------------------------------

    # Mapping from metric name to result dict keys
    _METRIC_KEYS = {
        "activity":        {"auc": "activity_auc",  "ratio": "active_ratio"},
        "distinctiveness": {"auc": "distinct_auc",   "ratio": "distinctive_ratio"},
        "corum":           {"auc": "corum_auc",      "ratio": "corum_ratio"},
        "chad":            {"auc": "chad_auc",       "ratio": "chad_ratio"},
    }

    def _get_metric_score(self, result_dict: Dict) -> float:
        """Return the AUC score for the configured metric (used for greedy decisions)."""
        return result_dict[self._METRIC_KEYS[self.metric]["auc"]]

    def _get_metric_ratio(self, result_dict: Dict) -> float:
        """Return the ratio score for the configured metric (used for plotting)."""
        return result_dict[self._METRIC_KEYS[self.metric]["ratio"]]

    # Reduced null_size for lightweight (greedy candidate) evaluation.
    # 10K samples gives ~1% p-value precision — sufficient for ranking.
    _LIGHTWEIGHT_NULL_SIZE = 10_000
    # Moderate null_size for intermediate step re-evaluation (stored results).
    # 100K gives ~0.3% precision — more than enough for per-step CSVs.
    # Only the baseline uses the full 1M.
    _REEVAL_NULL_SIZE = 100_000

    def _run_map_battery(
        self,
        adata_guide: ad.AnnData,
        adata_gene: ad.AnnData,
        run_label: str,
        lightweight: bool = False,
        null_size: Optional[int] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Run the target mAP metric (+ activity as dependency).

        Only computes the metric specified by ``self.metric``.  Activity is
        always computed first since distinctiveness/CORUM/CHAD depend on it.

        Parameters
        ----------
        lightweight : bool
            When True (used during greedy candidate evaluation), uses reduced
            null_size (10K vs 1M) for ~100x faster p-value estimation.
        null_size : int, optional
            Explicit null_size override. If provided, takes precedence over
            the lightweight flag. Allows intermediate re-evaluations to use
            a moderate value (100K) instead of the full 1M.

        Returns dict with results or None on failure.
        """
        try:
            t0 = time.time()
            metric = self.metric
            if null_size is not None:
                ns = null_size
            else:
                ns = self._LIGHTWEIGHT_NULL_SIZE if lightweight else 1_000_000

            # 1. Activity (always needed — dependency for all other metrics)
            activity_map, active_ratio = phenotypic_activity_assesment(
                adata_guide, plot_results=False, null_size=ns,
            )
            activity_auc = compute_auc_score(activity_map)
            logger.info(
                f"    [{run_label}] Activity ({time.time()-t0:.1f}s): "
                f"{active_ratio:.2%} active, AUC={activity_auc:.4f}"
            )

            # 2. Distinctiveness (guide level)
            distinct_map, distinctive_ratio, distinct_auc = None, 0.0, 0.0
            if metric == "distinctiveness":
                t1 = time.time()
                distinct_map, distinctive_ratio = phenotypic_distinctivness(
                    adata_guide, activity_map, plot_results=False, null_size=ns,
                )
                distinct_auc = compute_auc_score(distinct_map)
                logger.info(
                    f"    [{run_label}] Distinctiveness ({time.time()-t1:.1f}s): "
                    f"{distinctive_ratio:.2%}, AUC={distinct_auc:.4f}"
                )

            # 3. CORUM consistency (gene level)
            corum_map, corum_ratio, corum_auc = None, 0.0, 0.0
            if metric == "corum":
                t2 = time.time()
                corum_map, corum_ratio = phenotypic_consistency_corum(
                    adata_gene, activity_map, plot_results=False, null_size=ns,
                    cache_similarity=True,
                )
                corum_auc = compute_auc_score(corum_map)
                logger.info(
                    f"    [{run_label}] CORUM ({time.time()-t2:.1f}s): "
                    f"{corum_ratio:.2%}, AUC={corum_auc:.4f}"
                )

            # 4. CHAD consistency (gene level)
            chad_map, chad_ratio, chad_auc = None, 0.0, 0.0
            if metric == "chad":
                t3 = time.time()
                chad_map, chad_ratio = phenotypic_consistency_manual_annotation(
                    adata_gene, activity_map, plot_results=False, null_size=ns,
                    cache_similarity=True,
                )
                chad_auc = compute_auc_score(chad_map)
                logger.info(
                    f"    [{run_label}] CHAD ({time.time()-t3:.1f}s): "
                    f"{chad_ratio:.2%}, AUC={chad_auc:.4f}"
                )

            logger.info(f"    [{run_label}] Total: {time.time()-t0:.1f}s")

            return {
                "label": run_label,
                "activity_map": activity_map,
                "active_ratio": active_ratio,
                "activity_auc": activity_auc,
                "distinct_map": distinct_map,
                "distinctive_ratio": distinctive_ratio,
                "distinct_auc": distinct_auc,
                "corum_map": corum_map,
                "corum_ratio": corum_ratio,
                "corum_auc": corum_auc,
                "chad_map": chad_map,
                "chad_ratio": chad_ratio,
                "chad_auc": chad_auc,
            }

        except Exception as e:
            logger.error(f"mAP battery failed for '{run_label}': {e}")
            import traceback
            traceback.print_exc()
            return None

    def _save_baseline(self, baseline: Dict, result: StageResult) -> None:
        """Save baseline mAP results to CSV files (only for computed metrics)."""
        baseline_dir = self.output_dir / "baseline"
        baseline_dir.mkdir(parents=True, exist_ok=True)

        for key in ["activity_map", "distinct_map", "corum_map", "chad_map"]:
            if baseline[key] is not None:
                csv_path = baseline_dir / f"{key}.csv"
                baseline[key].to_csv(csv_path, index=False)
                result.add_file(csv_path)

        # Summary (only include computed metrics)
        summary = {
            "metric": ["activity", "distinctiveness", "corum", "chad"],
            "ratio": [
                baseline["active_ratio"],
                baseline["distinctive_ratio"],
                baseline["corum_ratio"],
                baseline["chad_ratio"],
            ],
            "auc": [
                baseline["activity_auc"],
                baseline["distinct_auc"],
                baseline["corum_auc"],
                baseline["chad_auc"],
            ],
        }
        summary_path = baseline_dir / "baseline_summary.csv"
        pd.DataFrame(summary).to_csv(summary_path, index=False)
        result.add_file(summary_path)

    # -------------------------------------------------------------------------
    # Step 6: Greedy backward elimination (cumulative knock-out)
    # -------------------------------------------------------------------------

    def _greedy_backward_elimination(
        self,
        adata_guide: ad.AnnData,
        adata_gene: ad.AnnData,
        label_to_cols: Dict[str, List[str]],
        baseline: Dict,
        norm_method: str,
        result: StageResult,
    ) -> List[Dict[str, Any]]:
        """
        Greedy backward elimination: cumulatively remove the least important channel.

        Starting from all channels, at each step try removing each remaining
        channel and permanently remove the one whose loss causes the least mAP
        damage.  This produces an elimination ordering from least → most important
        and a cumulative degradation curve.

        Returns list of ablation result dicts (one per elimination step), ordered
        from first-removed (least important) to last-removed (most important).
        """
        remaining = set(label_to_cols.keys())
        elimination_order: List[str] = []
        ablation_results: List[Dict[str, Any]] = []

        n_labels = len(remaining)
        n_raw_features = adata_guide.n_vars
        t_start = time.time()

        # Track the "previous step" result — starts as the full baseline
        prev_result = baseline

        steps: List[Dict[str, Any]] = []
        # Step 0: full baseline
        steps.append({
            "step": 0,
            "label_removed": "(none)",
            "n_channels_remaining": n_labels,
            "n_features_remaining": n_raw_features,
            "metric_auc": self._get_metric_score(baseline),
            "metric_ratio": self._get_metric_ratio(baseline),
        })

        # Use joblib with loky backend — spawns separate processes so no
        # OpenMP/BLAS thread deadlocks. Cap at 8 to limit memory usage.
        from joblib import Parallel, delayed
        from ops_utils.hpc.resource_manager import get_optimal_workers
        n_workers = min(get_optimal_workers(use_gpu=False, model_ram_gb=0.1, data_ram_gb=4.0, verbose=False), 8)

        # Pre-compute integer index arrays (optimization #3)
        label_to_idx_guide = self._precompute_label_indices(adata_guide, label_to_cols)
        label_to_idx_gene = self._precompute_label_indices(adata_gene, label_to_cols)

        pool = None  # kept for legacy branch guards
        logger.info(f"  Backward elimination across {n_labels} channel groups (joblib loky workers={n_workers})...")

        try:
            for step_num in range(1, n_labels):
                # At each step, try removing each remaining channel from the current set
                best_label = None
                best_score = -np.inf
                best_result_dict = None

                n_candidates = len(remaining)
                logger.info(f"  Step {step_num}/{n_labels-1}: testing {n_candidates} candidates...")

                candidates_sorted = sorted(remaining)

                if pool is not None and n_candidates > 1:
                    # Process pool path (fork-based, lightweight mode)
                    args_list = [
                        (c, list(remaining - {c}), f"bwd_without_{c}")
                        for c in candidates_sorted
                    ]
                    results = pool.map(_eval_candidate_process, args_list)
                    for j, (candidate, score, ratio) in enumerate(results, 1):
                        logger.info(f"    [{step_num}/{n_labels-1}] completed {j}/{n_candidates}: {candidate}")
                        if score is None:
                            continue
                        if score > best_score:
                            best_score = score
                            best_label = candidate
                else:
                    # joblib parallel evaluation (loky backend — no fork deadlocks)
                    def _eval_bwd_candidate(candidate):
                        """Evaluate removing one candidate channel."""
                        keep_labels = remaining - {candidate}
                        subset_guide, subset_gene = self._subset_channels(
                            adata_guide, adata_gene, label_to_cols, keep_labels
                        )
                        if subset_guide.n_vars == 0:
                            return candidate, None
                        if self.fast_mode and self.pca_optimized_dir is None:
                            subset_guide, subset_gene, _, _ = self._pca_reduce_combined(
                                subset_guide, subset_gene, label=f"bwd_without_{candidate}"
                            )
                        r = self._run_map_battery(
                            subset_guide, subset_gene, f"bwd_without_{candidate}",
                            lightweight=True,
                        )
                        return candidate, r

                    if n_workers > 1 and n_candidates > 1:
                        results_list = Parallel(n_jobs=n_workers, backend="loky")(
                            delayed(_eval_bwd_candidate)(c) for c in candidates_sorted
                        )
                        for j, (candidate, r) in enumerate(results_list, 1):
                            logger.info(f"    [{step_num}/{n_labels-1}] completed {j}/{n_candidates}: {candidate}")
                            if r is None:
                                continue
                            score = self._get_metric_score(r)
                            if score > best_score:
                                best_score = score
                                best_label = candidate
                                best_result_dict = r
                    else:
                        for j, candidate in enumerate(candidates_sorted, 1):
                            logger.info(f"    [{step_num}/{n_labels-1}] candidate {j}/{n_candidates}: {candidate}")
                            candidate, r = _eval_bwd_candidate(candidate)
                            if r is None:
                                continue
                            score = self._get_metric_score(r)
                            if score > best_score:
                                best_score = score
                                best_label = candidate
                                best_result_dict = r

                if best_label is None:
                    logger.warning(f"  Step {step_num}: no valid candidate, stopping")
                    break

                # Permanently remove the least impactful channel
                remaining.discard(best_label)
                elimination_order.append(best_label)

                # Re-evaluate winner with moderate precision (null_size=100K)
                # for per-step CSV storage. Only the baseline uses full 1M.
                keep_labels = set(remaining)
                subset_guide, subset_gene = self._subset_channels(
                    adata_guide, adata_gene, label_to_cols, keep_labels
                )
                if self.fast_mode and self.pca_optimized_dir is None:
                    subset_guide, subset_gene, _, _ = self._pca_reduce_combined(
                        subset_guide, subset_gene, label=f"bwd_without_{best_label}_full"
                    )
                best_result_dict = self._run_map_battery(
                    subset_guide, subset_gene, f"bwd_without_{best_label}",
                    null_size=self._REEVAL_NULL_SIZE,
                )
                if best_result_dict is None:
                    best_result_dict = {"activity_map": prev_result["activity_map"],
                                        **{k: 0.0 for k in ["active_ratio", "distinctive_ratio",
                                            "corum_ratio", "chad_ratio", "activity_auc", "distinct_auc",
                                            "corum_auc", "chad_auc"]}}

                # Compute per-perturbation deltas (against previous step, not baseline)
                per_pert_delta = self._compute_per_perturbation_delta(
                    prev_result, best_result_dict, best_label
                )

                n_feats_remaining = sum(len(label_to_cols[l]) for l in remaining)

                ablation_results.append({
                    "label": best_label,
                    "step": step_num,
                    "n_features_removed": len(label_to_cols[best_label]),
                    "n_features_remaining": n_feats_remaining,
                    "n_channels_remaining": len(remaining),
                    "active_ratio": best_result_dict["active_ratio"],
                    "distinctive_ratio": best_result_dict["distinctive_ratio"],
                    "corum_ratio": best_result_dict["corum_ratio"],
                    "chad_ratio": best_result_dict["chad_ratio"],
                    "activity_auc": best_result_dict["activity_auc"],
                    "distinct_auc": best_result_dict["distinct_auc"],
                    "corum_auc": best_result_dict["corum_auc"],
                    "chad_auc": best_result_dict["chad_auc"],
                    "per_pert_delta": per_pert_delta,
                    "full_result": best_result_dict,
                })

                steps.append({
                    "step": step_num,
                    "label_removed": best_label,
                    "n_channels_remaining": len(remaining),
                    "n_features_remaining": n_feats_remaining,
                    "metric_auc": self._get_metric_score(best_result_dict),
                    "metric_ratio": self._get_metric_ratio(best_result_dict),
                })

                elapsed = time.time() - t_start
                avg = elapsed / step_num
                eta = avg * (n_labels - 1 - step_num)
                logger.info(
                    f"  Step {step_num}/{n_labels-1}: removed '{best_label}' "
                    f"({len(remaining)} channels left) → "
                    f"{self.metric} AUC={self._get_metric_score(best_result_dict):.4f}, "
                    f"ratio={self._get_metric_ratio(best_result_dict):.2%} "
                    f"({avg:.1f}s/step, ETA {eta:.0f}s)"
                )

                prev_result = best_result_dict

        finally:
            # Clean up process pool and shared state
            if pool is not None:
                pool.close()
                pool.join()
            _POOL_SHARED.clear()

        # Save elimination order table
        steps_df = pd.DataFrame(steps)
        steps_df["metric"] = self.metric
        steps_df["baseline_auc"] = self._get_metric_score(baseline)
        steps_df["baseline_ratio"] = self._get_metric_ratio(baseline)

        csv_path = self.output_dir / "backward_elimination_order.csv"
        steps_df.to_csv(csv_path, index=False)
        result.add_file(csv_path)
        logger.info(f"  Saved: {csv_path}")

        # Generate degradation curve plot
        self._plot_backward_elimination(steps_df, baseline, norm_method, result)

        return ablation_results

    @staticmethod
    def _wrap_label(text: str, max_chars: int = 18) -> str:
        """Wrap a long label with newlines so it stays readable horizontally."""
        if len(text) <= max_chars:
            return text
        # Try splitting on comma first (e.g. "lipid droplet, PLIN2")
        if ", " in text:
            parts = text.split(", ", 1)
            return parts[0] + ",\n" + parts[1]
        # Fall back to splitting on space nearest to midpoint
        mid = len(text) // 2
        best = text.rfind(" ", 0, mid + 5)
        if best == -1:
            best = text.find(" ", mid)
        if best == -1:
            return text  # no space found, leave as-is
        return text[:best] + "\n" + text[best + 1:]

    def _plot_backward_elimination(
        self,
        steps_df: pd.DataFrame,
        baseline: Dict,
        norm_method: str,
        result: StageResult,
    ) -> None:
        """Cumulative degradation curve as channels are removed.

        Two subplots on the same canvas: AUC (scoring metric) and ratio (% significant).
        X-axis tick labels show which channel was removed at each step.
        """
        metric_label = self.metric.capitalize()
        bl_auc = self._get_metric_score(baseline)
        bl_ratio = self._get_metric_ratio(baseline)

        n_steps = len(steps_df)
        fig_width = max(14, n_steps * 1.2)
        fig, (ax_auc, ax_ratio) = plt.subplots(1, 2, figsize=(fig_width, 8))

        # Build x-tick labels: step 0 = "all", then channel removed at each step
        tick_labels = []
        for _, row in steps_df.iterrows():
            if row["step"] == 0:
                tick_labels.append("all channels")
            else:
                tick_labels.append(str(row["label_removed"]))

        panels = [
            (ax_auc,   "metric_auc",   f"{metric_label} AUC",              bl_auc),
            (ax_ratio, "metric_ratio", f"{metric_label} Ratio (p<0.05)",   bl_ratio),
        ]

        for ax, col, title, bl_val in panels:
            step_vals = steps_df["step"].values
            values = steps_df[col].values

            ax.plot(step_vals, values, "o-", color="#d32f2f", linewidth=2,
                    markersize=8, zorder=3)

            # Baseline reference
            ax.axhline(bl_val, color="#1976d2", linestyle="--", linewidth=1.5,
                        label=f"Full baseline ({bl_val:.3f})")

            # 95% threshold band
            threshold_95 = bl_val * 0.95
            ax.axhspan(threshold_95, bl_val, color="#1976d2", alpha=0.08)
            ax.axhline(threshold_95, color="#1976d2", linestyle=":", linewidth=1,
                        alpha=0.5, label=f"95% threshold ({threshold_95:.3f})")

            # Find where score drops below 95%
            below_95 = np.where(values < threshold_95)[0]
            if len(below_95) > 0:
                first_drop = below_95[0]
                ax.axvline(step_vals[first_drop], color="#388e3c", linestyle="--",
                           linewidth=1.5, alpha=0.7,
                           label=f"Below 95% at step {step_vals[first_drop]}")

            # Use channel names as x-tick labels (diagonal for readability)
            ax.set_xticks(step_vals)
            ax.set_xticklabels(tick_labels, fontsize=10, ha="right", rotation=45)

            ax.set_xlabel("Channel Removed (cumulative, least important first)", fontsize=11)
            ax.set_ylabel(title, fontsize=12)
            ax.set_title(f"Backward Elimination: {title}", fontsize=13, fontweight="bold")
            ax.legend(fontsize=9, loc="lower left")
            ax.set_xlim(-0.5, step_vals[-1] + 0.5)
            ax.grid(True, alpha=0.3)

        fig.suptitle(
            f"Greedy Backward Elimination — {metric_label}\n"
            f"(normalization: {norm_method}, least important removed first, scored by AUC)",
            fontsize=14, fontweight="bold", y=1.02,
        )
        plt.tight_layout()

        path = save_figure(fig, self.output_dir / "backward_elimination_curve.png")
        result.add_file(path)
        logger.info(f"  Saved: {path}")

    def _compute_per_perturbation_delta(
        self,
        baseline: Dict,
        ablated: Dict,
        label: str,
    ) -> pd.DataFrame:
        """
        Compute per-perturbation mAP delta between baseline and ablated.

        Returns DataFrame with columns: perturbation, baseline_mAP, ablated_mAP, delta, lost_significance.
        """
        b_act = baseline["activity_map"].set_index("perturbation")
        a_act = ablated["activity_map"].set_index("perturbation")

        common = b_act.index.intersection(a_act.index)
        delta_df = pd.DataFrame({
            "perturbation": common,
            "baseline_mAP": b_act.loc[common, "mean_average_precision"].values,
            "baseline_significant": b_act.loc[common, "below_corrected_p"].values,
            "ablated_mAP": a_act.loc[common, "mean_average_precision"].values,
            "ablated_significant": a_act.loc[common, "below_corrected_p"].values,
        })
        delta_df["delta_mAP"] = delta_df["baseline_mAP"] - delta_df["ablated_mAP"]
        delta_df["lost_significance"] = (
            delta_df["baseline_significant"] & ~delta_df["ablated_significant"]
        )
        delta_df["ablated_label"] = label

        return delta_df

    # -------------------------------------------------------------------------
    # Step 7: Attribution analysis and outputs
    # -------------------------------------------------------------------------

    def _compute_and_save_attribution(
        self,
        baseline: Dict,
        ablation_results: List[Dict],
        label_to_cols: Dict[str, List[str]],
        result: StageResult,
    ) -> None:
        """Compute attribution matrices and generate all outputs."""
        if not ablation_results:
            result.add_error("No ablation results to analyze")
            return

        # 1. Summary table
        summary_df = self._build_summary_table(baseline, ablation_results)
        summary_path = self.output_dir / "organelle_attribution_summary.csv"
        summary_df.to_csv(summary_path, index=False)
        result.add_file(summary_path)
        logger.info(f"  Saved summary: {summary_path}")

        # 2. Per-perturbation attribution matrix
        attr_matrix = self._build_attribution_matrix(ablation_results)
        if attr_matrix is not None:
            matrix_path = self.output_dir / "perturbation_attribution_matrix.csv"
            attr_matrix.to_csv(matrix_path)
            result.add_file(matrix_path)
            logger.info(f"  Saved attribution matrix: {matrix_path}")

        # 3. Perturbation-organelle dependency table
        dep_df = self._build_dependency_table(ablation_results)
        if dep_df is not None:
            dep_path = self.output_dir / "perturbation_organelle_dependency.csv"
            dep_df.to_csv(dep_path, index=False)
            result.add_file(dep_path)
            logger.info(f"  Saved dependency table: {dep_path}")

        # 4. Generate plots
        self._generate_plots(summary_df, attr_matrix, ablation_results, result)

    def _build_summary_table(
        self, baseline: Dict, ablation_results: List[Dict]
    ) -> pd.DataFrame:
        """Build summary table: one row per channel label with baseline/ablated/delta."""
        rows = []
        for abl in ablation_results:
            row = {
                "channel_label": abl["label"],
                "n_features_removed": abl["n_features_removed"],
                "n_features_remaining": abl["n_features_remaining"],
                # Ablated scores
                "activity_ratio": abl["active_ratio"],
                "distinct_ratio": abl["distinctive_ratio"],
                "corum_ratio": abl["corum_ratio"],
                "chad_ratio": abl["chad_ratio"],
                "activity_auc": abl["activity_auc"],
                "distinct_auc": abl["distinct_auc"],
                "corum_auc": abl["corum_auc"],
                "chad_auc": abl["chad_auc"],
                # Deltas (positive = removing hurts = organelle is important)
                "delta_activity_ratio": baseline["active_ratio"] - abl["active_ratio"],
                "delta_distinct_ratio": baseline["distinctive_ratio"] - abl["distinctive_ratio"],
                "delta_corum_ratio": baseline["corum_ratio"] - abl["corum_ratio"],
                "delta_chad_ratio": baseline["chad_ratio"] - abl["chad_ratio"],
                "delta_activity_auc": baseline["activity_auc"] - abl["activity_auc"],
                "delta_distinct_auc": baseline["distinct_auc"] - abl["distinct_auc"],
                "delta_corum_auc": baseline["corum_auc"] - abl["corum_auc"],
                "delta_chad_auc": baseline["chad_auc"] - abl["chad_auc"],
            }
            rows.append(row)

        df = pd.DataFrame(rows)
        # Sort by average delta across ratio metrics (most important first)
        delta_cols = ["delta_activity_ratio", "delta_distinct_ratio",
                      "delta_corum_ratio", "delta_chad_ratio"]
        df["avg_delta"] = df[delta_cols].mean(axis=1)
        df = df.sort_values("avg_delta", ascending=False).reset_index(drop=True)
        return df

    def _build_attribution_matrix(
        self, ablation_results: List[Dict]
    ) -> Optional[pd.DataFrame]:
        """
        Build perturbation x organelle matrix of activity mAP deltas.

        Each cell = baseline_mAP - ablated_mAP for that perturbation when
        that organelle's features are removed.
        """
        all_deltas = []
        for abl in ablation_results:
            delta_df = abl["per_pert_delta"]
            if delta_df is not None and len(delta_df) > 0:
                all_deltas.append(
                    delta_df[["perturbation", "delta_mAP"]].rename(
                        columns={"delta_mAP": abl["label"]}
                    ).set_index("perturbation")
                )

        if not all_deltas:
            return None

        matrix = pd.concat(all_deltas, axis=1)
        matrix = matrix.fillna(0)
        return matrix

    def _build_dependency_table(
        self, ablation_results: List[Dict]
    ) -> Optional[pd.DataFrame]:
        """
        Build table of perturbations that lost significance for each channel label.
        """
        rows = []
        for abl in ablation_results:
            delta_df = abl["per_pert_delta"]
            if delta_df is None:
                continue
            lost = delta_df[delta_df["lost_significance"]]
            for _, row in lost.iterrows():
                rows.append({
                    "perturbation": row["perturbation"],
                    "channel_label": abl["label"],
                    "baseline_mAP": row["baseline_mAP"],
                    "ablated_mAP": row["ablated_mAP"],
                    "delta_mAP": row["delta_mAP"],
                })

        if not rows:
            return None
        return pd.DataFrame(rows).sort_values("delta_mAP", ascending=False)

    # -------------------------------------------------------------------------
    # Plots
    # -------------------------------------------------------------------------

    def _generate_plots(
        self,
        summary_df: pd.DataFrame,
        attr_matrix: Optional[pd.DataFrame],
        ablation_results: List[Dict],
        result: StageResult,
    ) -> None:
        """Generate all attribution plots."""
        # 1. Delta heatmap (organelle x metric)
        self._plot_delta_heatmap(summary_df, result)

        # 2. Ranked bar chart
        self._plot_ranked_importance(summary_df, result)

        # 3. Clustered attribution matrix
        if attr_matrix is not None and len(attr_matrix) > 5:
            self._plot_clustered_attribution(attr_matrix, result)

        # 4. Waterfall plots per organelle
        self._plot_waterfall(ablation_results, result)

    def _plot_delta_heatmap(self, summary_df: pd.DataFrame, result: StageResult) -> None:
        """Heatmap of delta scores: organelle x metric."""
        delta_cols = [
            "delta_activity_ratio", "delta_distinct_ratio",
            "delta_corum_ratio", "delta_chad_ratio",
        ]
        plot_data = summary_df.set_index("channel_label")[delta_cols].copy()
        plot_data.columns = ["Activity", "Distinctiveness", "CORUM", "CHAD"]

        fig, ax = plt.subplots(figsize=(10, max(6, len(plot_data) * 0.4)))
        sns.heatmap(
            plot_data,
            cmap="RdYlBu_r",
            center=0,
            annot=True,
            fmt=".3f",
            ax=ax,
            cbar_kws={"label": "Delta (baseline - ablated)"},
        )
        ax.set_title("Organelle Attribution: mAP Delta When Removed\n(positive = important)")
        ax.set_xlabel("mAP Metric")
        ax.set_ylabel("Channel Label (removed)")
        plt.xticks(rotation=0)
        plt.yticks(rotation=0)
        plt.tight_layout()

        path = save_figure(fig, self.output_dir / "delta_heatmap.png")
        result.add_file(path)

    def _plot_ranked_importance(self, summary_df: pd.DataFrame, result: StageResult) -> None:
        """Bar chart of organelle importance ranked by average delta."""
        fig, ax = plt.subplots(figsize=(max(10, len(summary_df) * 0.7), 6))

        labels = summary_df["channel_label"]
        avg_deltas = summary_df["avg_delta"]

        colors = ["#d32f2f" if d > 0 else "#1976d2" for d in avg_deltas]
        bars = ax.bar(range(len(labels)), avg_deltas, color=colors)

        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.set_ylabel("Average mAP Delta (baseline - ablated)")
        ax.set_title("Organelle Importance Ranking\n(higher = more important for phenotypic discrimination)")
        ax.axhline(0, color="black", linewidth=0.5)

        # Value labels
        for bar, val in zip(bars, avg_deltas):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.002 if val >= 0 else bar.get_height() - 0.005,
                f"{val:.3f}",
                ha="center", va="bottom" if val >= 0 else "top",
                fontsize=8,
            )

        plt.tight_layout()
        path = save_figure(fig, self.output_dir / "ranked_importance.png")
        result.add_file(path)

    def _plot_clustered_attribution(
        self, attr_matrix: pd.DataFrame, result: StageResult
    ) -> None:
        """Clustered heatmap of perturbation x organelle attribution."""
        try:
            # Limit to top perturbations by max delta
            max_delta = attr_matrix.abs().max(axis=1)
            top_n = min(80, len(attr_matrix))
            top_perts = max_delta.nlargest(top_n).index
            plot_data = attr_matrix.loc[top_perts]

            g = sns.clustermap(
                plot_data,
                cmap="RdBu_r",
                center=0,
                figsize=(max(12, len(plot_data.columns) * 1.0),
                         max(10, top_n * 0.15)),
                dendrogram_ratio=(0.1, 0.15),
                cbar_pos=(0.02, 0.8, 0.03, 0.15),
            )
            g.ax_heatmap.set_xlabel("Channel Label (removed)")
            g.ax_heatmap.set_ylabel("Perturbation")
            g.fig.suptitle(
                f"Perturbation-Organelle Attribution Matrix (top {top_n} perturbations)",
                y=1.02,
            )

            path = self.output_dir / "clustered_attribution_matrix.png"
            g.fig.savefig(path, dpi=150, bbox_inches="tight")
            plt.close(g.fig)
            result.add_file(path)
            logger.info(f"  Saved clustered attribution: {path}")

        except Exception as e:
            logger.warning(f"Could not generate clustered attribution heatmap: {e}")

    def _plot_waterfall(
        self, ablation_results: List[Dict], result: StageResult
    ) -> None:
        """Waterfall plots: per-organelle, which perturbations lost the most mAP."""
        waterfall_dir = self.output_dir / "waterfall_plots"
        waterfall_dir.mkdir(parents=True, exist_ok=True)

        for abl in ablation_results:
            delta_df = abl["per_pert_delta"]
            if delta_df is None or len(delta_df) == 0:
                continue

            # Only show perturbations with notable positive delta
            notable = delta_df[delta_df["delta_mAP"] > 0.01].sort_values(
                "delta_mAP", ascending=False
            )
            if len(notable) < 2:
                continue

            top_n = min(30, len(notable))
            plot_df = notable.head(top_n)

            fig, ax = plt.subplots(figsize=(10, max(6, top_n * 0.3)))
            colors = ["#d32f2f" if lost else "#ff9800"
                       for lost in plot_df["lost_significance"]]
            ax.barh(range(top_n), plot_df["delta_mAP"].values, color=colors)
            ax.set_yticks(range(top_n))
            ax.set_yticklabels(plot_df["perturbation"].values)
            ax.invert_yaxis()
            ax.set_xlabel("mAP Delta (baseline - ablated)")
            ax.set_title(f"Perturbations Most Affected by Removing: {abl['label']}\n"
                          f"(red = lost significance)")
            plt.tight_layout()

            safe_name = abl["label"].replace("/", "_").replace(" ", "_").replace(",", "")
            path = save_figure(fig, waterfall_dir / f"waterfall_{safe_name}.png")
            result.add_file(path)

    # -------------------------------------------------------------------------
    # Step 8: Greedy forward selection — minimal set for full coverage
    # -------------------------------------------------------------------------

    def _greedy_minimal_set(
        self,
        adata_guide: ad.AnnData,
        adata_gene: ad.AnnData,
        label_to_cols: Dict[str, List[str]],
        baseline: Dict,
        norm_method: str,
        result: StageResult,
    ) -> None:
        """
        Greedy forward selection to find the minimal organelle set for full mAP coverage.

        Starting from an empty feature set, iteratively adds the organelle group
        whose features yield the largest mAP gain.  Stops when mAP reaches the
        full baseline.  This reveals which organelles are essential vs redundant.

        Outputs:
        - ``minimal_coverage_order.csv`` — step-by-step table
        - ``minimal_coverage_curve.png`` — cumulative gain plot
        """
        labels = sorted(label_to_cols.keys())
        remaining = set(labels)
        selected_order: List[str] = []

        # Track scores at each step
        steps: List[Dict[str, Any]] = []

        # Step 0: no features (all zeros → everything random)
        steps.append({
            "step": 0,
            "label_added": "(none)",
            "n_features": 0,
            "metric_auc": 0.0,
            "metric_ratio": 0.0,
        })

        # Enforce Phase as first channel if requested
        if self.enforce_phase:
            phase_label = None
            for label in labels:
                if label.lower() == "phase" or label.lower().startswith("phase,"):
                    phase_label = label
                    break
            if phase_label and phase_label in remaining:
                logger.info(f"  --enforce-phase: forcing '{phase_label}' as step 1")
                selected_order.append(phase_label)
                remaining.discard(phase_label)
                # Evaluate Phase to record its score
                subset_guide, subset_gene = self._subset_channels(
                    adata_guide, adata_gene, label_to_cols, set(selected_order)
                )
                if self.fast_mode and self.pca_optimized_dir is None:
                    subset_guide, subset_gene, _, _ = self._pca_reduce_combined(
                        subset_guide, subset_gene, label="fwd_phase_enforced"
                    )
                r = self._run_map_battery(subset_guide, subset_gene, "fwd_phase_enforced", lightweight=True)
                n_feats = sum(len(label_to_cols[l]) for l in selected_order)
                steps.append({
                    "step": 1,
                    "label_added": phase_label,
                    "n_features": n_feats,
                    "metric_auc": self._get_metric_score(r) if r else 0.0,
                    "metric_ratio": self._get_metric_ratio(r) if r else 0.0,
                })
                logger.info(
                    f"  Step 1 (enforced): +'{phase_label}' → "
                    f"AUC={steps[-1]['metric_auc']:.4f}, "
                    f"ratio={steps[-1]['metric_ratio']:.2%}, "
                    f"features={n_feats}"
                )
            else:
                logger.warning(f"  --enforce-phase: no 'Phase' label found among {labels}")

        from joblib import Parallel, delayed
        from ops_utils.hpc.resource_manager import get_optimal_workers
        n_workers = min(get_optimal_workers(use_gpu=False, model_ram_gb=0.1, data_ram_gb=4.0, verbose=False), 8)
        n_labels = len(labels)

        # Pre-compute integer index arrays (optimization #3)
        label_to_idx_guide = self._precompute_label_indices(adata_guide, label_to_cols)
        label_to_idx_gene = self._precompute_label_indices(adata_gene, label_to_cols)

        pool = None  # kept for legacy branch guards
        logger.info(f"  Forward selection across {n_labels} organelle groups (joblib loky workers={n_workers})...")

        start_step = len(selected_order) + 1
        try:
            for step_num in range(start_step, n_labels + 1):
                best_label = None
                best_score = -1.0
                best_ratio = 0.0

                n_candidates = len(remaining)
                logger.info(f"  Step {step_num}/{n_labels}: testing {n_candidates} candidates...")

                candidates_sorted = sorted(remaining)

                if pool is not None and n_candidates > 1:
                    # Process pool path (fork-based, lightweight mode)
                    args_list = [
                        (c, list(set(selected_order + [c])), f"fwd_{c}")
                        for c in candidates_sorted
                    ]
                    results = pool.map(_eval_candidate_process, args_list)
                    for j, (candidate, score, ratio) in enumerate(results, 1):
                        logger.info(f"    [{step_num}/{n_labels}] completed {j}/{n_candidates}: {candidate}")
                        if score is None:
                            continue
                        if score > best_score:
                            best_score = score
                            best_ratio = ratio
                            best_label = candidate
                else:
                    # joblib parallel evaluation (loky backend — no fork deadlocks)
                    def _eval_fwd_candidate(candidate):
                        """Evaluate adding one candidate channel."""
                        include_labels = set(selected_order + [candidate])
                        subset_guide, subset_gene = self._subset_channels(
                            adata_guide, adata_gene, label_to_cols, include_labels
                        )
                        if subset_guide.n_vars == 0:
                            return candidate, None
                        if self.fast_mode and self.pca_optimized_dir is None:
                            subset_guide, subset_gene, _, _ = self._pca_reduce_combined(
                                subset_guide, subset_gene, label=f"fwd_{candidate}"
                            )
                        r = self._run_map_battery(
                            subset_guide, subset_gene, f"fwd_{candidate}",
                            lightweight=True,
                        )
                        return candidate, r

                    if n_workers > 1 and n_candidates > 1:
                        results_list = Parallel(n_jobs=n_workers, backend="loky")(
                            delayed(_eval_fwd_candidate)(c) for c in candidates_sorted
                        )
                        for j, (candidate, r) in enumerate(results_list, 1):
                            logger.info(f"    [{step_num}/{n_labels}] completed {j}/{n_candidates}: {candidate}")
                            if r is None:
                                continue
                            score = self._get_metric_score(r)
                            if score > best_score:
                                best_score = score
                                best_ratio = self._get_metric_ratio(r)
                                best_label = candidate
                    else:
                        for j, candidate in enumerate(candidates_sorted, 1):
                            logger.info(f"    [{step_num}/{n_labels}] candidate {j}/{n_candidates}: {candidate}")
                            candidate, r = _eval_fwd_candidate(candidate)
                            if r is None:
                                continue
                            score = self._get_metric_score(r)
                            if score > best_score:
                                best_score = score
                                best_ratio = self._get_metric_ratio(r)
                                best_label = candidate

                if best_label is None:
                    logger.warning(f"  Step {step_num}: no valid candidate, stopping")
                    break

                selected_order.append(best_label)
                remaining.discard(best_label)

                n_feats = sum(len(label_to_cols[l]) for l in selected_order)
                steps.append({
                    "step": step_num,
                    "label_added": best_label,
                    "n_features": n_feats,
                    "metric_auc": best_score,
                    "metric_ratio": best_ratio,
                })
        finally:
            if pool is not None:
                pool.close()
                pool.join()
            _POOL_SHARED.clear()

            logger.info(
                f"  Step {step_num}: +'{best_label}' → "
                f"AUC={best_score:.4f}, "
                f"ratio={best_ratio:.2%}, "
                f"features={n_feats}"
            )

        # Save step table
        steps_df = pd.DataFrame(steps)
        steps_df["metric"] = self.metric
        steps_df["baseline_auc"] = self._get_metric_score(baseline)
        steps_df["baseline_ratio"] = self._get_metric_ratio(baseline)

        csv_path = self.output_dir / "minimal_coverage_order.csv"
        steps_df.to_csv(csv_path, index=False)
        result.add_file(csv_path)
        logger.info(f"  Saved: {csv_path}")

        # Generate cumulative gain plot
        self._plot_minimal_coverage(steps_df, baseline, norm_method, result)

    def _plot_minimal_coverage(
        self,
        steps_df: pd.DataFrame,
        baseline: Dict,
        norm_method: str,
        result: StageResult,
    ) -> None:
        """
        Cumulative gain curve showing how mAP grows as organelles are added.

        Two subplots on the same canvas: AUC (scoring metric) and ratio (% significant).
        """
        metric_label = self.metric.capitalize()
        bl_auc = self._get_metric_score(baseline)
        bl_ratio = self._get_metric_ratio(baseline)

        # Skip step 0 (no channels) — start from first channel added
        plot_df = steps_df[steps_df["step"] > 0].reset_index(drop=True)

        # Build x-tick labels from channel names
        tick_labels = [str(row["label_added"]) for _, row in plot_df.iterrows()]

        n_steps = len(plot_df)
        fig_width = max(14, n_steps * 1.2)
        fig, (ax_auc, ax_ratio) = plt.subplots(1, 2, figsize=(fig_width, 8))

        panels = [
            (ax_auc,   "metric_auc",   f"{metric_label} AUC",              bl_auc),
            (ax_ratio, "metric_ratio", f"{metric_label} Ratio (p<0.05)",   bl_ratio),
        ]

        for ax, col, title, bl_val in panels:
            step_vals = plot_df["step"].values
            values = plot_df[col].values

            # Main curve
            ax.plot(step_vals, values, "o-", color="#1976d2", linewidth=2,
                    markersize=8, zorder=3)

            # Baseline reference
            ax.axhline(bl_val, color="#d32f2f", linestyle="--", linewidth=1.5,
                        label=f"Full baseline ({bl_val:.3f})")

            # 95% threshold band
            threshold_95 = bl_val * 0.95
            ax.axhspan(threshold_95, bl_val, color="#d32f2f", alpha=0.08)
            ax.axhline(threshold_95, color="#d32f2f", linestyle=":", linewidth=1,
                        alpha=0.5, label=f"95% threshold ({threshold_95:.3f})")

            # Find where 95% is first reached
            above_95 = np.where(values >= threshold_95)[0]
            if len(above_95) > 0:
                first_95 = above_95[0]
                ax.axvline(step_vals[first_95], color="#388e3c", linestyle="--",
                           linewidth=1.5, alpha=0.7,
                           label=f"95% reached at step {step_vals[first_95]}")

            # Use channel names as x-tick labels (diagonal for readability)
            ax.set_xticks(step_vals)
            ax.set_xticklabels(tick_labels, fontsize=10, ha="right", rotation=45)

            ax.set_xlabel("Channel Added (cumulative, most important first)", fontsize=11)
            ax.set_ylabel(title, fontsize=12)
            ax.set_title(f"Forward Selection: {title}", fontsize=13, fontweight="bold")
            ax.legend(fontsize=9, loc="lower right")
            ax.set_xlim(step_vals[0] - 0.5, step_vals[-1] + 0.5)
            ax.grid(True, alpha=0.3)

        fig.suptitle(
            f"Greedy Forward Selection — {metric_label}\n"
            f"(normalization: {norm_method}, scored by AUC)",
            fontsize=14, fontweight="bold", y=1.02,
        )
        plt.tight_layout()

        path = save_figure(fig, self.output_dir / "minimal_coverage_curve.png")
        result.add_file(path)
        logger.info(f"  Saved: {path}")


# =============================================================================
# Top-level job function for SLURM (must be picklable)
# =============================================================================

def run_attribution_job(
    output_dir: str,
    config_path: str,
    norm_methods: Optional[List[str]] = None,
    fast_mode: bool = False,
    variance_threshold: float = 0.95,
    mode: str = "all",
    metric: str = "activity",
    variance_sweep: Optional[List[float]] = None,
    agg_funcs: Optional[List[str]] = None,
    pca_optimized_dir: Optional[str] = None,
    enforce_phase: bool = False,
) -> str:
    """
    Run organelle attribution as a standalone SLURM job.

    Top-level function (not a method) so submitit can pickle it.
    Returns a status string.
    """
    import traceback
    from types import SimpleNamespace

    _check_copairs_installed()

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

        stage = OrganelleAttributionStage(
            data_context=data_shim,
            config=config_shim,
            level="guide",
            norm_methods=norm_methods,
            config_path=Path(config_path),
            fast_mode=fast_mode,
            variance_threshold=variance_threshold,
            pca_optimized_dir=pca_optimized_dir,
            mode=mode,
            metric=metric,
            variance_sweep=variance_sweep,
            agg_funcs=agg_funcs,
            enforce_phase=enforce_phase,
        )
        # Override output_dir directly since we're not going through orchestrator
        stage._output_dir = output_dir / "13_organelle_attribution"
        stage._output_dir.mkdir(parents=True, exist_ok=True)

        result = stage.run()

        if result.errors:
            return f"FAILED: {'; '.join(result.errors)}"
        return f"OK: {len(result.output_files)} files generated"

    except Exception as e:
        traceback.print_exc()
        return f"ERROR: {e}"


# =============================================================================
# CLI
# =============================================================================

def main():
    """Run organelle attribution as a standalone script."""
    _check_copairs_installed()

    import argparse
    from types import SimpleNamespace

    parser = argparse.ArgumentParser(
        description="Organelle Attribution: Leave-one-out mAP analysis on cross-experiment dino coembedding"
    )
    parser.add_argument("-o", "--output-dir", default=None,
                        help="Output directory (default: auto-generated)")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH),
                        help=f"Config YAML path (default: {DEFAULT_CONFIG_PATH})")
    parser.add_argument("--norm-method", default="ntc", choices=["global", "ntc", "both"],
                        help="Normalization method: global, ntc, or both (default: ntc)")
    parser.add_argument("--fast", action="store_true",
                        help="Enable per-channel PCA reduction for faster iteration "
                             "(reduces ~40k features to PCs based on explained variance)")
    parser.add_argument("--variance-threshold", type=float, default=0.95,
                        help="Cumulative explained variance threshold for PCA (default: 0.95). "
                             "Lower values = more compression = faster but lossier.")
    parser.add_argument("--metric", default="all",
                        choices=["all", "activity", "distinctiveness", "corum", "chad"],
                        help="Which mAP metric to score by. Default: all. "
                             "With --slurm, 'all' submits a separate job per metric.")
    parser.add_argument("--mode", default="all", choices=["all", "knockout", "knockin"],
                        help="Run mode: all (both directions), knockout (backward elimination only), "
                             "knockin (forward selection only). Default: all. "
                             "With --slurm, 'all' submits knockout and knockin as parallel jobs.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Discover experiments/channels and print summary without loading data")
    parser.add_argument("--baseline-only", action="store_true",
                        help="Only compute the baseline mAP (no elimination). "
                             "Useful for comparing --fast vs full-feature baselines.")
    parser.add_argument("--variance-sweep", action="store_true",
                        help="With --baseline-only: sweep multiple PCA variance thresholds "
                             "(95%%, 99%%, 99.5%%, 99.9%%) to measure how PCA affects mAP. "
                             "PCA is fit once, then sliced — fast and memory-efficient.")
    parser.add_argument("--multi-agg", action="store_true",
                        help="Use multi-statistic aggregation (mean,std,min,max,median,sum) "
                             "instead of mean-only. Produces 6x more features but enables "
                             "NTC normalization to have an effect on DINO features.")
    parser.add_argument("--pca-optimized", type=str,
                        default="/hpc/projects/icd.fast.ops/organelle_attribution/pca_optimized",
                        help="Path to directory containing guide_pca_optimized.h5ad and "
                             "gene_pca_optimized.h5ad from the PCA optimization stage. "
                             "When set, loads pre-reduced data instead of building from scratch.")
    parser.add_argument("--enforce-phase", action="store_true",
                        help="Force Phase as the first channel in forward selection (knockin). "
                             "Useful when Phase is the baseline morphology channel.")
    parser.add_argument("--downsampled", action="store_true",
                        help="Use downsampled signal-group PCA data. Appends /downsampled "
                             "to --pca-optimized path.")

    # SLURM options
    slurm_group = parser.add_argument_group("SLURM options")
    slurm_group.add_argument("--slurm", action="store_true",
                             help="Submit as a SLURM job")
    slurm_group.add_argument("--no-wait", action="store_true",
                             help="Don't wait for SLURM job to complete")
    slurm_group.add_argument("--yes", "-y", action="store_true",
                             help="Skip confirmation prompt")
    slurm_group.add_argument("--slurm-memory", type=str, default="500GB",
                             help="Memory (default: 200GB)")
    slurm_group.add_argument("--slurm-time", type=int, default=720,
                             help="Time limit in minutes (default: 720)")
    slurm_group.add_argument("--slurm-cpus", type=int, default=64,
                             help="CPUs (default: 64)")

    args = parser.parse_args()

    # Resolve --downsampled: append /downsampled to pca-optimized path
    if args.downsampled and args.pca_optimized:
        args.pca_optimized = str(Path(args.pca_optimized) / "downsampled")

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    # Determine output path
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = Path("/hpc/projects/icd.fast.ops/organelle_attribution")
    if args.enforce_phase:
        output_dir = output_dir / "phase_first"
    output_dir.mkdir(parents=True, exist_ok=True)

    config_path = Path(args.config)

    # Resolve norm methods
    if args.norm_method == "both":
        norm_methods = ["ntc", "global"]
    else:
        norm_methods = [args.norm_method]

    # --- Dry-run mode ---
    if args.dry_run:
        data_shim = SimpleNamespace(
            experiment="cross_experiment",
            graph_output_path=output_dir,
        )
        config_shim = SimpleNamespace(experiment="cross_experiment")
        stage = OrganelleAttributionStage(
            data_context=data_shim,
            config=config_shim,
            level="guide",
            norm_methods=norm_methods,
            config_path=config_path,
        )
        stage.dry_run()
        return

    # --- baseline-only → override mode so elimination is skipped ---
    if args.baseline_only:
        args.mode = "baseline"
        # Default to single metric if not specified
        if args.metric == "all":
            args.metric = "activity"
        # Enable fast mode for sweep (PCA is always used in sweep)
        if args.variance_sweep:
            args.fast = True

    # --- SLURM mode ---
    if args.slurm:
        _run_slurm_mode(args, output_dir, config_path, norm_methods)
        return

    # --- Local mode ---
    metrics = (
        list(OrganelleAttributionStage.VALID_METRICS)
        if args.metric == "all"
        else [args.metric]
    )

    for metric in metrics:
        print(f"\n--- Running metric: {metric} ---")
        data_shim = SimpleNamespace(
            experiment="cross_experiment",
            graph_output_path=output_dir,
        )
        config_shim = SimpleNamespace(experiment="cross_experiment")

        sweep_thresholds = None
        if getattr(args, "variance_sweep", False):
            sweep_thresholds = [0.90, 0.95, 0.99, 0.995, 0.999]

        multi_agg = None
        if getattr(args, "multi_agg", False):
            multi_agg = ["mean", "std", "min", "max", "median", "sum"]

        stage = OrganelleAttributionStage(
            data_context=data_shim,
            config=config_shim,
            level="guide",
            norm_methods=norm_methods,
            config_path=config_path,
            fast_mode=args.fast,
            variance_threshold=args.variance_threshold,
            pca_optimized_dir=args.pca_optimized,
            mode=args.mode,
            metric=metric,
            variance_sweep=sweep_thresholds,
            agg_funcs=multi_agg,
            enforce_phase=args.enforce_phase,
        )
        stage._output_dir = output_dir / "13_organelle_attribution"
        stage._output_dir.mkdir(parents=True, exist_ok=True)

        result = stage.run()

        print(f"\nOutput: {stage.output_dir}")
        print(f"Files: {len(result.output_files)}")
        if result.errors:
            print(f"Errors: {len(result.errors)}")
            for err in result.errors:
                print(f"  - {err}")


def _run_slurm_mode(
    args, output_dir: Path, config_path: Path, norm_methods: List[str]
) -> None:
    """Submit the attribution pipeline as SLURM job(s).

    When ``--mode all`` (the default), submits knockout and knockin as two
    parallel jobs so they run concurrently on the cluster.  Otherwise submits
    a single job for the requested mode.
    """
    from ops_utils.hpc.slurm_batch_utils import submit_parallel_jobs

    slurm_params = {
        "timeout_min": args.slurm_time,
        "mem": args.slurm_memory,
        "cpus_per_task": args.slurm_cpus,
        "slurm_partition": "cpu,gpu",
    }

    # Variance sweep thresholds (if requested)
    sweep_thresholds = None
    if getattr(args, "variance_sweep", False):
        sweep_thresholds = [0.90, 0.95, 0.99, 0.995, 0.999]

    multi_agg = None
    if getattr(args, "multi_agg", False):
        multi_agg = ["mean", "std", "min", "max", "median", "sum"]

    common_kwargs = {
        "output_dir": str(output_dir),
        "config_path": str(config_path),
        "norm_methods": norm_methods,
        "fast_mode": args.fast,
        "variance_threshold": args.variance_threshold,
        "variance_sweep": sweep_thresholds,
        "agg_funcs": multi_agg,
        "pca_optimized_dir": args.pca_optimized,
        "enforce_phase": getattr(args, "enforce_phase", False),
    }

    # Resolve metric and mode lists
    metrics = (
        list(OrganelleAttributionStage.VALID_METRICS)
        if args.metric == "all"
        else [args.metric]
    )
    modes = ["knockout", "knockin"] if args.mode == "all" else [args.mode]

    # Build job list: one job per (metric, mode) combination
    jobs = []
    for metric in metrics:
        for mode in modes:
            jobs.append({
                "name": f"organelle_attribution_{metric}_{mode}",
                "func": run_attribution_job,
                "kwargs": {**common_kwargs, "mode": mode, "metric": metric},
            })

    mode_desc = f"{len(metrics)} metrics × {len(modes)} modes = {len(jobs)} parallel jobs"

    if not args.yes:
        print(f"\nOrganelle Attribution SLURM Job(s):")
        print(f"  Output:    {output_dir}")
        print(f"  Config:    {config_path}")
        print(f"  Norm:      {', '.join(norm_methods)}")
        print(f"  Metrics:   {', '.join(metrics)}")
        print(f"  Modes:     {', '.join(modes)}")
        print(f"  Jobs:      {mode_desc}")
        if args.pca_optimized:
            print(f"  PCA:       pre-optimized ({args.pca_optimized})")
        else:
            print(f"  Fast:      {args.fast}" + (f" (variance threshold: {args.variance_threshold})" if args.fast else ""))
        print(f"  Partition: cpu,gpu")
        print(f"  Memory:    {args.slurm_memory}")
        print(f"  Time:      {args.slurm_time} min")
        print(f"  CPUs:      {args.slurm_cpus}")
        confirm = input("\nSubmit? [y/N] ").strip().lower()
        if confirm != "y":
            print("Cancelled.")
            return

    result = submit_parallel_jobs(
        jobs_to_submit=jobs,
        experiment="organelle_attribution",
        slurm_params=slurm_params,
        log_dir="organelle_attribution",
        manifest_prefix="organelle_attribution",
        wait_for_completion=not args.no_wait,
    )

    if result.get("success"):
        print(f"\nJob(s) submitted: {result.get('base_job_id')}")
        print(f"  Jobs: {len(jobs)}")
    else:
        print("\nJob submission failed!")


if __name__ == "__main__":
    main()
