"""
Organelle Attribution Stage: Leave-One-Out mAP Analysis on Cross-Experiment Dino Coembedding.

Answers: **which organelle channels drive phenotypic discrimination of gene perturbations?**

Workflow:
1. Discover all experiments with dino guide_bulked_*.h5ad files, filter bad experiments
2. Build biology-aware coembedding via concatenate_experiments_comprehensive
3. Run the 4 copairs mAP metrics (activity, distinctiveness, CORUM, CHAD) on full features
4. For each channel label, remove that label's features and re-run mAP metrics
5. Compute deltas to reveal each organelle's contribution + which perturbations depend on it

Usage:
  # Local mode:
  python -m organelle_profiler.fe_graphs.stages.fe_graphs_organelle_attribution_stage -o /path/to/output

  # SLURM mode:
  python -m organelle_profiler.fe_graphs.stages.fe_graphs_organelle_attribution_stage --slurm
  python -m organelle_profiler.fe_graphs.stages.fe_graphs_organelle_attribution_stage --slurm --slurm-memory 256GB

CLI arguments:
  -o, --output-dir      Output directory (default: auto-generated)
  --config              Path to config YAML (default: organelle_attribution_config.yaml)
  --norm-method         Normalization method(s): global, ntc, or both (default: both)
  --dry-run             Discover experiments/channels and print summary without loading data

SLURM options:
  --slurm               Submit as a single SLURM job
  --slurm-memory        Memory (default: 256GB)
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
    compute_threshold_sweep_auc,
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

# Default config path
DEFAULT_CONFIG_PATH = Path(__file__).parents[4] / "configs" / "organelle_attribution_config.yaml"

# Channel maps path (prefer fast_ops partition which has the most up-to-date entries)
CHANNEL_MAPS_PATH = Path("/hpc/projects/icd.fast.ops/configs/ops_channel_maps.yaml")
CHANNEL_MAPS_PATH_FALLBACK = Path("/hpc/projects/intracellular_dashboard/ops/configs/ops_channel_maps.yaml")

# Storage roots to search for dino features (priority order)
DEFAULT_STORAGE_ROOTS = [
    Path("/hpc/projects/icd.fast.ops"),
    Path("/hpc/projects/intracellular_dashboard/ops"),
    Path("/hpc/projects/icd.ops"),
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

    def __init__(
        self,
        data_context,
        config,
        level: str = "guide",
        norm_methods: Optional[List[str]] = None,
        config_path: Optional[Path] = None,
        **kwargs,
    ):
        super().__init__(data_context, config, level, **kwargs)
        self.norm_methods = norm_methods or ["ntc", "global"]
        self.config_path = config_path or DEFAULT_CONFIG_PATH

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
        self._feature_dir = self.attr_config.get("feature_dir", "dino_features_v1")
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
        from ops_model.data.feature_metadata import FeatureMetadata
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
            logger.info(f"  NORMALIZATION: {norm_method}")
            logger.info(f"{'='*70}")

            # Each norm method gets its own output subdirectory
            norm_dir = base_output_dir / f"{norm_method}_norm"
            norm_dir.mkdir(parents=True, exist_ok=True)
            self._output_dir = norm_dir

            # Copy raw data (normalization writes in-place)
            t_copy = time.time()
            adata_guide = adata_guide_raw.copy()
            adata_gene = adata_gene_raw.copy()
            logger.info(f"  Data copy: {time.time()-t_copy:.1f}s")

            # Step 4: Z-score normalize
            t_norm = time.time()
            logger.info(f"Step 4: Z-score normalizing (method={norm_method})...")
            adata_guide, adata_gene = self._normalize_adata(
                adata_guide, adata_gene, norm_method
            )
            logger.info(f"  Normalization done: {time.time()-t_norm:.1f}s")

            # Step 5: Baseline mAP
            logger.info("Step 5: Running baseline mAP battery...")
            baseline = self._run_map_battery(adata_guide, adata_gene, "baseline")
            if baseline is None:
                result.add_error(f"Baseline mAP computation failed ({norm_method} norm)")
                continue

            self._save_baseline(baseline, result)
            logger.info(f"  Baseline ({norm_method}): activity={baseline['active_ratio']:.2%}, "
                         f"distinct={baseline['distinctive_ratio']:.2%}")

            # Step 6: Leave-one-out ablation
            logger.info("Step 6: Leave-one-out ablation...")
            ablation_results = self._leave_one_out(
                adata_guide, adata_gene, label_to_cols, baseline
            )

            # Step 7: Compute attribution and generate outputs
            logger.info("Step 7: Computing attribution and generating outputs...")
            self._compute_and_save_attribution(baseline, ablation_results, label_to_cols, result)

            # Step 8: Greedy forward selection — minimal set for full coverage
            logger.info("Step 8: Greedy forward selection for minimal coverage set...")
            self._greedy_minimal_set(
                adata_guide, adata_gene, label_to_cols, baseline, norm_method, result
            )

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

        file_norm = _norm(file_channel)

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
        from ops_model.data.feature_metadata import FeatureMetadata
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
                feature_dir=self._feature_dir,
                recompute_embeddings=False,
                compute_pca=False,
                compute_umap=False,
                compute_phate=False,
                normalize_on_pooling=True,
                normalize_on_controls=False,
                join=self._join,
                verbose=False,
                search_dirs=self._storage_roots,
                use_preaggregated=True,
                metadata_path=_maps,
                signal_map=signal_map,
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
        """Z-score normalize the combined AnnData objects."""
        for adata, level_name in [(adata_guide, "guide"), (adata_gene, "gene")]:
            feature_cols = list(adata.var_names)
            df = pd.DataFrame(adata.X, columns=feature_cols)
            for col in adata.obs.columns:
                df[col] = adata.obs[col].values

            df = zscore_normalize(
                df, feature_cols,
                method=norm_method,
                perturbation_col="perturbation",
            )

            # Write back normalized features
            adata.X = df[feature_cols].values.astype(np.float32)
            logger.info(f"  Normalized {level_name}-level ({norm_method}): {adata.n_obs} obs, {adata.n_vars} features")

        return adata_guide, adata_gene

    # -------------------------------------------------------------------------
    # Step 5: mAP battery
    # -------------------------------------------------------------------------

    def _run_map_battery(
        self,
        adata_guide: ad.AnnData,
        adata_gene: ad.AnnData,
        run_label: str,
    ) -> Optional[Dict[str, Any]]:
        """
        Run all 4 mAP metrics + AUC scores.

        Returns dict with all results or None on failure.
        """
        try:
            t0 = time.time()

            # 1. Activity (guide level)
            logger.info(f"    [{run_label}] Running activity mAP ({adata_guide.n_obs} guides × {adata_guide.n_vars} features)...")
            activity_map, active_ratio = phenotypic_activity_assesment(
                adata_guide, plot_results=False
            )
            activity_auc = compute_auc_score(activity_map)
            activity_sweep = compute_threshold_sweep_auc(activity_map)
            logger.info(f"    [{run_label}] Activity done ({time.time()-t0:.1f}s): {active_ratio:.2%} active")

            # 2. Distinctiveness (guide level)
            t1 = time.time()
            logger.info(f"    [{run_label}] Running distinctiveness mAP...")
            distinct_map, distinctive_ratio = phenotypic_distinctivness(
                adata_guide, activity_map, plot_results=False
            )
            distinct_auc = compute_auc_score(distinct_map)
            distinct_sweep = compute_threshold_sweep_auc(distinct_map)
            logger.info(f"    [{run_label}] Distinctiveness done ({time.time()-t1:.1f}s): {distinctive_ratio:.2%} distinct")

            # 3. CORUM consistency (gene level)
            t2 = time.time()
            logger.info(f"    [{run_label}] Running CORUM consistency mAP ({adata_gene.n_obs} genes)...")
            corum_map, corum_ratio = phenotypic_consistency_corum(
                adata_gene, activity_map, plot_results=False
            )
            corum_auc = compute_auc_score(corum_map)
            corum_sweep = compute_threshold_sweep_auc(corum_map)
            logger.info(f"    [{run_label}] CORUM done ({time.time()-t2:.1f}s): {corum_ratio:.2%}")

            # 4. CHAD consistency (gene level)
            t3 = time.time()
            logger.info(f"    [{run_label}] Running CHAD consistency mAP...")
            chad_map, chad_ratio = phenotypic_consistency_manual_annotation(
                adata_gene, activity_map, plot_results=False
            )
            chad_auc = compute_auc_score(chad_map)
            chad_sweep = compute_threshold_sweep_auc(chad_map)
            logger.info(f"    [{run_label}] CHAD done ({time.time()-t3:.1f}s): {chad_ratio:.2%}")
            logger.info(f"    [{run_label}] mAP battery total: {time.time()-t0:.1f}s")

            return {
                "label": run_label,
                "activity_map": activity_map,
                "active_ratio": active_ratio,
                "activity_auc": activity_auc,
                "activity_sweep": activity_sweep,
                "distinct_map": distinct_map,
                "distinctive_ratio": distinctive_ratio,
                "distinct_auc": distinct_auc,
                "distinct_sweep": distinct_sweep,
                "corum_map": corum_map,
                "corum_ratio": corum_ratio,
                "corum_auc": corum_auc,
                "corum_sweep": corum_sweep,
                "chad_map": chad_map,
                "chad_ratio": chad_ratio,
                "chad_auc": chad_auc,
                "chad_sweep": chad_sweep,
            }

        except Exception as e:
            logger.error(f"mAP battery failed for '{run_label}': {e}")
            import traceback
            traceback.print_exc()
            return None

    def _save_baseline(self, baseline: Dict, result: StageResult) -> None:
        """Save baseline mAP results to CSV files."""
        baseline_dir = self.output_dir / "baseline"
        baseline_dir.mkdir(parents=True, exist_ok=True)

        for key in ["activity_map", "distinct_map", "corum_map", "chad_map"]:
            csv_path = baseline_dir / f"{key}.csv"
            baseline[key].to_csv(csv_path, index=False)
            result.add_file(csv_path)

        # Summary
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
            "sweep_auc": [
                baseline["activity_sweep"],
                baseline["distinct_sweep"],
                baseline["corum_sweep"],
                baseline["chad_sweep"],
            ],
        }
        summary_path = baseline_dir / "baseline_summary.csv"
        pd.DataFrame(summary).to_csv(summary_path, index=False)
        result.add_file(summary_path)

    # -------------------------------------------------------------------------
    # Step 6: Leave-one-out ablation
    # -------------------------------------------------------------------------

    def _leave_one_out(
        self,
        adata_guide: ad.AnnData,
        adata_gene: ad.AnnData,
        label_to_cols: Dict[str, List[str]],
        baseline: Dict,
    ) -> List[Dict[str, Any]]:
        """
        For each channel label, remove its features and re-run mAP battery.
        """
        ablation_results = []
        n_labels = len(label_to_cols)
        t_loo_start = time.time()

        for i, (label, cols_to_remove) in enumerate(sorted(label_to_cols.items()), 1):
            t_iter = time.time()
            logger.info(f"\n  Ablation {i}/{n_labels}: removing '{label}' ({len(cols_to_remove)} features)")

            # Create ablated AnnData by dropping columns
            cols_set = set(cols_to_remove)
            keep_mask = np.array([v not in cols_set for v in adata_guide.var_names])

            if keep_mask.sum() == 0:
                logger.warning(f"  Skipping '{label}': would remove ALL features")
                continue

            ablated_guide = adata_guide[:, keep_mask].copy()
            # For gene-level, use same column mask (var_names should match)
            keep_mask_gene = np.array([v not in cols_set for v in adata_gene.var_names])
            ablated_gene = adata_gene[:, keep_mask_gene].copy()

            logger.info(f"  Ablated: {ablated_guide.n_vars} features remaining "
                         f"(removed {len(cols_to_remove)})")

            # Run mAP battery on ablated data
            abl_result = self._run_map_battery(ablated_guide, ablated_gene, label)

            if abl_result is None:
                logger.warning(f"  mAP battery failed for ablation '{label}', skipping")
                continue

            # Compute per-perturbation deltas
            per_pert_delta = self._compute_per_perturbation_delta(
                baseline, abl_result, label
            )

            iter_elapsed = time.time() - t_iter
            total_elapsed = time.time() - t_loo_start
            avg_per_iter = total_elapsed / i
            eta = avg_per_iter * (n_labels - i)
            logger.info(f"  Ablation {i}/{n_labels} done in {iter_elapsed:.1f}s "
                         f"(avg {avg_per_iter:.1f}s/iter, ETA {eta:.0f}s)")

            ablation_results.append({
                "label": label,
                "n_features_removed": len(cols_to_remove),
                "n_features_remaining": int(keep_mask.sum()),
                "active_ratio": abl_result["active_ratio"],
                "distinctive_ratio": abl_result["distinctive_ratio"],
                "corum_ratio": abl_result["corum_ratio"],
                "chad_ratio": abl_result["chad_ratio"],
                "activity_auc": abl_result["activity_auc"],
                "distinct_auc": abl_result["distinct_auc"],
                "corum_auc": abl_result["corum_auc"],
                "chad_auc": abl_result["chad_auc"],
                "activity_sweep": abl_result["activity_sweep"],
                "distinct_sweep": abl_result["distinct_sweep"],
                "corum_sweep": abl_result["corum_sweep"],
                "chad_sweep": abl_result["chad_sweep"],
                "per_pert_delta": per_pert_delta,
                "full_result": abl_result,
            })

            logger.info(
                f"  Result: activity={abl_result['active_ratio']:.2%} "
                f"(delta={baseline['active_ratio'] - abl_result['active_ratio']:+.2%}), "
                f"distinct={abl_result['distinctive_ratio']:.2%} "
                f"(delta={baseline['distinctive_ratio'] - abl_result['distinctive_ratio']:+.2%})"
            )

        return ablation_results

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
        all_features = list(adata_guide.var_names)
        remaining = set(labels)
        selected_order: List[str] = []

        # Track scores at each step
        steps: List[Dict[str, Any]] = []

        # Step 0: no features (all zeros → everything random)
        steps.append({
            "step": 0,
            "label_added": "(none)",
            "n_features": 0,
            "activity_ratio": 0.0,
            "distinct_ratio": 0.0,
            "activity_auc": 0.0,
            "distinct_auc": 0.0,
        })

        logger.info(f"  Forward selection across {len(labels)} organelle groups...")

        for step_num in range(1, len(labels) + 1):
            best_label = None
            best_score = -1.0
            best_result_dict = None

            for candidate in sorted(remaining):
                # Build feature set = all selected so far + candidate
                include_labels = selected_order + [candidate]
                include_cols = []
                for lbl in include_labels:
                    include_cols.extend(label_to_cols[lbl])

                # Subset adata to only these features
                include_set = set(include_cols)
                keep_mask = np.array([v in include_set for v in all_features])
                if keep_mask.sum() == 0:
                    continue

                subset_guide = adata_guide[:, keep_mask].copy()
                # Gene-level uses same features
                keep_mask_gene = np.array([v in include_set for v in adata_gene.var_names])
                subset_gene = adata_gene[:, keep_mask_gene].copy()

                r = self._run_map_battery(subset_guide, subset_gene, f"fwd_{candidate}")
                if r is None:
                    continue

                # Score = average of activity_auc + distinct_auc (primary metrics)
                score = (r["activity_auc"] + r["distinct_auc"]) / 2.0
                if score > best_score:
                    best_score = score
                    best_label = candidate
                    best_result_dict = r

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
                "activity_ratio": best_result_dict["active_ratio"],
                "distinct_ratio": best_result_dict["distinctive_ratio"],
                "activity_auc": best_result_dict["activity_auc"],
                "distinct_auc": best_result_dict["distinct_auc"],
            })

            logger.info(
                f"  Step {step_num}: +'{best_label}' → "
                f"activity={best_result_dict['active_ratio']:.2%}, "
                f"distinct={best_result_dict['distinctive_ratio']:.2%}, "
                f"features={n_feats}"
            )

        # Save step table
        steps_df = pd.DataFrame(steps)
        steps_df["baseline_activity_ratio"] = baseline["active_ratio"]
        steps_df["baseline_distinct_ratio"] = baseline["distinctive_ratio"]
        steps_df["baseline_activity_auc"] = baseline["activity_auc"]
        steps_df["baseline_distinct_auc"] = baseline["distinct_auc"]

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

        X-axis: number of organelle groups, Y-axis: mAP metric score.
        Horizontal dashed line at full baseline, shaded 95% band.
        Each point labeled with the organelle added at that step.
        """
        fig, axes = plt.subplots(1, 2, figsize=(20, 8))

        metrics = [
            ("activity_auc", "Activity AUC", baseline["activity_auc"]),
            ("distinct_auc", "Distinctiveness AUC", baseline["distinct_auc"]),
        ]

        for ax, (col, title, bl_val) in zip(axes, metrics):
            steps = steps_df["step"].values
            values = steps_df[col].values

            # Main curve
            ax.plot(steps, values, "o-", color="#1976d2", linewidth=2,
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
                ax.axvline(steps[first_95], color="#388e3c", linestyle="--",
                           linewidth=1.5, alpha=0.7,
                           label=f"95% reached at step {steps[first_95]}")

            # Label each point with the organelle added
            for i, row in steps_df.iterrows():
                if row["step"] == 0:
                    continue
                label = row["label_added"]
                # Truncate long labels
                if len(label) > 25:
                    label = label[:22] + "..."
                ax.annotate(
                    label,
                    (row["step"], row[col]),
                    textcoords="offset points",
                    xytext=(8, 8 if i % 2 == 0 else -14),
                    fontsize=7,
                    rotation=30,
                    ha="left",
                    va="bottom" if i % 2 == 0 else "top",
                )

            ax.set_xlabel("Number of Organelle Groups Included", fontsize=12)
            ax.set_ylabel(title, fontsize=12)
            ax.set_title(f"Minimal Coverage: {title}", fontsize=13, fontweight="bold")
            ax.legend(fontsize=9, loc="lower right")
            ax.set_xlim(-0.5, steps[-1] + 0.5)
            ax.grid(True, alpha=0.3)

        fig.suptitle(
            f"Greedy Forward Selection — Minimal Organelle Set for Full mAP Coverage\n"
            f"(normalization: {norm_method})",
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
    parser.add_argument("--norm-method", default="both", choices=["global", "ntc", "both"],
                        help="Normalization method: global, ntc, or both (default: both)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Discover experiments/channels and print summary without loading data")

    # SLURM options
    slurm_group = parser.add_argument_group("SLURM options")
    slurm_group.add_argument("--slurm", action="store_true",
                             help="Submit as a SLURM job")
    slurm_group.add_argument("--no-wait", action="store_true",
                             help="Don't wait for SLURM job to complete")
    slurm_group.add_argument("--yes", "-y", action="store_true",
                             help="Skip confirmation prompt")
    slurm_group.add_argument("--slurm-memory", type=str, default="256GB",
                             help="Memory (default: 256GB)")
    slurm_group.add_argument("--slurm-time", type=int, default=120,
                             help="Time limit in minutes (default: 120)")
    slurm_group.add_argument("--slurm-cpus", type=int, default=16,
                             help="CPUs (default: 16)")

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    # Determine output path
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = Path("/hpc/projects/icd.fast.ops/organelle_attribution")
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

    # --- SLURM mode ---
    if args.slurm:
        _run_slurm_mode(args, output_dir, config_path, norm_methods)
        return

    # --- Local mode ---
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
    """Submit the attribution pipeline as a single SLURM job."""
    from ops_utils.hpc.slurm_batch_utils import submit_parallel_jobs

    slurm_params = {
        "timeout_min": args.slurm_time,
        "mem": args.slurm_memory,
        "cpus_per_task": args.slurm_cpus,
        "slurm_partition": "cpu",
    }

    job = {
        "name": "organelle_attribution",
        "func": run_attribution_job,
        "kwargs": {
            "output_dir": str(output_dir),
            "config_path": str(config_path),
            "norm_methods": norm_methods,
        },
    }

    if not args.yes:
        print(f"\nOrganelle Attribution SLURM Job:")
        print(f"  Output: {output_dir}")
        print(f"  Config: {config_path}")
        print(f"  Norm:   {', '.join(norm_methods)}")
        print(f"  Memory: {args.slurm_memory}")
        print(f"  Time:   {args.slurm_time} min")
        print(f"  CPUs:   {args.slurm_cpus}")
        confirm = input("\nSubmit? [y/N] ").strip().lower()
        if confirm != "y":
            print("Cancelled.")
            return

    result = submit_parallel_jobs(
        jobs_to_submit=[job],
        experiment="organelle_attribution",
        slurm_params=slurm_params,
        log_dir=str(output_dir / "slurm_logs"),
        manifest_prefix="organelle_attribution",
        wait_for_completion=not args.no_wait,
    )

    if result.get("success"):
        print(f"\nJob submitted: {result.get('base_job_id')}")
    else:
        print("\nJob submission failed!")


if __name__ == "__main__":
    main()
