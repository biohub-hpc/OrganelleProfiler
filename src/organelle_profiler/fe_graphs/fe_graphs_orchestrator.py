"""
Analysis orchestrator for feature graph generation.

Coordinates analysis across levels (cell → guide → gene) using a stage-based
pipeline. Results chain from level to level for integrated analysis.

New Architecture (v3.0):
- Level-based: cell → guide → gene hierarchy
- Stage-based: QC → Embedding → Clustering → Differential → Summary
- Chaining: Results propagate between levels
"""

import logging
from pathlib import Path
from typing import Dict, Optional, List, Type, Any
from dataclasses import dataclass, field
import pandas as pd

from .fe_graphs_config import GraphConfig, PlotConfig, AnalysisConfig
from .core.fe_graphs_data_loader import DataLoader, DataContext
from .fe_graphs_level_pipeline import LevelPipeline, LevelResult

from ops_utils.profiling.decorators import notify_step, versioned_function

logger = logging.getLogger(__name__)


@dataclass
class OrchestratorResult:
    """Container for overall orchestration results."""
    success: bool = True
    level_results: Dict[str, LevelResult] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)
    
    @property
    def all_output_files(self) -> List[Path]:
        """Get all output files from all levels."""
        files = []
        for level_result in self.level_results.values():
            files.extend(level_result.get_all_files())
        return files
    
    @property
    def all_metrics(self) -> Dict:
        """Get all metrics from all levels."""
        metrics = {}
        for level, level_result in self.level_results.items():
            for key, val in level_result.get_all_metrics().items():
                metrics[f"{level}_{key}"] = val
        return metrics


class AnalysisOrchestrator:
    """
    Coordinates feature graph analysis across cell/guide/gene levels.
    
    Runs a unified analytical pipeline at each level:
    1. QC - Quality control (spatial drift, batch effects)
    2. Embedding - PCA and UMAP dimensionality reduction
    3. Clustering - HDBSCAN, Leiden, KMeans
    4. Differential - NTC comparison, volcano plots
    5. Summary - Aggregation and reporting
    
    Results chain between levels for integrated analysis.
    
    Parameters
    ----------
    experiment : str
        Experiment name or shorthand.
    debug_cell_fraction : float, optional
        Fraction of cells to sample for debug mode.
    skip_object_features : bool
        Whether to skip loading object-level CSVs.
    use_cuml : bool
        Whether to use cuML GPU acceleration.
    cluster_algo : str
        Clustering algorithm(s): "all", "hdbscan", "kmeans", or "leiden".
    skip_interactive_plots : bool
        Whether to skip interactive dashboard data.
    drift_method : str
        Method for drift analysis.
    analysis_mode : str
        Which levels to run: "all", "guide", "gene", or "guide_and_gene".
    
    Examples
    --------
    >>> # Full pipeline (cell → guide → gene)
    >>> orchestrator = AnalysisOrchestrator("ops0094_20251217")
    >>> result = orchestrator.run()
    
    >>> # Specific levels only
    >>> result = orchestrator.run(levels=["gene"])
    
    >>> # Skip specific stages
    >>> result = orchestrator.run(skip_stages=["clustering"])
    """
    
    # Default level order (cell first so it can chain to guide and gene)
    DEFAULT_LEVELS = ["cell", "guide", "gene"]
    
    def __init__(
        self,
        experiment: str,
        debug_cell_fraction: Optional[float] = None,
        skip_object_features: bool = False,
        just_cell_painting: bool = False,
        use_cuml: bool = True,
        cluster_algo: str = "all",
        skip_interactive_plots: bool = True,
        drift_method: str = "three_segment_regression",
        analysis_mode: str = "all",
        use_cache: bool = True,
        skip_complete: bool = False,
    ):
        # Store config
        self.config = GraphConfig(
            experiment=experiment,
            debug_cell_fraction=debug_cell_fraction,
            skip_object_features=skip_object_features,
            just_cell_painting=just_cell_painting,
            use_cuml=use_cuml,
            cluster_algo=cluster_algo,
            skip_interactive=skip_interactive_plots,
            drift_method=drift_method,
            analysis_mode=analysis_mode,
            use_cache=use_cache,
            skip_complete=skip_complete,
        )
        self.plot_config = PlotConfig()
        self.analysis_config = AnalysisConfig()
        
        # Load data
        print("Initializing AnalysisOrchestrator...")
        loader = DataLoader(
            experiment,
            debug_cell_fraction=debug_cell_fraction,
            skip_object_features=skip_object_features,
            just_cell_painting=just_cell_painting,
        )
        self.data = loader.load()
        
        print(f"Feature data loaded from: {self.data.analysis_path}")
        print(f"Graphs will be saved to: {self.data.graph_output_path}")
        
        # Validate organelle groups (single source of truth)
        n_groups = len(self.data.organelle_groups)
        n_features = len(self.data.feature_columns)
        print(f"Organelle groups: {n_groups} (from adata.var['organelle'])")
        
        if n_groups == 0:
            print("WARNING: No organelle groups found! Check that adata.var has 'organelle' column.")
            print("  Run: python -m ops_utils.io.anndata_utils -e <experiment>")
        elif n_groups > 50:
            print(f"WARNING: Found {n_groups} organelle groups for {n_features} features - seems too high!")
            print(f"  Expected ~20 organelles. First 5: {list(self.data.organelle_groups.keys())[:5]}")
            print("  Check that adata.var['organelle'] contains organelle names, not feature names.")
        else:
            print(f"  Organelles: {list(self.data.organelle_groups.keys())}")
        
        # Store level results for chaining
        self.level_results: Dict[str, LevelResult] = {}
    
    # Legacy aliases for backwards compatibility
    @property
    def experiment(self) -> str:
        return self.config.experiment
    
    @property
    def graph_output_path(self) -> Path:
        return self.data.graph_output_path
    
    @notify_step(
        step_message="Started graph generation (v3)",
        success_message="Finished graph generation (v3)",
    )
    @versioned_function("v3.0")
    def run(
        self,
        levels: Optional[List[str]] = None,
        stages: Optional[List[str]] = None,
        skip_stages: Optional[List[str]] = None,
    ) -> OrchestratorResult:
        """
        Run the analysis pipeline.
        
        Parameters
        ----------
        levels : list[str], optional
            Levels to run: "cell", "guide", "gene". If None, uses analysis_mode.
        stages : list[str], optional
            Stages to run: "qc", "embedding", "clustering", "differential", "summary".
        skip_stages : list[str], optional
            Stages to skip.
            
        Returns
        -------
        OrchestratorResult
            Container with all level results.
        """
        print("\n" + "="*70)
        print("  FEATURE GRAPH ANALYSIS PIPELINE (v3.0)")
        print("="*70)
        print(f"  Experiment: {self.config.experiment}")
        print(f"  Analysis mode: {self.config.analysis_mode}")
        print("="*70 + "\n")
        
        result = OrchestratorResult()
        
        # Determine which levels to run
        levels_to_run = self._get_levels_to_run(levels)
        
        print(f"Running levels: {levels_to_run}")
        if stages:
            print(f"Stages: {stages}")
        if skip_stages:
            print(f"Skipping stages: {skip_stages}")
        print()
        
        # Run each level in order, chaining results
        for level in levels_to_run:
            try:
                level_result = self._run_level(level, stages, skip_stages)
                result.level_results[level] = level_result
                self.level_results[level] = level_result
                
                if not level_result.success:
                    for stage_name, stage_result in level_result.stages.items():
                        for error in stage_result.errors:
                            result.errors.append(f"{level}/{stage_name}: {error}")
                            
            except Exception as e:
                import traceback
                error_msg = f"{level}: {str(e)}"
                result.errors.append(error_msg)
                logger.error(f"Error at {level} level: {e}")
                traceback.print_exc()
        
        # Run integrated analysis if we have multiple levels
        if len(self.level_results) > 1:
            self._run_integrated_analysis(result)
        
        result.success = len(result.errors) == 0
        
        # Print summary
        self._print_summary(result)
        
        return result
    
    def _get_levels_to_run(self, levels: Optional[List[str]]) -> List[str]:
        """Determine which levels to run based on config and arguments."""
        if levels is not None:
            return [l for l in self.DEFAULT_LEVELS if l in levels]
        
        mode = self.config.analysis_mode
        
        if mode == "guide":
            return ["guide"]
        elif mode == "gene":
            return ["gene"]
        elif mode == "guide_and_gene":
            return ["guide", "gene"]
        else:  # "all"
            return self.DEFAULT_LEVELS
    
    def _run_level(
        self,
        level: str,
        stages: Optional[List[str]],
        skip_stages: Optional[List[str]],
    ) -> LevelResult:
        """Run the pipeline for a single level."""
        pipeline = LevelPipeline(
            data_context=self.data,
            config=self.config,
            level=level,
            upstream_level_results=self.level_results,  # Chain from prior levels
            plot_config=self.plot_config,
            analysis_config=self.analysis_config,
        )
        
        return pipeline.run(stages=stages, skip_stages=skip_stages)
    
    def _run_integrated_analysis(self, result: OrchestratorResult) -> None:
        """Run cross-level integrated analysis."""
        print("\n" + "#"*60)
        print("#  INTEGRATED ANALYSIS (Cross-Level)")
        print("#"*60 + "\n")
        
        integrated_dir = self.data.graph_output_path / "4_integrated"
        integrated_dir.mkdir(parents=True, exist_ok=True)
        
        # Create integrated hit summary
        hit_summary_dir = integrated_dir / "hit_summary"
        hit_summary_dir.mkdir(exist_ok=True)
        
        self._create_integrated_hit_summary(hit_summary_dir, result)
        
        # Cross-level correlation
        correlation_dir = integrated_dir / "cross_level_correlation"
        correlation_dir.mkdir(exist_ok=True)
        
        self._create_cross_level_correlation(correlation_dir, result)
    
    def _create_integrated_hit_summary(self, output_dir: Path, result: OrchestratorResult) -> None:
        """Create integrated hit summary across levels."""
        import matplotlib.pyplot as plt
        from .plotting.fe_graphs_utils import save_figure
        
        hit_lists = {}
        
        for level, level_result in self.level_results.items():
            if "summary" in level_result.stages:
                hit_list = level_result.stages["summary"].data.get("hit_list")
                if hit_list is not None:
                    hit_lists[level] = hit_list
        
        if not hit_lists:
            logger.info("No hit lists available for integration")
            return
        
        # Combine hits
        all_hits = []
        for level, hits in hit_lists.items():
            hits = hits.copy()
            hits["level"] = level
            all_hits.append(hits)
        
        if all_hits:
            combined = pd.concat(all_hits, ignore_index=True)
            combined.to_csv(output_dir / "final_hits.csv", index=False)
            result.level_results["integrated"] = LevelResult(level="integrated")
            logger.info(f"Saved integrated hits to {output_dir / 'final_hits.csv'}")
    
    def _create_cross_level_correlation(self, output_dir: Path, result: OrchestratorResult) -> None:
        """Create cross-level correlation analysis."""
        # This would compare cell-level vs gene-level effects
        # Placeholder for now
        logger.info("Cross-level correlation analysis (placeholder)")
    
    def _print_summary(self, result: OrchestratorResult) -> None:
        """Print summary of results."""
        print("\n" + "="*70)
        print("  ANALYSIS COMPLETE")
        print("="*70)
        
        total_files = len(result.all_output_files)
        print(f"\n  Total output files: {total_files}")
        
        for level, level_result in result.level_results.items():
            n_files = len(level_result.get_all_files())
            status = "OK" if level_result.success else "ERRORS"
            print(f"    {level.upper()}: {n_files} files [{status}]")
        
        if result.errors:
            print(f"\n  Errors ({len(result.errors)}):")
            for err in result.errors[:5]:  # Show first 5
                print(f"    - {err}")
            if len(result.errors) > 5:
                print(f"    ... and {len(result.errors) - 5} more")
        
        print("\n" + "="*70 + "\n")


# Backwards compatibility alias
GraphGenerator = AnalysisOrchestrator
