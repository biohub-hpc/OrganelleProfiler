"""
Level Pipeline: Runs all stages for a single analysis level.

This class orchestrates the execution of QC → Embedding → Clustering →
Differential → Summary stages for cell, guide, or gene level data.

Results chain from one stage to the next within a level.
"""

import logging
from typing import Dict, Optional, List, Type
from dataclasses import dataclass, field

from .fe_graphs_config import GraphConfig, PlotConfig, AnalysisConfig
from .core.fe_graphs_data_loader import DataContext
from .stages.fe_graphs_stage_base import BaseStage, StageResult
from .stages.fe_graphs_qc_stage import QCStage
from .stages.fe_graphs_spatial_drift_stage import SpatialDriftStage
from .stages.fe_graphs_embedding_stage import EmbeddingStage
from .stages.fe_graphs_embedding_visualization_stage import EmbeddingVisualizationStage
from .stages.fe_graphs_clustering_stage import ClusteringStage
from .stages.fe_graphs_differential_stage import DifferentialStage
from .stages.fe_graphs_volcano_enrichment_stage import VolcanoEnrichmentStage
from .stages.fe_graphs_summary_stage import SummaryStage
from .stages.fe_graphs_organelle_discrimination_stage import OrganelleDiscriminationStage
from .stages.fe_graphs_positive_controls_stage import PositiveControlsStage
from .stages.fe_graphs_cp_comparison_stage import CPComparisonStage
from .stages.fe_graphs_gene_relationships_stage import GeneRelationshipsStage
from .stages.fe_graphs_positive_controls_stage import PositiveControlsStage
from .stages.fe_graphs_cp_comparison_stage import CPComparisonStage
from .stages.fe_graphs_cp_challenge_stage import CPChallengeStage

logger = logging.getLogger(__name__)


@dataclass
class LevelResult:
    """Container for all stage results at a level."""
    level: str
    stages: Dict[str, StageResult] = field(default_factory=dict)
    success: bool = True
    
    def get_stage(self, stage_name: str) -> Optional[StageResult]:
        return self.stages.get(stage_name)
    
    def get_all_files(self) -> List:
        """Get all output files from all stages."""
        files = []
        for stage_result in self.stages.values():
            files.extend(stage_result.output_files)
        return files
    
    def get_all_metrics(self) -> Dict:
        """Get all metrics from all stages."""
        metrics = {}
        for stage_name, stage_result in self.stages.items():
            for key, val in stage_result.metrics.items():
                metrics[f"{stage_name}_{key}"] = val
        return metrics


class LevelPipeline:
    """
    Runs the full analysis pipeline for a single level.
    
    The pipeline executes stages in order:
    1. QC - Quality control
    2. Embedding - UMAP/PCA  
    3. Clustering - Unsupervised clustering
    4. Differential - NTC comparison
    5. Summary - Result aggregation
    
    Each stage receives results from prior stages for chaining.
    
    Parameters
    ----------
    data_context : DataContext
        Loaded data and paths.
    config : GraphConfig
        Main configuration.
    level : str
        Analysis level: "cell", "guide", or "gene".
    upstream_level_results : dict, optional
        Results from previous levels (for cross-level chaining).
    """
    
    # Ordered list of stages
    # Note: Positive controls runs right after embedding for early validation
    STAGES: List[tuple[str, Type[BaseStage]]] = [
        ("qc", QCStage),
        ("spatial_drift", SpatialDriftStage),  # Comprehensive radial drift (cell level)
        ("embedding", EmbeddingStage),
        ("embedding_visualization", EmbeddingVisualizationStage),  # Annotated UMAP/PCA plots
        ("positive_controls", PositiveControlsStage),  # Early: validate known clusters on UMAP
        ("clustering", ClusteringStage),
        ("differential", DifferentialStage),
        ("volcano_enrichment", VolcanoEnrichmentStage),  # Per-gene volcano + Enrichr GO
        ("organelle_discrimination", OrganelleDiscriminationStage),  # Cell/guide/gene level
        ("gene_relationships", GeneRelationshipsStage),  # Gene level only - PHATE clustering
        ("cp_comparison", CPComparisonStage),  # Cell Painting vs non-CP comparison (only when --just-cp)
        ("cp_challenge", CPChallengeStage),  # CP vs live-cell mAP head-to-head (guide level only)
        ("summary", SummaryStage),
    ]
    
    def __init__(
        self,
        data_context: DataContext,
        config: GraphConfig,
        level: str,
        upstream_level_results: Optional[Dict[str, LevelResult]] = None,
        plot_config: Optional[PlotConfig] = None,
        analysis_config: Optional[AnalysisConfig] = None,
    ):
        self.data = data_context
        self.config = config
        self.level = level
        self.upstream_levels = upstream_level_results or {}
        self.plot_config = plot_config or PlotConfig()
        self.analysis_config = analysis_config or AnalysisConfig()
    
    def run(
        self,
        stages: Optional[List[str]] = None,
        skip_stages: Optional[List[str]] = None,
    ) -> LevelResult:
        """
        Run the pipeline for this level.
        
        Parameters
        ----------
        stages : list, optional
            Specific stages to run. If None, runs all.
        skip_stages : list, optional
            Stages to skip.
        
        Returns
        -------
        LevelResult
            Results from all executed stages.
        """
        result = LevelResult(level=self.level)
        
        logger.info(f"\n{'#'*60}")
        logger.info(f"#  {self.level.upper()} LEVEL PIPELINE")
        logger.info(f"{'#'*60}\n")
        
        # Determine which stages to run
        stages_to_run = self._get_stages_to_run(stages, skip_stages)
        
        # Track upstream stage results for chaining within this level
        stage_results: Dict[str, StageResult] = {}
        
        for stage_name, StageClass in self.STAGES:
            if stage_name not in stages_to_run:
                logger.info(f"Skipping {stage_name} stage")
                continue
            
            try:
                stage = StageClass(
                    data_context=self.data,
                    config=self.config,
                    level=self.level,
                    upstream_results=stage_results,  # Chain from prior stages
                    plot_config=self.plot_config,
                    analysis_config=self.analysis_config,
                )
                
                stage_result = stage.run()
                stage_results[stage_name] = stage_result
                result.stages[stage_name] = stage_result
                
                if not stage_result.success:
                    logger.warning(f"{stage_name} stage completed with errors")
                    for err in stage_result.errors:
                        logger.error(f"  - {err}")
                
            except Exception as e:
                logger.error(f"Failed to run {stage_name} stage: {e}")
                stage_result = StageResult(success=False)
                stage_result.add_error(str(e))
                result.stages[stage_name] = stage_result
                result.success = False
                break
        
        # Log summary
        n_files = len(result.get_all_files())
        logger.info(f"\n{self.level.upper()} LEVEL COMPLETE: {n_files} output files generated")
        
        return result
    
    def _get_stages_to_run(
        self,
        stages: Optional[List[str]],
        skip_stages: Optional[List[str]],
    ) -> List[str]:
        """Determine which stages to run."""
        all_stages = [name for name, _ in self.STAGES]
        
        if stages is not None:
            stages_to_run = [s for s in all_stages if s in stages]
        else:
            stages_to_run = all_stages
        
        if skip_stages is not None:
            stages_to_run = [s for s in stages_to_run if s not in skip_stages]
        
        return stages_to_run
