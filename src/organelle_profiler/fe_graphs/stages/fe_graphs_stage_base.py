"""
Base class for analysis stages.

Each stage runs a specific phase of analysis and can be applied to any level.
Stages receive upstream results for chaining.
"""

import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Any
import logging

from ..fe_graphs_config import GraphConfig, PlotConfig, AnalysisConfig, NTC_PATTERNS, METADATA_EXCLUDE_PATTERNS
from ..core.fe_graphs_data_loader import DataContext

logger = logging.getLogger(__name__)


@dataclass
class StageResult:
    """Container for stage results."""
    success: bool = True
    output_files: List[Path] = field(default_factory=list)
    data: Dict[str, Any] = field(default_factory=dict)
    metrics: Dict[str, Any] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)
    
    def add_file(self, path: Path) -> None:
        self.output_files.append(path)
    
    def add_error(self, error: str) -> None:
        self.errors.append(error)
        self.success = False
    
    def add_metric(self, key: str, value: Any) -> None:
        self.metrics[key] = value


class BaseStage(ABC):
    """
    Abstract base class for analysis stages.
    
    Each stage:
    1. Operates on a specific level (cell/guide/gene)
    2. Can receive upstream results for chaining
    3. Produces standardized outputs
    
    Parameters
    ----------
    data_context : DataContext
        Loaded data and paths.
    config : GraphConfig
        Main configuration.
    level : str
        Analysis level: "cell", "guide", or "gene".
    upstream_results : dict, optional
        Results from previous stages or levels for chaining.
    """
    
    # Stage number for output directory naming
    STAGE_NUMBER: int = 0
    STAGE_NAME: str = "base"
    
    def __init__(
        self,
        data_context: DataContext,
        config: GraphConfig,
        level: str,
        upstream_results: Optional[Dict[str, StageResult]] = None,
        plot_config: Optional[PlotConfig] = None,
        analysis_config: Optional[AnalysisConfig] = None,
    ):
        self.data = data_context
        self.config = config
        self.level = level
        self.upstream = upstream_results or {}
        self.plot_config = plot_config or PlotConfig()
        self.analysis_config = analysis_config or AnalysisConfig()
        
        # Output directory
        self._output_dir = None
    
    @property
    def output_dir(self) -> Path:
        """Output directory for this stage at this level."""
        if self._output_dir is None:
            level_num = {"cell": 1, "guide": 2, "gene": 3}.get(self.level, 0)
            level_dir = self.data.graph_output_path / f"{level_num}_{self.level}_level"
            self._output_dir = level_dir / f"{self.STAGE_NUMBER}_{self.STAGE_NAME}"
            self._output_dir.mkdir(parents=True, exist_ok=True)
        return self._output_dir
    
    @property
    def df(self) -> pd.DataFrame:
        """Get the DataFrame for this level."""
        if self.level == "cell":
            return self.data.cell_df
        elif self.level == "guide":
            return self.data.guide_df
        elif self.level == "gene":
            return self.data.gene_df
        else:
            raise ValueError(f"Unknown level: {self.level}")
    
    @property
    def level_label(self) -> str:
        """Human-readable level label."""
        return self.level.capitalize()
    
    @property
    def item_name(self) -> str:
        """Name for items at this level (cell/guide/gene)."""
        return self.level
    
    @property
    def n_items(self) -> int:
        """Number of items at this level."""
        return len(self.df)
    
    @abstractmethod
    def run(self) -> StageResult:
        """Execute the stage and return results."""
        pass
    
    # --- Shared utilities ---
    
    def get_ntc_mask(self, df: Optional[pd.DataFrame] = None, gene_col: str = "gene_name") -> pd.Series:
        """
        Identify NTC items using multiple detection methods.
        
        Checks:
        1. gene_id == -1 or NCBI_ID == -1 (if columns exist)
        2. gene_name is None or empty string
        3. String patterns: "ntc", "non-targeting", "^0$" (case-insensitive)
        """
        df = df if df is not None else self.df
        if gene_col not in df.columns:
            return pd.Series([False] * len(df), index=df.index)
        
        ntc_mask = pd.Series(False, index=df.index)
        
        # Method 1: Check for gene_id == -1 or NCBI_ID == -1
        if "gene_id" in df.columns:
            ntc_mask |= (df["gene_id"] == -1)
        if "NCBI_ID" in df.columns:
            ntc_mask |= (df["NCBI_ID"] == -1)
        
        # Method 2: Check for None or empty gene_name
        none_or_empty = df[gene_col].isna() | (df[gene_col].astype(str).str.strip() == "") | (df[gene_col].astype(str) == "None")
        ntc_mask |= none_or_empty
        
        # Method 3: String pattern matching
        pattern = "|".join(NTC_PATTERNS)
        pattern_match = df[gene_col].astype(str).str.contains(pattern, case=False, regex=True, na=False)
        ntc_mask |= pattern_match
        
        return ntc_mask
    
    def get_feature_columns(self, df: Optional[pd.DataFrame] = None) -> List[str]:
        """Get feature columns (exclude metadata)."""
        df = df if df is not None else self.df
        return [
            col for col in df.select_dtypes(include=np.number).columns
            if not any(p in col for p in METADATA_EXCLUDE_PATTERNS)
        ]
    
    def get_features(self, df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """Extract feature matrix from DataFrame."""
        df = df if df is not None else self.df
        feature_cols = self.get_feature_columns(df)
        features = df[feature_cols].copy()
        features.fillna(0, inplace=True)
        return features
    
    def group_features_by_organelle(self, feature_cols: List[str] = None) -> Dict[str, List[str]]:
        """
        Get feature columns grouped by organelle.
        
        Uses the centralized organelle_groups from DataContext (single source of truth).
        Automatically uses level-specific mappings (guide/gene features have _mean, _std suffixes).
        If feature_cols is provided, filters to only include those columns.
        
        Parameters
        ----------
        feature_cols : list, optional
            If provided, filter groups to only include these columns.
            If None, returns all organelle groups from DataContext for this level.
            
        Returns
        -------
        dict
            Mapping of organelle name -> list of feature columns.
        """
        # Use level-specific organelle groups if available
        if hasattr(self.data, 'organelle_groups_all_levels') and self.data.organelle_groups_all_levels:
            if self.level in self.data.organelle_groups_all_levels:
                all_groups = self.data.organelle_groups_all_levels[self.level]
                logger.debug(f"Using level-specific organelle groups for {self.level} level: {len(all_groups)} groups")
            else:
                # Fallback to cell-level groups (backward compatibility)
                all_groups = self.data.organelle_groups
                logger.debug(f"No {self.level}-specific organelle groups found, using cell-level groups")
        else:
            # Backward compatibility: use original organelle_groups
            all_groups = self.data.organelle_groups
        
        if not all_groups:
            logger.warning(f"No organelle groups found in DataContext for {self.level} level! Check data loading.")
            return {}
        
        if feature_cols is None:
            return all_groups
        
        # Filter to only include requested columns
        feature_set = set(feature_cols)
        filtered_groups = {}
        for organelle, cols in all_groups.items():
            filtered = [c for c in cols if c in feature_set]
            if filtered:
                filtered_groups[organelle] = filtered
        
        # Sanity check
        if len(filtered_groups) > 50:
            logger.warning(
                f"Found {len(filtered_groups)} organelle groups - seems too high! "
                f"Expected ~20 organelles."
            )
        
        return filtered_groups
    
    def log_start(self, message: Optional[str] = None) -> None:
        msg = message or f"Running {self.STAGE_NAME} stage for {self.level} level"
        logger.info(f"\n{'='*60}")
        logger.info(f"  {msg}")
        logger.info(f"{'='*60}")
    
    def log_complete(self, result: StageResult) -> None:
        logger.info(f"  Completed: {len(result.output_files)} files generated")
        if result.errors:
            for e in result.errors:
                logger.error(f"  Error: {e}")
