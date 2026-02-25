"""
Base analyzer class for feature graph generation.

Provides common functionality shared across all analysis types:
- NTC identification
- Feature column extraction
- Output directory management
- Standard result formatting
"""

import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Any
import logging

from ..fe_graphs_config import (
    GraphConfig, 
    PlotConfig, 
    AnalysisConfig,
    METADATA_EXCLUDE_PATTERNS,
    NTC_PATTERNS,
)
from ..core.fe_graphs_data_loader import DataContext

logger = logging.getLogger(__name__)


@dataclass
class AnalysisResult:
    """
    Container for analysis results.
    
    Attributes
    ----------
    success : bool
        Whether the analysis completed successfully.
    output_files : list
        List of output file paths created.
    data : dict
        Any data to pass to downstream analyses.
    errors : list
        List of error messages if any.
    """
    success: bool = True
    output_files: List[Path] = field(default_factory=list)
    data: Dict[str, Any] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)
    
    def add_file(self, path: Path) -> None:
        """Register an output file."""
        self.output_files.append(path)
    
    def add_error(self, error: str) -> None:
        """Register an error."""
        self.errors.append(error)
        self.success = False


class BaseAnalyzer(ABC):
    """
    Abstract base class for all analysis types.
    
    Provides common functionality shared across analyzers:
    - NTC identification (replaces 8+ duplicate implementations)
    - Feature column extraction (replaces duplicated logic)
    - Output directory management
    - Consistent logging
    
    Subclasses must implement:
    - analysis_name: Unique identifier for the analysis
    - run(): Execute the analysis and return results
    
    Parameters
    ----------
    data_context : DataContext
        Container with loaded data and paths.
    config : GraphConfig
        Main configuration settings.
    plot_config : PlotConfig, optional
        Plotting configuration.
    analysis_config : AnalysisConfig, optional
        Analysis algorithm configuration.
    
    Examples
    --------
    >>> class MyAnalyzer(BaseAnalyzer):
    ...     @property
    ...     def analysis_name(self):
    ...         return "my_analysis"
    ...     
    ...     def run(self):
    ...         ntc_mask = self.get_ntc_mask(self.cell_df)
    ...         # ... do analysis ...
    ...         return AnalysisResult()
    """
    
    def __init__(
        self,
        data_context: DataContext,
        config: GraphConfig,
        plot_config: Optional[PlotConfig] = None,
        analysis_config: Optional[AnalysisConfig] = None,
    ):
        self.data = data_context
        self.config = config
        self.plot_config = plot_config or PlotConfig()
        self.analysis_config = analysis_config or AnalysisConfig()
        
        # Create output directory for this analysis
        self._output_dir = None
    
    @property
    @abstractmethod
    def analysis_name(self) -> str:
        """Unique name for this analysis type."""
        pass
    
    @abstractmethod
    def run(self) -> AnalysisResult:
        """
        Execute the analysis and return results.
        
        Returns
        -------
        AnalysisResult
            Container with output files and any data to pass downstream.
        """
        pass
    
    @property
    def output_dir(self) -> Path:
        """Output directory for this analysis."""
        if self._output_dir is None:
            self._output_dir = self.data.graph_output_path / self.analysis_name
            self._output_dir.mkdir(parents=True, exist_ok=True)
        return self._output_dir
    
    @property
    def cell_df(self) -> pd.DataFrame:
        """Convenience accessor for cell DataFrame."""
        return self.data.cell_df
    
    @property
    def guide_df(self) -> pd.DataFrame:
        """Convenience accessor for guide DataFrame."""
        return self.data.guide_df
    
    @property
    def gene_df(self) -> pd.DataFrame:
        """Convenience accessor for gene DataFrame."""
        return self.data.gene_df
    
    # --- Shared utility methods (replaces duplicated code) ---
    
    def get_ntc_mask(self, df: pd.DataFrame, gene_col: str = "gene_name") -> pd.Series:
        """
        Identify non-targeting control (NTC) cells/guides/genes.
        
        This consolidates 8+ duplicate implementations from the original fe_graphs.py.
        
        Parameters
        ----------
        df : pd.DataFrame
            DataFrame with gene information.
        gene_col : str
            Column containing gene names. Defaults to "gene_name".
            
        Returns
        -------
        pd.Series
            Boolean mask where True indicates NTC.
        """
        if gene_col not in df.columns:
            logger.warning(f"Column '{gene_col}' not found for NTC identification")
            return pd.Series([False] * len(df), index=df.index)
        
        pattern = "|".join(NTC_PATTERNS)
        return df[gene_col].astype(str).str.contains(pattern, case=False, regex=True)
    
    def get_feature_columns(
        self, 
        df: pd.DataFrame,
        exclude_patterns: Optional[List[str]] = None,
    ) -> List[str]:
        """
        Get feature columns by excluding metadata columns.
        
        This consolidates duplicated feature column extraction logic.
        
        Parameters
        ----------
        df : pd.DataFrame
            DataFrame with features and metadata.
        exclude_patterns : list, optional
            Additional patterns to exclude. If None, uses defaults.
            
        Returns
        -------
        list
            List of feature column names.
        """
        patterns = exclude_patterns or METADATA_EXCLUDE_PATTERNS
        
        feature_cols = [
            col for col in df.select_dtypes(include=np.number).columns
            if not any(pattern in col for pattern in patterns)
        ]
        return feature_cols
    
    def prepare_features(
        self,
        df: pd.DataFrame,
        feature_cols: Optional[List[str]] = None,
        remove_duplicates: bool = True,
        remove_low_variance: bool = True,
        variance_threshold: Optional[float] = None,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Prepare features for analysis by cleaning and filtering.
        
        Parameters
        ----------
        df : pd.DataFrame
            DataFrame with features and metadata.
        feature_cols : list, optional
            Specific feature columns to use. If None, auto-detects.
        remove_duplicates : bool
            Whether to remove duplicate rows.
        remove_low_variance : bool
            Whether to remove low-variance features.
        variance_threshold : float, optional
            Threshold for low variance removal.
            
        Returns
        -------
        tuple[pd.DataFrame, pd.DataFrame]
            (features_df, filtered_cell_df) - features and corresponding metadata.
        """
        if feature_cols is None:
            feature_cols = self.get_feature_columns(df)
        
        features = df[feature_cols].copy()
        features.fillna(0, inplace=True)
        
        # Track original index for filtering metadata
        original_idx = features.index.copy()
        
        # Remove duplicates
        if remove_duplicates:
            n_original = len(features)
            features.drop_duplicates(inplace=True)
            if len(features) < n_original:
                logger.info(f"Removed {n_original - len(features)} duplicate rows")
        
        # Remove low-variance features
        if remove_low_variance:
            variances = features.var()
            
            # Use percentile-based threshold if not explicitly provided
            if variance_threshold is None:
                # Adaptive threshold: 5th percentile, minimum 1e-8
                threshold = max(np.percentile(variances, 5.0), 1e-8) if len(variances) > 0 else 1e-8
            else:
                threshold = variance_threshold
            
            low_var_mask = variances < threshold
            low_var_cols = features.columns[low_var_mask]
            
            if len(low_var_cols) > 0:
                pct_removed = len(low_var_cols) / len(features.columns) * 100
                logger.info(f"Removing {len(low_var_cols)} low-variance features ({pct_removed:.1f}%, threshold={threshold:.2e})")
                features = features.drop(columns=low_var_cols)
        
        # Filter metadata to match
        filtered_df = df.loc[features.index].copy()
        
        return features, filtered_df
    
    def add_gene_group_column(
        self, 
        df: pd.DataFrame, 
        gene_col: str = "gene_name",
        output_col: str = "gene_group",
    ) -> pd.DataFrame:
        """
        Add a column distinguishing NTC from perturbed cells.
        
        Parameters
        ----------
        df : pd.DataFrame
            DataFrame to modify.
        gene_col : str
            Column containing gene names.
        output_col : str
            Name for the output column.
            
        Returns
        -------
        pd.DataFrame
            DataFrame with added gene_group column.
        """
        df = df.copy()
        ntc_mask = self.get_ntc_mask(df, gene_col)
        df[output_col] = np.where(ntc_mask, "NTC", "Perturbed")
        return df
    
    def calculate_radial_positions(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate well and tile radial positions for cells.
        
        Parameters
        ----------
        df : pd.DataFrame
            DataFrame with coordinate columns.
            
        Returns
        -------
        pd.DataFrame
            DataFrame with added radial position columns.
        """
        df = df.copy()
        
        # Well radial position
        df = self._calculate_well_radial_pos(df)
        
        # Tile radial position
        df = self._calculate_tile_radial_pos(df)
        
        return df
    
    def _calculate_well_radial_pos(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate radial position within well."""
        pos_cols = ["x_global_pheno", "y_global_pheno", "well"]
        
        if not all(col in df.columns for col in pos_cols):
            return df
        
        # Ensure numeric
        for col in ["x_global_pheno", "y_global_pheno"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        
        radial_pos = pd.Series(index=df.index, dtype=float, name="well_radial_pos")
        valid_idx = df[pos_cols].dropna().index
        
        if not valid_idx.empty:
            pos_df = df.loc[valid_idx].copy()
            
            # Calculate bounding box center for each well
            well_bounds = pos_df.groupby("well", observed=True).agg(
                x_min=("x_global_pheno", "min"),
                x_max=("x_global_pheno", "max"),
                y_min=("y_global_pheno", "min"),
                y_max=("y_global_pheno", "max"),
            )
            well_bounds["center_x"] = (well_bounds["x_min"] + well_bounds["x_max"]) / 2
            well_bounds["center_y"] = (well_bounds["y_min"] + well_bounds["y_max"]) / 2
            
            # Calculate distance from center
            centers_x = pos_df["well"].map(well_bounds["center_x"]).astype(float).values
            centers_y = pos_df["well"].map(well_bounds["center_y"]).astype(float).values
            
            radial_values = np.sqrt(
                (pos_df["x_global_pheno"].values - centers_x) ** 2 +
                (pos_df["y_global_pheno"].values - centers_y) ** 2
            )
            radial_pos.loc[valid_idx] = radial_values
        
        df["well_radial_pos"] = radial_pos
        return df
    
    def _calculate_tile_radial_pos(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate radial position within tile."""
        pos_cols = ["x_local_pheno", "y_local_pheno", "tile_pheno"]
        
        if not all(col in df.columns for col in pos_cols):
            return df
        
        # Ensure numeric
        for col in pos_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        
        radial_pos = pd.Series(index=df.index, dtype=float, name="tile_radial_pos")
        valid_idx = df[pos_cols].dropna().index
        
        if not valid_idx.empty:
            pos_df = df.loc[valid_idx].copy()
            
            # Calculate bounding box center for each tile
            tile_bounds = pos_df.groupby("tile_pheno", observed=True).agg(
                x_min=("x_local_pheno", "min"),
                x_max=("x_local_pheno", "max"),
                y_min=("y_local_pheno", "min"),
                y_max=("y_local_pheno", "max"),
            )
            tile_bounds["center_x"] = (tile_bounds["x_min"] + tile_bounds["x_max"]) / 2
            tile_bounds["center_y"] = (tile_bounds["y_min"] + tile_bounds["y_max"]) / 2
            
            # Calculate distance from center
            centers_x = pos_df["tile_pheno"].map(tile_bounds["center_x"]).astype(float).values
            centers_y = pos_df["tile_pheno"].map(tile_bounds["center_y"]).astype(float).values
            
            radial_values = np.sqrt(
                (pos_df["x_local_pheno"].values - centers_x) ** 2 +
                (pos_df["y_local_pheno"].values - centers_y) ** 2
            )
            radial_pos.loc[valid_idx] = radial_values
        
        df["tile_radial_pos"] = radial_pos
        return df
    
    def group_features_by_organelle(
        self, feature_cols: List[str]
    ) -> Dict[str, List[str]]:
        """
        Group feature columns by organelle prefix.
        
        Parameters
        ----------
        feature_cols : list
            List of feature column names.
            
        Returns
        -------
        dict
            Dictionary mapping organelle name to list of feature columns.
        """
        organelle_features = {}
        
        for col in feature_cols:
            parts = col.split("_")
            if parts[0] == "network":
                if len(parts) > 2:
                    organelle = parts[1]
                else:
                    continue
            else:
                if len(parts) > 1:
                    organelle = parts[0]
                else:
                    continue
            
            if organelle not in organelle_features:
                organelle_features[organelle] = []
            organelle_features[organelle].append(col)
        
        return organelle_features
    
    def log_start(self, message: Optional[str] = None) -> None:
        """Log analysis start."""
        msg = message or f"Starting {self.analysis_name} analysis..."
        logger.info(f"\n--- {msg} ---")
    
    def log_complete(self, message: Optional[str] = None) -> None:
        """Log analysis completion."""
        msg = message or f"{self.analysis_name} analysis complete."
        logger.info(f"--- {msg} ---")
