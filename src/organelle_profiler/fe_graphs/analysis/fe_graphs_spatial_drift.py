"""
Spatial drift analysis.

Analyzes how feature values drift with radial distance from well/tile centers.
Useful for identifying edge effects and systematic spatial biases.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional, Dict, List, Tuple
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from statsmodels.nonparametric.smoothers_lowess import lowess
from tqdm import tqdm
import logging

from .fe_graphs_base import BaseAnalyzer, AnalysisResult
from ..plotting.fe_graphs_utils import save_figure

logger = logging.getLogger(__name__)


class SpatialDriftAnalyzer(BaseAnalyzer):
    """
    Spatial drift analysis for detecting edge effects.
    
    Generates:
    - Radial drift plots showing feature deviation vs distance
    - Tile-level heatmaps
    - Well-level heatmaps
    - Summary statistics
    
    Parameters
    ----------
    data_context : DataContext
        Container with loaded data and paths.
    config : GraphConfig
        Main configuration settings.
    feature_subset_name : str
        Name for this feature subset (e.g., "all_features", "organelle_mito").
    features : pd.DataFrame, optional
        Pre-computed feature matrix. If None, uses all cell features.
    """
    
    def __init__(
        self,
        *args,
        feature_subset_name: str = "all_features",
        features: Optional[pd.DataFrame] = None,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.feature_subset_name = feature_subset_name
        self._features = features
        self._analysis_name = f"radial_drift/{feature_subset_name}"
    
    @property
    def analysis_name(self) -> str:
        return self._analysis_name
    
    def run(self) -> AnalysisResult:
        """Execute spatial drift analysis."""
        self.log_start(f"Generating Radial Drift Plots for '{self.feature_subset_name}'")
        result = AnalysisResult()
        
        # Get features
        if self._features is not None:
            features = self._features
        else:
            feature_cols = self.get_feature_columns(self.cell_df)
            features = self.cell_df[feature_cols].copy()
            features.fillna(0, inplace=True)
        
        if features.empty:
            result.add_error("No features available for drift analysis")
            return result
        
        # Generate drift plots
        well_cutoff = self._plot_radial_drift("well_radial_pos", features, result)
        tile_cutoff = self._plot_radial_drift("tile_radial_pos", features, result)
        
        # Store cutoff results
        result.data["well_percent_cutoff"] = well_cutoff
        result.data["tile_percent_cutoff"] = tile_cutoff
        
        # Generate heatmaps
        self._plot_average_tile_heatmap(features, result)
        self._plot_well_heatmap(features, result)
        
        self.log_complete()
        return result
    
    def _plot_radial_drift(
        self,
        pos_type: str,
        features: pd.DataFrame,
        result: AnalysisResult,
    ) -> Optional[float]:
        """Generate radial drift plot for well or tile position."""
        logger.info(f"Generating radial drift plot for {pos_type}...")
        
        # Calculate positions and drift score
        analysis_df = self.cell_df.copy()
        analysis_df = self.calculate_radial_positions(analysis_df)
        
        if pos_type not in analysis_df.columns or analysis_df[pos_type].isna().all():
            logger.warning(f"Skipping {pos_type}: column not found or all NaN")
            return None
        
        # Calculate feature drift score (mean absolute scaled feature value)
        scaled_features = StandardScaler().fit_transform(features)
        feature_drift_score = np.mean(np.abs(scaled_features), axis=1)
        analysis_df["feature_drift_score"] = feature_drift_score
        
        # Prepare data
        plot_data = analysis_df[[pos_type, "feature_drift_score"]].dropna()
        if len(plot_data) < 100:
            logger.warning(f"Skipping {pos_type}: insufficient data ({len(plot_data)})")
            return None
        
        # Bin by radial distance
        num_bins = 50
        plot_data["radial_bin"] = pd.cut(plot_data[pos_type], bins=num_bins)
        
        binned = plot_data.groupby("radial_bin", observed=False).agg(
            mean_drift=("feature_drift_score", "mean"),
            cell_count=("feature_drift_score", "count"),
        ).reset_index()
        
        # Calculate cell density
        binned["r_inner"] = [b.left for b in binned["radial_bin"]]
        binned["r_outer"] = [b.right for b in binned["radial_bin"]]
        bin_area = np.pi * (binned["r_outer"]**2 - binned["r_inner"]**2)
        bin_area = bin_area.replace(0, np.nan)
        binned["cell_density"] = binned["cell_count"] / bin_area
        binned["radial_midpoint"] = [b.mid for b in binned["radial_bin"]]
        binned.dropna(subset=["mean_drift", "cell_density", "radial_midpoint"], inplace=True)
        
        if len(binned) < 5:
            logger.warning(f"Skipping {pos_type}: not enough valid bins")
            return None
        
        # Normalize drift relative to center
        center_drift = binned.sort_values("radial_midpoint").iloc[0]["mean_drift"]
        if center_drift > 1e-9:
            normalized_drift = binned["mean_drift"] / center_drift
            y_label = "Normalized Feature Drift (Relative to Center)"
        else:
            normalized_drift = binned["mean_drift"]
            y_label = "Mean Absolute Scaled Feature Value (Drift Score)"
        
        x_data = binned["radial_midpoint"].values
        y_data = normalized_drift.values
        
        # Find breakpoint
        breakpoint_x, models = self._find_breakpoint(x_data, y_data)
        
        # Calculate cell counts for annotation
        cells_before = plot_data[plot_data[pos_type] <= breakpoint_x].shape[0]
        cells_after = plot_data[plot_data[pos_type] > breakpoint_x].shape[0]
        total_cells = cells_before + cells_after
        percent_after = 100 * cells_after / total_cells if total_cells > 0 else 0
        
        # Generate plots (density and count versions)
        for plot_type in ["density", "count"]:
            fig, ax1 = plt.subplots(figsize=(14, 8))
            
            # Plot drift line
            ax1.plot(x_data, y_data, "o-", alpha=0.6, label="Mean Feature Drift", zorder=10)
            
            # Plot regression lines
            if models is not None:
                for segment_x, segment_model, color in models:
                    ax1.plot(
                        segment_x,
                        segment_model.predict(segment_x.reshape(-1, 1)),
                        color=color, linestyle="--", linewidth=2.5, zorder=20,
                    )
            
            # Breakpoint line
            ax1.axvline(breakpoint_x, color="black", linestyle="--", 
                       label=f"Inflection at {breakpoint_x:.2f}", zorder=15)
            
            ax1.set_xlabel(f'Radial Distance from {pos_type.split("_")[0].capitalize()} Center')
            ax1.set_ylabel(y_label)
            ax1.grid(True, which="both", linestyle="--", linewidth=0.5)
            
            # Secondary y-axis for density/count
            ax2 = ax1.twinx()
            bar_width = (binned["radial_midpoint"].max() / num_bins) * 0.9
            
            if plot_type == "density":
                ax2.bar(binned["radial_midpoint"], binned["cell_density"],
                       width=bar_width, alpha=0.2, color="gray", label="Cell Density")
                ax2.set_ylabel("Cell Density (Cells / Pixel²)", color="gray")
                title = f"Feature Drift vs. {pos_type.split('_')[0].capitalize()} Position ({self.feature_subset_name}, vs. Density)"
                suffix = "density"
            else:
                ax2.bar(binned["radial_midpoint"], binned["cell_count"],
                       width=bar_width, alpha=0.2, color="gray", label="Cell Count")
                ax2.set_ylabel("Cell Count per Bin", color="gray")
                title = f"Feature Drift vs. {pos_type.split('_')[0].capitalize()} Position ({self.feature_subset_name}, vs. Count)"
                suffix = "count"
            
            ax2.tick_params(axis="y", labelcolor="gray")
            ax1.set_zorder(ax2.get_zorder() + 1)
            ax1.patch.set_visible(False)
            
            fig.suptitle(title, fontsize=16)
            
            # Legend with cell count info
            handles, labels = ax1.get_legend_handles_labels()
            handles2, labels2 = ax2.get_legend_handles_labels()
            
            percent_before = 100 - percent_after
            info_text = (
                f"Cells Before Inflection: {cells_before:,} ({percent_before:.1f}%)\n"
                f"Cells After Inflection: {cells_after:,} ({percent_after:.1f}%)"
            )
            
            leg = fig.legend(handles=handles + handles2, labels=labels + labels2,
                           loc="upper right", bbox_to_anchor=(0.9, 0.9))
            leg.set_title(info_text, prop={"size": "small", "weight": "bold"})
            
            plt.tight_layout(rect=[0, 0, 1, 0.96])
            
            filename = f"{self.feature_subset_name}_{pos_type.replace('_pos', '')}_feature_drift_{suffix}.png"
            path = save_figure(fig, self.output_dir / filename, dpi=300)
            result.add_file(path)
        
        return percent_after
    
    def _find_breakpoint(
        self, x_data: np.ndarray, y_data: np.ndarray
    ) -> Tuple[float, Optional[List]]:
        """Find inflection point using segmented regression."""
        drift_method = self.config.drift_method
        
        if drift_method == "three_segment_regression":
            return self._three_segment_regression(x_data, y_data)
        elif drift_method == "segmented_regression":
            return self._two_segment_regression(x_data, y_data)
        elif drift_method == "second_derivative":
            return self._second_derivative_method(x_data, y_data)
        else:
            return x_data.mean(), None
    
    def _three_segment_regression(
        self, x_data: np.ndarray, y_data: np.ndarray
    ) -> Tuple[float, Optional[List]]:
        """Find two breakpoints using three-segment regression."""
        min_error = np.inf
        best_breakpoints = (-1, -1)
        n = len(x_data)
        
        for i in range(2, n - 4):
            for j in range(i + 2, n - 2):
                x1, y1 = x_data[:i], y_data[:i]
                x2, y2 = x_data[i:j], y_data[i:j]
                x3, y3 = x_data[j:], y_data[j:]
                
                model1 = LinearRegression().fit(x1.reshape(-1, 1), y1)
                model2 = LinearRegression().fit(x2.reshape(-1, 1), y2)
                model3 = LinearRegression().fit(x3.reshape(-1, 1), y3)
                
                error = (
                    np.sum((y1 - model1.predict(x1.reshape(-1, 1)))**2) +
                    np.sum((y2 - model2.predict(x2.reshape(-1, 1)))**2) +
                    np.sum((y3 - model3.predict(x3.reshape(-1, 1)))**2)
                )
                
                if error < min_error:
                    min_error = error
                    best_breakpoints = (i, j)
        
        if best_breakpoints[0] == -1:
            return x_data.mean(), None
        
        bp1, bp2 = best_breakpoints
        
        # Fit models for plotting
        x1, y1 = x_data[:bp1], y_data[:bp1]
        x2, y2 = x_data[bp1:bp2], y_data[bp1:bp2]
        x3, y3 = x_data[bp2:], y_data[bp2:]
        
        model1 = LinearRegression().fit(x1.reshape(-1, 1), y1)
        model2 = LinearRegression().fit(x2.reshape(-1, 1), y2)
        model3 = LinearRegression().fit(x3.reshape(-1, 1), y3)
        
        models = [
            (x1, model1, "red"),
            (x2, model2, "cyan"),
            (x3, model3, "magenta"),
        ]
        
        # Choose inflection point based on position
        x_midpoint = (x_data[0] + x_data[-1]) / 2
        if x_data[bp1] < x_midpoint:
            breakpoint_x = x_data[bp2]
        else:
            breakpoint_x = x_data[bp1]
        
        return breakpoint_x, models
    
    def _two_segment_regression(
        self, x_data: np.ndarray, y_data: np.ndarray
    ) -> Tuple[float, Optional[List]]:
        """Find one breakpoint using two-segment regression."""
        min_error = np.inf
        best_idx = -1
        
        for i in range(2, len(x_data) - 2):
            x1, y1 = x_data[:i], y_data[:i]
            x2, y2 = x_data[i:], y_data[i:]
            
            model1 = LinearRegression().fit(x1.reshape(-1, 1), y1)
            model2 = LinearRegression().fit(x2.reshape(-1, 1), y2)
            
            error = (
                np.sum((y1 - model1.predict(x1.reshape(-1, 1)))**2) +
                np.sum((y2 - model2.predict(x2.reshape(-1, 1)))**2)
            )
            
            if error < min_error:
                min_error = error
                best_idx = i
        
        if best_idx == -1:
            return x_data.mean(), None
        
        x1, y1 = x_data[:best_idx], y_data[:best_idx]
        x2, y2 = x_data[best_idx:], y_data[best_idx:]
        
        model1 = LinearRegression().fit(x1.reshape(-1, 1), y1)
        model2 = LinearRegression().fit(x2.reshape(-1, 1), y2)
        
        models = [
            (x1, model1, "red"),
            (x2, model2, "cyan"),
        ]
        
        return x_data[best_idx], models
    
    def _second_derivative_method(
        self, x_data: np.ndarray, y_data: np.ndarray
    ) -> Tuple[float, Optional[List]]:
        """Find inflection point using second derivative of smoothed curve."""
        smoothed = lowess(y_data, x_data, frac=0.5)
        second_derivative = np.gradient(np.gradient(smoothed[:, 1], smoothed[:, 0]), smoothed[:, 0])
        bp_idx = np.argmax(second_derivative)
        
        return smoothed[bp_idx, 0], None
    
    def _plot_average_tile_heatmap(
        self, features: pd.DataFrame, result: AnalysisResult
    ) -> None:
        """Generate average tile drift heatmap."""
        logger.info("Generating average tile drift heatmap...")
        
        if "x_local_pheno" not in self.cell_df.columns or "y_local_pheno" not in self.cell_df.columns:
            return
        
        # Calculate drift score
        scaled = StandardScaler().fit_transform(features)
        drift_score = np.mean(np.abs(scaled), axis=1)
        
        analysis_df = self.cell_df.copy()
        analysis_df["feature_drift_score"] = drift_score
        analysis_df.dropna(subset=["x_local_pheno", "y_local_pheno"], inplace=True)
        
        if analysis_df.empty:
            return
        
        # Bin into grid
        bin_size = 25
        max_x = int(analysis_df["x_local_pheno"].max())
        max_y = int(analysis_df["y_local_pheno"].max())
        
        x_bins = np.arange(0, max_x + bin_size, bin_size)
        y_bins = np.arange(0, max_y + bin_size, bin_size)
        
        analysis_df["x_bin"] = pd.cut(analysis_df["x_local_pheno"], bins=x_bins, labels=False, right=False)
        analysis_df["y_bin"] = pd.cut(analysis_df["y_local_pheno"], bins=y_bins, labels=False, right=False)
        
        binned = analysis_df.groupby(["y_bin", "x_bin"])["feature_drift_score"].mean().unstack()
        
        # Normalize by center
        if not binned.empty:
            center_y = binned.shape[0] // 2
            center_x = binned.shape[1] // 2
            center_val = binned.iloc[center_y, center_x]
            
            if pd.notna(center_val) and center_val > 1e-9:
                binned = binned / center_val
                cbar_label = "Normalized Feature Drift"
            else:
                cbar_label = "Mean Feature Drift Score"
        
        # Plot
        fig, ax = plt.subplots(figsize=(10, 10))
        
        all_scores = binned.values.flatten()
        all_scores = all_scores[~np.isnan(all_scores)]
        
        if len(all_scores) == 0:
            plt.close(fig)
            return
        
        vmin = np.percentile(all_scores, 5)
        vmax = np.percentile(all_scores, 95)
        
        im = ax.imshow(binned, cmap="viridis", origin="lower", vmin=vmin, vmax=vmax)
        
        ax.set_title("Average Feature Drift Across All Tiles", fontsize=16)
        ax.set_xlabel(f"Tile X-position (in {bin_size}px bins)")
        ax.set_ylabel(f"Tile Y-position (in {bin_size}px bins)")
        
        cbar = plt.colorbar(im, fraction=0.046, pad=0.04)
        cbar.set_label(cbar_label)
        
        path = save_figure(
            fig, self.output_dir / f"{self.feature_subset_name}_average_tile_drift_heatmap.png",
            dpi=300
        )
        result.add_file(path)
    
    def _plot_well_heatmap(self, features: pd.DataFrame, result: AnalysisResult) -> None:
        """Generate per-well drift heatmap."""
        required_cols = ["well", "x_global_pheno", "y_global_pheno"]
        if not all(col in self.cell_df.columns for col in required_cols):
            return
        
        logger.info("Generating well drift heatmap...")
        
        # Calculate drift score
        scaled = StandardScaler().fit_transform(features)
        drift_score = np.mean(np.abs(scaled), axis=1)
        
        analysis_df = self.cell_df.copy()
        analysis_df["feature_drift_score"] = drift_score
        
        try:
            from ops_utils.io.tiling import split_into_tiles
        except ImportError:
            logger.warning("Cannot generate well heatmap - split_into_tiles not available")
            return
        
        wells = sorted(analysis_df["well"].unique())
        grid_size = 30
        
        plate_drift_scores = {}
        well_indices = {}
        
        for well in wells:
            well_df = analysis_df[analysis_df["well"] == well].copy()
            if well_df.empty:
                continue
            
            shape = (
                int(well_df["y_global_pheno"].max()) + 1,
                int(well_df["x_global_pheno"].max()) + 1,
            )
            tile_list, indx = split_into_tiles(shape, grid_size, 0)
            
            well_scores = []
            for tile in tile_list:
                row_min, row_max, col_min, col_max = tile
                
                tile_cells = well_df[
                    (well_df["y_global_pheno"] >= row_min) &
                    (well_df["y_global_pheno"] < row_max) &
                    (well_df["x_global_pheno"] >= col_min) &
                    (well_df["x_global_pheno"] < col_max)
                ]
                
                mean_drift = tile_cells["feature_drift_score"].mean() if not tile_cells.empty else np.nan
                well_scores.append(mean_drift)
            
            plate_drift_scores[well] = well_scores
            well_indices[well] = indx
        
        # Calculate normalization factor from center tiles
        center_scores = []
        center_coord = (grid_size // 2, grid_size // 2)
        
        for well in wells:
            indx = well_indices.get(well)
            scores = plate_drift_scores.get(well)
            if not indx or not scores:
                continue
            
            distances = [np.sqrt((i - center_coord[0])**2 + (j - center_coord[1])**2) for i, j in indx]
            center_k = np.argmin(distances)
            
            if not np.isnan(scores[center_k]):
                center_scores.append(scores[center_k])
        
        norm_factor = np.mean(center_scores) if center_scores else 1.0
        cbar_label = "Normalized Feature Drift" if norm_factor > 1e-9 else "Feature Drift Score"
        
        # Plot
        num_wells = len(wells)
        fig, axes = plt.subplots(1, num_wells, figsize=(5 * num_wells, 5), squeeze=False)
        ax_flat = axes.flatten()
        
        all_scores = [s / norm_factor for scores in plate_drift_scores.values() for s in scores if not np.isnan(s)]
        
        if not all_scores:
            plt.close(fig)
            return
        
        vmin = np.percentile(all_scores, 5)
        vmax = np.percentile(all_scores, 95)
        
        im = None
        for i, well in enumerate(wells):
            scores = plate_drift_scores.get(well, [])
            indx = well_indices.get(well, [])
            
            if not scores or not indx:
                continue
            
            normalized = [s / norm_factor for s in scores]
            
            out = np.full((grid_size, grid_size), np.nan)
            indx_i = [a[0] for a in indx]
            indx_j = [a[1] for a in indx]
            out[indx_i, indx_j] = normalized
            
            ax = ax_flat[i]
            im = ax.imshow(out, vmin=vmin, vmax=vmax, cmap="viridis", origin="lower")
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(well)
        
        if im is not None:
            cbar = fig.colorbar(im, ax=axes.ravel().tolist(), fraction=0.045, pad=0.04)
            cbar.set_label(cbar_label)
        
        fig.suptitle("Feature Drift Heatmap Across Plate", fontsize=16)
        
        path = save_figure(
            fig, self.output_dir / f"{self.feature_subset_name}_feature_drift_heatmap.png",
            dpi=300
        )
        result.add_file(path)
