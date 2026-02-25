"""
Spatial Drift Stage: Comprehensive radial drift and edge effect analysis.

This mirrors the original fe_graphs.py radial drift implementation including:
- Three-segment regression for inflection point detection
- Dual-axis plots (drift + cell density/count)
- Per-organelle drift analysis
- Summary plots across all feature groups
- Feature drift heatmaps

This is a more comprehensive implementation than the basic QC stage drift check.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Optional, Dict, List, Tuple
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from statsmodels.nonparametric.smoothers_lowess import lowess
import logging

from .fe_graphs_stage_base import BaseStage, StageResult
from ..plotting.fe_graphs_utils import save_figure

logger = logging.getLogger(__name__)


class SpatialDriftStage(BaseStage):
    """
    Comprehensive spatial drift analysis.
    
    Analyzes edge effects by measuring how feature values change with
    radial distance from well/tile centers.
    
    Methods available:
    - three_segment_regression: Fits 3 linear segments, finds inflection
    - segmented_regression: Fits 2 linear segments
    - second_derivative: Uses LOWESS smoothing + acceleration
    """
    
    STAGE_NUMBER = 2  # After QC, before embedding
    STAGE_NAME = "spatial_drift"
    
    def __init__(self, *args, drift_method: str = "three_segment_regression", **kwargs):
        super().__init__(*args, **kwargs)
        self.drift_method = drift_method
        self.well_drift_summary = []
        self.tile_drift_summary = []
    
    def run(self) -> StageResult:
        """Run comprehensive spatial drift analysis."""
        self.log_start("Generating Comprehensive Radial Drift Analysis")
        result = StageResult()
        
        # Only meaningful at cell level
        if self.level != "cell":
            logger.info(f"Skipping spatial drift at {self.level} level (cell level only)")
            result.data["skipped"] = True
            return result
        
        df = self.df.copy()
        features = self.get_features(df)
        
        if features.empty:
            result.add_error("No features available")
            return result
        
        # Calculate radial positions
        df = self._calculate_radial_positions(df)
        
        # Check if we have position data
        has_well = "well_radial_pos" in df.columns and not df["well_radial_pos"].isna().all()
        has_tile = "tile_radial_pos" in df.columns and not df["tile_radial_pos"].isna().all()
        
        # Log what analyses will run
        logger.info(f"Position data available:")
        logger.info(f"  Well radial position: {'YES' if has_well else 'NO (skipping well drift)'}")
        logger.info(f"  Tile radial position: {'YES' if has_tile else 'NO (skipping tile drift)'}")
        
        if not has_well and not has_tile:
            result.add_error("No radial position data available for either well or tile analysis")
            return result
        
        # Store drift results for summary
        result.data["has_well_drift"] = has_well
        result.data["has_tile_drift"] = has_tile
        
        # Get organelle groups once (single source of truth from adata.var['organelle'])
        organelle_features = self.group_features_by_organelle(features.columns.tolist())
        n_groups = len(organelle_features)
        
        # Validate organelle groups
        if n_groups == 0:
            logger.warning("No organelle groups found - will only analyze 'all_features'")
        elif n_groups > 50:
            logger.warning(f"Found {n_groups} groups - expected ~20. Check adata.var['organelle']")
            logger.warning(f"First 5 groups: {list(organelle_features.keys())[:5]}")
        else:
            logger.info(f"Found {n_groups} organelle groups: {list(organelle_features.keys())}")
        
        # Create output subdirectories for each drift type
        # Structure: 2_spatial_drift/well_drift/ and 2_spatial_drift/tile_drift/
        if has_well:
            well_dir = self.output_dir / "well_drift"
            well_dir.mkdir(exist_ok=True, parents=True)
            logger.info("\n" + "="*60)
            logger.info("  WELL-LEVEL DRIFT ANALYSIS (radial from well center)")
            logger.info("="*60)
            self._run_drift_analysis_for_position(
                features, df, organelle_features, 
                "well_radial_pos", well_dir, result
            )
        
        if has_tile:
            tile_dir = self.output_dir / "tile_drift"
            tile_dir.mkdir(exist_ok=True, parents=True)
            logger.info("\n" + "="*60)
            logger.info("  TILE/FOV DRIFT ANALYSIS (radial from tile center)")
            logger.info("="*60)
            self._run_drift_analysis_for_position(
                features, df, organelle_features,
                "tile_radial_pos", tile_dir, result
            )
        
        # Generate combined summary plot (comparing well vs tile)
        self._plot_combined_summary(result)
        
        self.log_complete(result)
        return result
    
    def _run_drift_analysis_for_position(
        self,
        features: pd.DataFrame,
        df: pd.DataFrame,
        organelle_features: Dict[str, List[str]],
        pos_type: str,
        output_dir: Path,
        result: StageResult,
    ) -> None:
        """
        Run drift analysis for a specific position type (well or tile).
        
        Generates plots for:
        1. All features combined
        2. Each organelle group
        3. Summary across all groups
        """
        pos_name = pos_type.split("_")[0]  # "well" or "tile"
        
        # 1. All features combined
        logger.info(f"  Analyzing ALL features combined ({pos_name})...")
        self._run_single_drift_analysis(features, df, "all_features", pos_type, output_dir, result)
        
        # 2. Per-organelle analysis
        logger.info(f"  Analyzing {len(organelle_features)} organelle groups...")
        skipped_organelles = []
        for organelle, cols in sorted(organelle_features.items()):
            if len(cols) < 3:
                logger.debug(f"    Skipping {organelle}: only {len(cols)} features (need >= 3)")
                skipped_organelles.append((organelle, len(cols)))
                continue
            
            # Check if features actually exist in the features DataFrame
            available_cols = [c for c in cols if c in features.columns]
            if len(available_cols) < 3:
                logger.warning(f"    Skipping {organelle}: only {len(available_cols)}/{len(cols)} features available in DataFrame (need >= 3)")
                skipped_organelles.append((organelle, len(available_cols)))
                continue
            
            org_features = features[available_cols].copy()
            
            # Check if all features have zero variance
            variances = org_features.var()
            if (variances == 0).all() or variances.isna().all():
                logger.warning(f"    Skipping {organelle}: all features have zero variance")
                skipped_organelles.append((organelle, f"zero_var"))
                continue
            
            self._run_single_drift_analysis(
                org_features, df, organelle, pos_type, output_dir, result
            )
        
        if skipped_organelles:
            logger.info(f"  Skipped {len(skipped_organelles)} organelle groups:")
            for org, reason in skipped_organelles:
                logger.info(f"    - {org}: {reason}")
        
        # 3. Summary plot for this position type
        summary_data = self.well_drift_summary if pos_name == "well" else self.tile_drift_summary
        if summary_data:
            self._plot_position_summary(summary_data, pos_name, output_dir, result)
    
    def _run_single_drift_analysis(
        self,
        features: pd.DataFrame,
        df: pd.DataFrame,
        name: str,
        pos_type: str,
        output_dir: Path,
        result: StageResult,
    ) -> None:
        """Run drift analysis for a single feature set and position type."""
        if pos_type not in df.columns or df[pos_type].isna().all():
            return
        
        # Calculate drift score
        scaled = StandardScaler().fit_transform(features.fillna(0))
        drift_score = np.mean(np.abs(scaled), axis=1)
        
        analysis_df = df.copy()
        analysis_df["feature_drift_score"] = drift_score
        
        percent_cutoff = self._create_drift_plot(
            analysis_df, pos_type, name, output_dir, result
        )
        
        if percent_cutoff is not None:
            summary_entry = {
                "group": name, 
                "percent_cutoff": percent_cutoff,
                "n_features": len(features.columns),
            }
            pos_name = pos_type.split("_")[0]
            if pos_name == "well":
                self.well_drift_summary.append(summary_entry)
            else:
                self.tile_drift_summary.append(summary_entry)
    
    def _calculate_radial_positions(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate radial positions from well and tile centers."""
        
        # === WELL RADIAL POSITION ===
        well_cols = ["x_global_pheno", "y_global_pheno", "well"]
        missing_well_cols = [c for c in well_cols if c not in df.columns]
        
        if missing_well_cols:
            logger.warning(f"Cannot calculate well radial position - missing columns: {missing_well_cols}")
        else:
            df["x_global_pheno"] = pd.to_numeric(df["x_global_pheno"], errors="coerce")
            df["y_global_pheno"] = pd.to_numeric(df["y_global_pheno"], errors="coerce")
            
            df["well_radial_pos"] = np.nan
            wells_processed = 0
            
            for well in df["well"].unique():
                well_mask = df["well"] == well
                well_df = df.loc[well_mask]
                
                valid_mask = well_df[["x_global_pheno", "y_global_pheno"]].notna().all(axis=1)
                if valid_mask.sum() < 10:
                    continue
                
                valid_idx = well_df[valid_mask].index
                
                # Calculate center as geometric center of cell positions
                cx = (well_df.loc[valid_idx, "x_global_pheno"].min() + 
                      well_df.loc[valid_idx, "x_global_pheno"].max()) / 2
                cy = (well_df.loc[valid_idx, "y_global_pheno"].min() + 
                      well_df.loc[valid_idx, "y_global_pheno"].max()) / 2
                
                radial = np.sqrt(
                    (well_df.loc[valid_idx, "x_global_pheno"] - cx) ** 2 +
                    (well_df.loc[valid_idx, "y_global_pheno"] - cy) ** 2
                )
                df.loc[valid_idx, "well_radial_pos"] = radial.values
                wells_processed += 1
            
            n_valid = df["well_radial_pos"].notna().sum()
            logger.info(f"Well radial positions: {n_valid:,} cells across {wells_processed} wells")
        
        # === TILE RADIAL POSITION ===
        tile_cols = ["x_local_pheno", "y_local_pheno", "tile_pheno"]
        missing_tile_cols = [c for c in tile_cols if c not in df.columns]
        
        if missing_tile_cols:
            logger.warning(f"Cannot calculate tile radial position - missing columns: {missing_tile_cols}")
            logger.warning("  Tile drift analysis will be SKIPPED")
            logger.warning("  To enable: ensure adata.obs has x_local_pheno, y_local_pheno, tile_pheno")
        else:
            # Convert coordinate columns to numeric (tile_pheno stays categorical)
            df["x_local_pheno"] = pd.to_numeric(df["x_local_pheno"], errors="coerce")
            df["y_local_pheno"] = pd.to_numeric(df["y_local_pheno"], errors="coerce")
            # tile_pheno is a categorical identifier (e.g., "A1", "B2"), don't convert to numeric
            
            # Check for all-NaN columns
            x_nan = df["x_local_pheno"].isna().all()
            y_nan = df["y_local_pheno"].isna().all()
            tile_nan = df["tile_pheno"].isna().all()
            
            if x_nan or y_nan or tile_nan:
                nan_cols = []
                if x_nan: nan_cols.append("x_local_pheno")
                if y_nan: nan_cols.append("y_local_pheno")
                if tile_nan: nan_cols.append("tile_pheno")
                logger.warning(f"Tile columns exist but are all NaN: {nan_cols}")
                logger.warning("  Tile drift analysis will be SKIPPED")
                logger.warning("  These columns may need to be enriched from linked_results CSV")
            else:
                df["tile_radial_pos"] = np.nan
                tiles_processed = 0
                
                for tile in df["tile_pheno"].dropna().unique():
                    tile_mask = df["tile_pheno"] == tile
                    tile_df = df.loc[tile_mask]
                    
                    valid_mask = tile_df[["x_local_pheno", "y_local_pheno"]].notna().all(axis=1)
                    if valid_mask.sum() < 10:
                        continue
                    
                    valid_idx = tile_df[valid_mask].index
                    
                    cx = (tile_df.loc[valid_idx, "x_local_pheno"].min() + 
                          tile_df.loc[valid_idx, "x_local_pheno"].max()) / 2
                    cy = (tile_df.loc[valid_idx, "y_local_pheno"].min() + 
                          tile_df.loc[valid_idx, "y_local_pheno"].max()) / 2
                    
                    radial = np.sqrt(
                        (tile_df.loc[valid_idx, "x_local_pheno"] - cx) ** 2 +
                        (tile_df.loc[valid_idx, "y_local_pheno"] - cy) ** 2
                    )
                    df.loc[valid_idx, "tile_radial_pos"] = radial.values
                    tiles_processed += 1
                
                n_valid = df["tile_radial_pos"].notna().sum()
                logger.info(f"Tile radial positions: {n_valid:,} cells across {tiles_processed} tiles")
        
        return df
    
    def _create_drift_plot(
        self,
        df: pd.DataFrame,
        pos_type: str,
        name: str,
        output_dir: Path,
        result: StageResult,
    ) -> Optional[float]:
        """Create a single drift plot with regression analysis."""
        plot_data = df[[pos_type, "feature_drift_score"]].dropna()
        
        if len(plot_data) < 100:
            logger.info(f"    Skipping {pos_type}: insufficient data ({len(plot_data)})")
            return None
        
        # Bin by radial distance
        num_bins = 50
        plot_data["radial_bin"] = pd.cut(plot_data[pos_type], bins=num_bins)
        
        binned = (
            plot_data.groupby("radial_bin", observed=False)
            .agg(
                mean_drift=("feature_drift_score", "mean"),
                cell_count=("feature_drift_score", "count"),
            )
            .reset_index()
        )
        
        # Calculate cell density (area of annulus)
        binned["r_inner"] = [b.left for b in binned["radial_bin"]]
        binned["r_outer"] = [b.right for b in binned["radial_bin"]]
        bin_area = np.pi * (binned["r_outer"] ** 2 - binned["r_inner"] ** 2)
        bin_area = bin_area.replace(0, np.nan)
        binned["cell_density"] = binned["cell_count"] / bin_area
        
        binned["radial_midpoint"] = [b.mid for b in binned["radial_bin"]]
        binned.dropna(subset=["mean_drift", "cell_density", "radial_midpoint"], inplace=True)
        
        if len(binned) < 5:
            logger.info(f"    Skipping {pos_type}: not enough bins ({len(binned)})")
            return None
        
        # Normalize relative to center
        center_drift = binned.sort_values("radial_midpoint").iloc[0]["mean_drift"]
        if center_drift > 1e-9:
            normalized_drift = binned["mean_drift"] / center_drift
            y_label = "Normalized Feature Drift (Relative to Center)"
        else:
            normalized_drift = binned["mean_drift"]
            y_label = "Mean Absolute Scaled Feature Value"
        
        x_data = binned["radial_midpoint"].values
        y_data = normalized_drift.values
        
        # Find breakpoint using selected method
        breakpoint_x, fit_data = self._find_breakpoint(x_data, y_data)
        
        # Calculate cell counts
        cells_before = (plot_data[pos_type] <= breakpoint_x).sum()
        cells_after = (plot_data[pos_type] > breakpoint_x).sum()
        total_cells = cells_before + cells_after
        percent_after = 100 * cells_after / total_cells if total_cells > 0 else 0
        
        # Create plots (density and count versions in separate subdirs)
        for plot_type in ["density", "count"]:
            fig, ax1 = plt.subplots(figsize=(14, 8))
            
            # Main drift line
            ax1.plot(x_data, y_data, "o-", alpha=0.6, label="Mean Feature Drift per Bin", zorder=10)
            
            # Plot regression segments
            if fit_data is not None:
                self._plot_regression_fits(ax1, fit_data)
            
            # Inflection line
            ax1.axvline(breakpoint_x, color="black", linestyle="--",
                       label=f"Inflection Point at {breakpoint_x:.2f}", zorder=15)
            
            ax1.set_xlabel(f'Radial Distance from {pos_type.split("_")[0].capitalize()} Center')
            ax1.set_ylabel(y_label)
            ax1.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)
            
            # Secondary axis
            ax2 = ax1.twinx()
            if plot_type == "density":
                ax2.bar(binned["radial_midpoint"], binned["cell_density"],
                       width=(binned["radial_midpoint"].max() / num_bins) * 0.9,
                       alpha=0.2, color="gray", label="Cell Density")
                ax2.set_ylabel("Cell Density (Cells / Pixel²)", color="gray")
            else:
                ax2.bar(binned["radial_midpoint"], binned["cell_count"],
                       width=(binned["radial_midpoint"].max() / num_bins) * 0.9,
                       alpha=0.2, color="gray", label="Cell Count")
                ax2.set_ylabel("Cell Count per Bin", color="gray")
            
            ax2.tick_params(axis="y", labelcolor="gray")
            
            # Ensure drift line is on top
            ax1.set_zorder(ax2.get_zorder() + 1)
            ax1.patch.set_visible(False)
            
            # Title
            pos_name = pos_type.split("_")[0].capitalize()
            display_name = name.replace("_", " ").title() if name != "all_features" else "All Features"
            fig.suptitle(f"Feature Drift vs. {pos_name} Radial Position\n{display_name}", fontsize=14)
            
            # Legend with cell count info
            handles1, labels1 = ax1.get_legend_handles_labels()
            handles2, labels2 = ax2.get_legend_handles_labels()
            
            info_text = (
                f"Cells Before Inflection: {cells_before:,} ({100 - percent_after:.1f}%)\n"
                f"Cells After Inflection: {cells_after:,} ({percent_after:.1f}%)"
            )
            
            leg = fig.legend(handles=handles1 + handles2, labels=labels1 + labels2,
                           loc="upper right", bbox_to_anchor=(0.9, 0.88))
            leg.set_title(info_text, prop={"size": "small", "weight": "bold"})
            
            plt.tight_layout(rect=[0, 0, 1, 0.96])
            
            # Save to subdir: density/ or count/
            subdir = output_dir / plot_type
            subdir.mkdir(exist_ok=True, parents=True)
            
            # Simpler filename since subdir provides context
            filename = f"{name}.png"
            path = save_figure(fig, subdir / filename, dpi=300)
            result.add_file(path)
        
        return percent_after
    
    def _find_breakpoint(
        self,
        x_data: np.ndarray,
        y_data: np.ndarray,
    ) -> Tuple[float, Optional[Dict]]:
        """Find breakpoint using configured method."""
        if self.drift_method == "three_segment_regression":
            return self._three_segment_regression(x_data, y_data)
        elif self.drift_method == "segmented_regression":
            return self._segmented_regression(x_data, y_data)
        elif self.drift_method == "second_derivative":
            return self._second_derivative(x_data, y_data)
        else:
            return x_data.mean(), None
    
    def _three_segment_regression(
        self,
        x_data: np.ndarray,
        y_data: np.ndarray,
    ) -> Tuple[float, Optional[Dict]]:
        """Find breakpoint using three-segment regression."""
        n = len(x_data)
        min_error = np.inf
        best_breaks = (-1, -1)
        
        # Search for best two breakpoints
        for i in range(2, n - 4):
            for j in range(i + 2, n - 2):
                x1, y1 = x_data[:i], y_data[:i]
                x2, y2 = x_data[i:j], y_data[i:j]
                x3, y3 = x_data[j:], y_data[j:]
                
                m1 = LinearRegression().fit(x1.reshape(-1, 1), y1)
                m2 = LinearRegression().fit(x2.reshape(-1, 1), y2)
                m3 = LinearRegression().fit(x3.reshape(-1, 1), y3)
                
                error = (
                    np.sum((y1 - m1.predict(x1.reshape(-1, 1))) ** 2) +
                    np.sum((y2 - m2.predict(x2.reshape(-1, 1))) ** 2) +
                    np.sum((y3 - m3.predict(x3.reshape(-1, 1))) ** 2)
                )
                
                if error < min_error:
                    min_error = error
                    best_breaks = (i, j)
        
        if best_breaks[0] == -1:
            return x_data.mean(), None
        
        i, j = best_breaks
        
        # Fit final models
        x1, y1 = x_data[:i], y_data[:i]
        x2, y2 = x_data[i:j], y_data[i:j]
        x3, y3 = x_data[j:], y_data[j:]
        
        m1 = LinearRegression().fit(x1.reshape(-1, 1), y1)
        m2 = LinearRegression().fit(x2.reshape(-1, 1), y2)
        m3 = LinearRegression().fit(x3.reshape(-1, 1), y3)
        
        # Choose inflection point (use second breakpoint if first is before midpoint)
        x_mid = (x_data[0] + x_data[-1]) / 2
        if x_data[i] < x_mid:
            breakpoint_x = x_data[j]
        else:
            breakpoint_x = x_data[i]
        
        return breakpoint_x, {
            "method": "three_segment",
            "segments": [
                (x1, m1.predict(x1.reshape(-1, 1)), "red", "Segment 1"),
                (x2, m2.predict(x2.reshape(-1, 1)), "cyan", "Segment 2"),
                (x3, m3.predict(x3.reshape(-1, 1)), "magenta", "Segment 3"),
            ]
        }
    
    def _segmented_regression(
        self,
        x_data: np.ndarray,
        y_data: np.ndarray,
    ) -> Tuple[float, Optional[Dict]]:
        """Find breakpoint using two-segment regression."""
        n = len(x_data)
        min_error = np.inf
        best_break = -1
        
        for i in range(2, n - 2):
            x1, y1 = x_data[:i], y_data[:i]
            x2, y2 = x_data[i:], y_data[i:]
            
            m1 = LinearRegression().fit(x1.reshape(-1, 1), y1)
            m2 = LinearRegression().fit(x2.reshape(-1, 1), y2)
            
            error = (
                np.sum((y1 - m1.predict(x1.reshape(-1, 1))) ** 2) +
                np.sum((y2 - m2.predict(x2.reshape(-1, 1))) ** 2)
            )
            
            if error < min_error:
                min_error = error
                best_break = i
        
        if best_break == -1:
            return x_data.mean(), None
        
        x1, y1 = x_data[:best_break], y_data[:best_break]
        x2, y2 = x_data[best_break:], y_data[best_break:]
        
        m1 = LinearRegression().fit(x1.reshape(-1, 1), y1)
        m2 = LinearRegression().fit(x2.reshape(-1, 1), y2)
        
        return x_data[best_break], {
            "method": "segmented",
            "segments": [
                (x1, m1.predict(x1.reshape(-1, 1)), "red", "Segment 1"),
                (x2, m2.predict(x2.reshape(-1, 1)), "cyan", "Segment 2"),
            ]
        }
    
    def _second_derivative(
        self,
        x_data: np.ndarray,
        y_data: np.ndarray,
    ) -> Tuple[float, Optional[Dict]]:
        """Find breakpoint using LOWESS smoothing + second derivative."""
        smoothed = lowess(y_data, x_data, frac=0.5)
        second_deriv = np.gradient(np.gradient(smoothed[:, 1], smoothed[:, 0]), smoothed[:, 0])
        
        breakpoint_idx = np.argmax(second_deriv)
        breakpoint_x = smoothed[breakpoint_idx, 0]
        
        return breakpoint_x, {
            "method": "second_derivative",
            "smoothed": smoothed,
            "breakpoint_idx": breakpoint_idx,
        }
    
    def _plot_regression_fits(self, ax: plt.Axes, fit_data: Dict) -> None:
        """Plot regression fit lines."""
        if fit_data["method"] in ["three_segment", "segmented"]:
            for x_seg, y_pred, color, label in fit_data["segments"]:
                ax.plot(x_seg, y_pred, color=color, linestyle="--",
                       linewidth=2.5, label=f"{label} Fit", zorder=20)
        
        elif fit_data["method"] == "second_derivative":
            smoothed = fit_data["smoothed"]
            ax.plot(smoothed[:, 0], smoothed[:, 1], color="red",
                   linewidth=2.5, label="LOWESS Smoothed", zorder=20)
            
            idx = fit_data["breakpoint_idx"]
            ax.scatter(smoothed[idx, 0], smoothed[idx, 1],
                      s=150, c="black", marker="X", zorder=30, label="Inflection")
    
    def _plot_position_summary(
        self,
        summary_data: List[Dict],
        pos_name: str,
        output_dir: Path,
        result: StageResult,
    ) -> None:
        """Generate summary plot of inflection points for one position type."""
        if not summary_data:
            return
        
        df = pd.DataFrame(summary_data)
        
        # Clean up names for display
        df["group_display"] = (
            df["group"]
            .str.replace("_", " ")
            .str.title()
        )
        df.loc[df["group"] == "all_features", "group_display"] = "All Features (Combined)"
        
        # Sort: all_features first, then alphabetically
        df["sort_key"] = df["group"].apply(lambda x: "0" if x == "all_features" else x)
        df = df.sort_values("sort_key")
        
        # Plot
        fig, ax = plt.subplots(figsize=(12, max(6, len(df) * 0.4)))
        
        colors = sns.color_palette("viridis", len(df))
        bars = ax.barh(df["group_display"], df["percent_cutoff"], color=colors)
        
        ax.set_xlabel("Percent of Cells After Inflection Point (%)")
        ax.set_ylabel("Feature Group (Organelle)")
        ax.set_title(
            f"{pos_name.title()}-Level Drift Analysis Summary\n"
            f"(% of cells beyond edge-effect inflection point)",
            fontsize=14
        )
        ax.set_xlim(0, 100)
        
        # Add value labels
        for bar, val in zip(bars, df["percent_cutoff"]):
            ax.text(val + 1, bar.get_y() + bar.get_height() / 2,
                   f"{val:.1f}%", va="center", fontsize=9)
        
        plt.tight_layout()
        path = save_figure(fig, output_dir / f"inflection_summary.png", dpi=300)
        result.add_file(path)
    
    def _plot_combined_summary(self, result: StageResult) -> None:
        """Generate combined summary comparing well vs tile drift."""
        if not self.well_drift_summary and not self.tile_drift_summary:
            return
        
        # Prepare data
        well_df = pd.DataFrame(self.well_drift_summary) if self.well_drift_summary else pd.DataFrame()
        tile_df = pd.DataFrame(self.tile_drift_summary) if self.tile_drift_summary else pd.DataFrame()
        
        if well_df.empty and tile_df.empty:
            return
        
        # Mark source
        if not well_df.empty:
            well_df["position_type"] = "Well"
        if not tile_df.empty:
            tile_df["position_type"] = "Tile/FOV"
        
        # Combine
        combined = pd.concat([well_df, tile_df], ignore_index=True)
        
        # Clean up group names
        combined["group_display"] = (
            combined["group"]
            .str.replace("_", " ")
            .str.title()
        )
        combined.loc[combined["group"] == "all_features", "group_display"] = "All Features"
        
        # Pivot for side-by-side comparison
        pivot = combined.pivot(index="group_display", columns="position_type", values="percent_cutoff")
        pivot = pivot.reindex(["All Features"] + sorted([g for g in pivot.index if g != "All Features"]))
        
        # Plot side-by-side comparison
        fig, ax = plt.subplots(figsize=(14, max(6, len(pivot) * 0.5)))
        
        x = np.arange(len(pivot))
        width = 0.35
        
        colors = {"Well": "#3498db", "Tile/FOV": "#e74c3c"}
        
        for i, pos_type in enumerate(["Well", "Tile/FOV"]):
            if pos_type in pivot.columns:
                offset = width * (i - 0.5)
                bars = ax.barh(
                    x + offset, 
                    pivot[pos_type].fillna(0),
                    width, 
                    label=pos_type,
                    color=colors[pos_type],
                    alpha=0.8,
                )
                # Add value labels
                for j, (bar, val) in enumerate(zip(bars, pivot[pos_type])):
                    if pd.notna(val):
                        ax.text(val + 1, bar.get_y() + bar.get_height() / 2,
                               f"{val:.1f}%", va="center", fontsize=8)
        
        ax.set_yticks(x)
        ax.set_yticklabels(pivot.index)
        ax.set_xlabel("Percent of Cells After Inflection Point (%)")
        ax.set_ylabel("Feature Group (Organelle)")
        ax.set_title(
            "Spatial Drift Comparison: Well vs Tile/FOV Level\n"
            "(Higher = less edge effect impact)",
            fontsize=14
        )
        ax.set_xlim(0, 100)
        ax.legend(loc="lower right")
        ax.grid(axis="x", alpha=0.3)
        
        plt.tight_layout()
        path = save_figure(fig, self.output_dir / "well_vs_tile_comparison.png", dpi=300)
        result.add_file(path)
        
        # Save combined summary CSV
        combined.to_csv(self.output_dir / "drift_summary.csv", index=False)
        result.add_file(self.output_dir / "drift_summary.csv")
