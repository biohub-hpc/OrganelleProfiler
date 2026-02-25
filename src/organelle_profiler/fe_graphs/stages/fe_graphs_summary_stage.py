"""
Summary Stage: Aggregate results and generate reports.

Performs:
- Summary statistics
- Hit list compilation
- HTML report generation
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, Dict, List, Any
import logging

from .fe_graphs_stage_base import BaseStage, StageResult

logger = logging.getLogger(__name__)


class SummaryStage(BaseStage):
    """Summary stage for result aggregation."""
    
    STAGE_NUMBER = 8
    STAGE_NAME = "summary"
    
    def run(self) -> StageResult:
        """Generate summary and reports."""
        self.log_start()
        result = StageResult()
        
        # Collect metrics from all upstream stages
        summary_metrics = self._collect_metrics()
        
        # Save summary CSV
        summary_df = pd.DataFrame([summary_metrics])
        summary_df.to_csv(self.output_dir / f"{self.level}_level_summary.csv", index=False)
        result.add_file(self.output_dir / f"{self.level}_level_summary.csv")
        
        # Generate hit list (preliminary)
        hit_list = self._generate_hit_list()
        if hit_list is not None and not hit_list.empty:
            hit_list.to_csv(self.output_dir / f"top_hits_{self.level}_level.csv", index=False)
            result.add_file(self.output_dir / f"top_hits_{self.level}_level.csv")
            result.data["hit_list"] = hit_list
        
        # Store summary metrics
        for key, value in summary_metrics.items():
            result.add_metric(key, value)
        
        self.log_complete(result)
        return result
    
    def _collect_metrics(self) -> Dict[str, Any]:
        """Collect metrics from all upstream stages."""
        metrics = {
            "level": self.level,
            "n_items": self.n_items,
        }
        
        # QC metrics
        if "qc" in self.upstream:
            qc_metrics = self.upstream["qc"].metrics
            for key, val in qc_metrics.items():
                metrics[f"qc_{key}"] = val
        
        # Embedding metrics
        if "embedding" in self.upstream:
            emb_metrics = self.upstream["embedding"].metrics
            for key, val in emb_metrics.items():
                metrics[f"emb_{key}"] = val
        
        # Clustering metrics
        if "clustering" in self.upstream:
            clust_metrics = self.upstream["clustering"].metrics
            for key, val in clust_metrics.items():
                metrics[f"clust_{key}"] = val
        
        # Differential metrics
        if "differential" in self.upstream:
            diff_metrics = self.upstream["differential"].metrics
            for key, val in diff_metrics.items():
                metrics[f"diff_{key}"] = val
        
        return metrics
    
    def _generate_hit_list(self) -> Optional[pd.DataFrame]:
        """Generate preliminary hit list based on differential analysis."""
        if "differential" not in self.upstream:
            return None
        
        diff_stats = self.upstream["differential"].data.get("differential_stats")
        if diff_stats is None or diff_stats.empty:
            return None
        
        # Filter for significant hits
        zscore_thresh = self.analysis_config.zscore_threshold
        pval_thresh = self.analysis_config.pvalue_threshold
        
        hits = diff_stats[
            (diff_stats["abs_zscore"] > zscore_thresh) &
            (diff_stats["pvalue_adj"] < pval_thresh)
        ].copy()
        
        if hits.empty:
            # Fall back to top by z-score
            hits = diff_stats.nlargest(20, "abs_zscore").copy()
        
        # Rank
        hits["rank"] = range(1, len(hits) + 1)
        
        return hits[["rank", "feature", "zscore", "log2_fold_change", "pvalue", "pvalue_adj"]]
