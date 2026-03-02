"""
Phenotypic Activity Assessment using copairs mAP (mean Average Precision).

Backward-compatible re-export wrapper. All implementations now live in
``ops_utils.analysis.map_scores`` for cross-pipeline reuse.
"""

# Re-export everything from the shared ops_utils module
from ops_utils.analysis.map_scores import (  # noqa: F401
    _compute_single_complex_map,
    adata_to_copairs_df,
    compute_auc_score,
    compute_threshold_sweep_auc,
    phenotypic_activity_assesment,
    phenotypic_distinctivness,
    phenotypic_consistency_corum,
    phenotypic_consistency_manual_annotation,
    map_main,
)

__all__ = [
    "_compute_single_complex_map",
    "adata_to_copairs_df",
    "compute_auc_score",
    "compute_threshold_sweep_auc",
    "phenotypic_activity_assesment",
    "phenotypic_distinctivness",
    "phenotypic_consistency_corum",
    "phenotypic_consistency_manual_annotation",
    "map_main",
]
