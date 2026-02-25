"""
Reusable Cluster Analysis Functions.

Provides functions for:
- Computing distinguishing features for clusters (Cohen's d)
- Selecting representative items based on feature values
- Organizing organelle importance analysis

Used by both positive_controls_stage and embedding_visualization_stage.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


def compute_distinguishing_features(
    cluster_items: pd.Index,
    all_items: pd.Index,
    features: pd.DataFrame,
    top_n: int = 40,
) -> pd.DataFrame:
    """
    Compute features that distinguish a cluster from all other items using Cohen's d.
    
    Parameters
    ----------
    cluster_items : pd.Index
        Indices of items in the cluster
    all_items : pd.Index
        Indices of all items (for comparison)
    features : pd.DataFrame
        Feature matrix (items x features)
    top_n : int
        Number of top features to return (default: 40)
    
    Returns
    -------
    pd.DataFrame
        DataFrame with columns: feature, cluster_mean, cluster_std, other_mean, other_std, cohens_d, abs_cohens_d
        Sorted by abs_cohens_d descending
    """
    # Get cluster and non-cluster items
    cluster_mask = all_items.isin(cluster_items)
    other_mask = ~cluster_mask
    
    if cluster_mask.sum() == 0 or other_mask.sum() == 0:
        logger.warning("Empty cluster or comparison group")
        return pd.DataFrame()
    
    cluster_features = features.loc[cluster_mask]
    other_features = features.loc[other_mask]
    
    # Compute Cohen's d for each feature
    feature_stats = []
    for col in features.columns:
        cluster_vals = cluster_features[col].dropna()
        other_vals = other_features[col].dropna()
        
        if len(cluster_vals) == 0 or len(other_vals) == 0:
            continue
        
        # Means
        cluster_mean = cluster_vals.mean()
        other_mean = other_vals.mean()
        
        # Standard deviations
        cluster_std = cluster_vals.std()
        other_std = other_vals.std()
        
        # Pooled standard deviation
        n1 = len(cluster_vals)
        n2 = len(other_vals)
        pooled_std = np.sqrt(((n1 - 1) * cluster_std**2 + (n2 - 1) * other_std**2) / (n1 + n2 - 2))
        
        # Cohen's d
        if pooled_std > 0:
            cohens_d = (cluster_mean - other_mean) / pooled_std
        else:
            cohens_d = 0.0
        
        feature_stats.append({
            "feature": col,
            "cluster_mean": cluster_mean,
            "cluster_std": cluster_std,
            "other_mean": other_mean,
            "other_std": other_std,
            "cohens_d": cohens_d,
            "abs_cohens_d": abs(cohens_d),
        })
    
    if not feature_stats:
        return pd.DataFrame()
    
    feature_df = pd.DataFrame(feature_stats)
    feature_df = feature_df.sort_values("abs_cohens_d", ascending=False)
    
    return feature_df.head(top_n)


def compute_organelle_importance_from_features(
    feature_df: pd.DataFrame,
    organelle_groups: Dict[str, List[str]],
) -> pd.DataFrame:
    """
    Compute organelle importance based on feature Cohen's d values.
    
    Uses max |Cohen's d| as the metric - if one feature in an organelle really
    distinguishes the cluster, that organelle shines.
    
    Parameters
    ----------
    feature_df : pd.DataFrame
        Output from compute_distinguishing_features() with columns including:
        feature, cohens_d, abs_cohens_d
    organelle_groups : dict
        Mapping of organelle names to lists of feature names
    
    Returns
    -------
    pd.DataFrame
        DataFrame with columns: organelle, max_abs_d, max_feature, n_features, mean_cohens_d
        Sorted by max_abs_d descending
    """
    if feature_df.empty:
        return pd.DataFrame()
    
    # Map features to organelles
    feature_to_organelle = {}
    for org_name, org_features in organelle_groups.items():
        for feat in org_features:
            feature_to_organelle[feat] = org_name
    
    feature_df = feature_df.copy()
    feature_df["organelle"] = feature_df["feature"].map(feature_to_organelle)
    
    # Filter out unmapped features
    feature_df = feature_df[feature_df["organelle"].notna()]
    
    if feature_df.empty:
        return pd.DataFrame()
    
    # Aggregate by organelle using MAX Cohen's d
    organelle_agg = feature_df.groupby("organelle").agg({
        "abs_cohens_d": ["max", "count"],
        "cohens_d": "mean"
    }).reset_index()
    
    organelle_agg.columns = ["organelle", "max_abs_d", "n_features", "mean_cohens_d"]
    
    # Find the feature with max |Cohen's d| for each organelle
    max_feature_per_organelle = {}
    for org_name in organelle_agg["organelle"]:
        org_features = feature_df[feature_df["organelle"] == org_name]
        max_idx = org_features["abs_cohens_d"].idxmax()
        max_feature_per_organelle[org_name] = org_features.loc[max_idx, "feature"]
    
    organelle_agg["max_feature"] = organelle_agg["organelle"].map(max_feature_per_organelle)
    
    # Sort by max_abs_d descending
    organelle_agg = organelle_agg.sort_values("max_abs_d", ascending=False)
    
    return organelle_agg


def select_representative_items_by_features(
    items: pd.Index,
    features: pd.DataFrame,
    top_features: pd.DataFrame,
    n_items: int = 6,
) -> pd.Index:
    """
    Select representative items that best exemplify distinguishing features.
    
    Selects items with high scores on top distinguishing features.
    
    Parameters
    ----------
    items : pd.Index
        Indices of items to select from (e.g., cells in a cluster)
    features : pd.DataFrame
        Feature matrix (items x features)
    top_features : pd.DataFrame
        Output from compute_distinguishing_features() with top features
    n_items : int
        Number of items to select
    
    Returns
    -------
    pd.Index
        Indices of selected representative items
    """
    if len(items) == 0 or top_features.empty:
        return pd.Index([])
    
    # Get features for cluster items
    cluster_features = features.loc[items]
    
    # Use top features (up to 10) for scoring
    score_features = top_features["feature"].head(10).tolist()
    score_features = [f for f in score_features if f in cluster_features.columns]
    
    if not score_features:
        # Fallback: random selection
        n_select = min(n_items, len(items))
        return items[:n_select]
    
    # Compute composite score
    # Standardize each feature and sum (weighted by Cohen's d magnitude)
    scores = pd.Series(0.0, index=items)
    for feat in score_features:
        feat_vals = cluster_features[feat]
        
        # Standardize within cluster
        if feat_vals.std() > 0:
            standardized = (feat_vals - feat_vals.mean()) / feat_vals.std()
        else:
            standardized = feat_vals
        
        # Weight by Cohen's d (from top_features)
        weight = top_features.loc[top_features["feature"] == feat, "abs_cohens_d"].values[0]
        
        # Direction: if Cohen's d is positive, we want high values; if negative, low values
        direction = top_features.loc[top_features["feature"] == feat, "cohens_d"].values[0]
        if direction < 0:
            standardized = -standardized
        
        scores += standardized * weight
    
    # Select top N items by score
    top_items = scores.nlargest(n_items).index
    
    return top_items

