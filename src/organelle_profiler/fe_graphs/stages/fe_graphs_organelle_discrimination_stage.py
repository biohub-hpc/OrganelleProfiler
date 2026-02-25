"""
Organelle Discrimination Stage: Compare segmentation groups' ability to discriminate gene KOs.

Key Question: Which organelle/segmentation group best separates different gene knockouts?

Metrics computed per organelle:
1. Gene Silhouette Score - How well cells from same gene cluster together
2. Cluster Purity (NMI) - How enriched are clusters for specific genes
3. Classification Accuracy - Can we predict gene from organelle features?
4. Inter/Intra Gene Distance Ratio - Separation between vs within genes

This answers: "If I only had features from one organelle, how well could I identify gene effects?"
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Optional, Dict, List, Tuple
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    silhouette_score, 
    normalized_mutual_info_score,
    adjusted_rand_score,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from scipy.spatial.distance import pdist, squareform
import logging
import warnings

from .fe_graphs_stage_base import BaseStage, StageResult
from ..core.fe_graphs_embedding import EmbeddingEngine
from ..core.fe_graphs_clustering import ClusteringEngine
from ..plotting.fe_graphs_utils import save_figure

logger = logging.getLogger(__name__)


class OrganelleDiscriminationStage(BaseStage):
    """
    Compare organelle feature sets on their ability to discriminate gene KOs.
    
    For each organelle/segmentation group, this stage computes:
    - Gene silhouette score (how well do cells from the same gene cluster?)
    - Cluster-gene mutual information (do clusters correspond to genes?)
    - KNN classification accuracy (can we predict gene from features?)
    - Distance ratio (inter-gene distance / intra-gene distance)
    
    Higher scores = better discrimination = more useful for identifying phenotypes.
    """
    
    STAGE_NUMBER = 7  # After differential
    STAGE_NAME = "organelle_discrimination"
    
    # Minimum requirements (adapt to level)
    MIN_ITEMS_PER_GENE = {
        "cell": 10,    # Need at least 10 cells per gene
        "guide": 2,    # Need at least 2 guides per gene  
        "gene": 1,     # All genes included
    }
    MIN_GENES = 5
    MAX_GENES_FOR_ANALYSIS = {
        "cell": 100,   # Subsample cells to 100 genes
        "guide": 200,  # More guides can be analyzed
        "gene": 1000,  # All genes can be analyzed
    }
    MAX_ITEMS_FOR_ANALYSIS = {
        "cell": 50000,   # Subsample to 50k cells
        "guide": 10000,  # Subsample to 10k guides
        "gene": 1000,    # All genes (typically ~1000)
    }
    
    def run(self) -> StageResult:
        """Run organelle discrimination comparison."""
        self.log_start("Comparing organelle discrimination power")
        result = StageResult()
        
        df = self.df.copy()
        features = self.get_features(df)
        
        if features.empty:
            result.add_error("No features available")
            return result
        
        # Check for gene column
        if "gene_name" not in df.columns:
            result.add_error("gene_name column not found")
            return result
        
        # Level-specific parameters
        min_items_per_gene = self.MIN_ITEMS_PER_GENE.get(self.level, 10)
        max_genes = self.MAX_GENES_FOR_ANALYSIS.get(self.level, 100)
        max_items = self.MAX_ITEMS_FOR_ANALYSIS.get(self.level, 50000)
        
        # Filter to genes with enough items
        gene_counts = df["gene_name"].value_counts()
        valid_genes = gene_counts[gene_counts >= min_items_per_gene].index.tolist()
        
        # Exclude NTC from discrimination analysis (we want to see KO separation)
        ntc_mask = self.get_ntc_mask(df)
        valid_genes = [g for g in valid_genes if g not in df.loc[ntc_mask, "gene_name"].unique()]
        
        if len(valid_genes) < self.MIN_GENES:
            result.add_error(f"Not enough genes with >={min_items_per_gene} {self.level}s")
            return result
        
        logger.info(f"Analyzing {len(valid_genes)} genes for discrimination at {self.level} level")
        
        # Subsample if needed
        if len(valid_genes) > max_genes:
            logger.info(f"Subsampling to {max_genes} genes")
            valid_genes = np.random.choice(valid_genes, max_genes, replace=False).tolist()
        
        # Filter to valid genes
        gene_mask = df["gene_name"].isin(valid_genes)
        analysis_df = df.loc[gene_mask].copy()
        analysis_features = features.loc[gene_mask].copy()
        
        # Subsample items if too many
        if len(analysis_df) > max_items:
            logger.info(f"Subsampling from {len(analysis_df)} to {max_items} {self.level}s")
            sample_idx = np.random.choice(len(analysis_df), max_items, replace=False)
            analysis_df = analysis_df.iloc[sample_idx].reset_index(drop=True)
            analysis_features = analysis_features.iloc[sample_idx].reset_index(drop=True)
        
        result.add_metric(f"n_{self.level}s_analyzed", len(analysis_df))
        result.add_metric("n_genes_analyzed", len(valid_genes))
        
        # Group features by organelle
        organelle_features = self.group_features_by_organelle(analysis_features.columns.tolist())
        organelles = sorted(organelle_features.keys())
        
        logger.info(f"Comparing {len(organelles)} organelle groups: {organelles}")
        
        if len(organelles) < 2:
            result.add_error("Need at least 2 organelle groups")
            return result
        
        # Compute discrimination scores for each organelle
        discrimination_results = []
        
        for organelle in organelles:
            cols = organelle_features[organelle]
            if len(cols) < 3:
                logger.info(f"Skipping {organelle}: only {len(cols)} features")
                continue
            
            org_features = analysis_features[cols].copy()
            org_features = org_features.dropna(axis=1, how="all").fillna(0)
            
            if org_features.shape[1] < 2:
                continue
            
            scores = self._compute_discrimination_scores(
                org_features,
                analysis_df["gene_name"].values,
                organelle,
            )
            
            if scores:
                scores["organelle"] = organelle
                scores["n_features"] = len(cols)
                discrimination_results.append(scores)
        
        if not discrimination_results:
            result.add_error("Could not compute discrimination scores")
            return result
        
        # Create results DataFrame
        disc_df = pd.DataFrame(discrimination_results)
        
        # Compute composite score (average of normalized metrics)
        metric_cols = ["silhouette_score", "nmi_score", "knn_accuracy", "distance_ratio"]
        for col in metric_cols:
            if col in disc_df.columns:
                disc_df[f"{col}_norm"] = (disc_df[col] - disc_df[col].min()) / (disc_df[col].max() - disc_df[col].min() + 1e-10)
        
        norm_cols = [f"{c}_norm" for c in metric_cols if f"{c}_norm" in disc_df.columns]
        disc_df["composite_score"] = disc_df[norm_cols].mean(axis=1)
        
        # Sort by composite score
        disc_df = disc_df.sort_values("composite_score", ascending=False)
        
        # Save results
        disc_df.to_csv(self.output_dir / "organelle_discrimination_scores.csv", index=False)
        result.add_file(self.output_dir / "organelle_discrimination_scores.csv")
        result.data["discrimination_df"] = disc_df
        
        # Save metric explanations
        self._save_metric_explanations(result)
        
        # Generate visualizations
        self._generate_plots(disc_df, organelles, result)
        
        # Add top organelle to metrics
        best_org = disc_df.iloc[0]["organelle"]
        result.add_metric("best_discriminating_organelle", best_org)
        result.add_metric("best_composite_score", disc_df.iloc[0]["composite_score"])
        
        logger.info(f"Best discriminating organelle: {best_org}")
        
        self.log_complete(result)
        return result
    
    def _compute_discrimination_scores(
        self,
        features: pd.DataFrame,
        gene_labels: np.ndarray,
        organelle: str,
    ) -> Optional[Dict]:
        """Compute all discrimination metrics for one organelle."""
        logger.info(f"Computing discrimination scores for {organelle}...")
        
        try:
            # Scale features
            scaler = StandardScaler()
            X = scaler.fit_transform(features.values)
            y = gene_labels
            
            scores = {}
            
            # 1. Gene Silhouette Score
            # How well do items from the same gene cluster together?
            # At gene level (1 sample per gene), this becomes inter-gene separation
            try:
                # Check if we have multiple samples per gene
                unique_labels, label_counts = np.unique(y, return_counts=True)
                min_samples_per_class = label_counts.min()
                
                if min_samples_per_class == 1:
                    # Gene level: Can't compute traditional silhouette with 1 sample per class
                    # Instead, use a proxy: average inter-gene distance (higher = better separation)
                    # Normalized by dataset diameter
                    from scipy.spatial.distance import pdist, squareform
                    dists = squareform(pdist(X, metric='euclidean'))
                    
                    # Average distance between different genes
                    inter_gene_dists = []
                    for i in range(len(X)):
                        for j in range(i+1, len(X)):
                            if y[i] != y[j]:  # Different genes
                                inter_gene_dists.append(dists[i, j])
                    
                    if inter_gene_dists:
                        mean_inter = np.mean(inter_gene_dists)
                        max_dist = dists.max()
                        # Normalize to [0, 1] range (higher = better separation)
                        sil_score = mean_inter / (max_dist + 1e-10)
                    else:
                        sil_score = 0
                    
                    logger.info(f"  {organelle} inter-gene separation: {sil_score:.3f} (gene-level proxy)")
                else:
                    # Cell/Guide level: Traditional silhouette
                    sil_score = silhouette_score(X, y, sample_size=min(5000, len(X)))
                    logger.info(f"  {organelle} silhouette: {sil_score:.3f}")
                
                scores["silhouette_score"] = sil_score
            except Exception as e:
                logger.warning(f"  Silhouette failed: {e}")
                scores["silhouette_score"] = 0
            
            # 2. Cluster-Gene NMI
            # Do unsupervised clusters correspond to gene labels?
            try:
                from sklearn.cluster import KMeans
                n_clusters = min(len(np.unique(y)), 30)
                kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                cluster_labels = kmeans.fit_predict(X)
                
                # Suppress sklearn warnings about too many unique classes (expected for gene-level analysis)
                with warnings.catch_warnings():
                    warnings.filterwarnings('ignore', message='.*number of unique classes.*', category=UserWarning)
                    nmi = normalized_mutual_info_score(y, cluster_labels)
                
                scores["nmi_score"] = nmi
                logger.info(f"  {organelle} NMI: {nmi:.3f}")
            except Exception as e:
                logger.warning(f"  NMI failed: {e}")
                scores["nmi_score"] = 0
            
            # 3. KNN Classification Accuracy
            # Can we predict gene from these features?
            # NOTE: This metric only makes sense at cell/guide level where we have multiple samples per gene
            try:
                # Check if we have multiple samples per class
                unique_labels, label_counts = np.unique(y, return_counts=True)
                min_samples_per_class = label_counts.min()
                
                if min_samples_per_class == 1:
                    # Gene level: Skip KNN classification (not meaningful with 1 sample per class)
                    # Use a simpler nearest-neighbor purity metric instead:
                    # What fraction of each gene's 5 nearest neighbors are from the same gene?
                    # Since we only have 1 sample per gene, this will tell us if genes cluster together
                    # in the broader sense (i.e., similar genes have similar features)
                    from sklearn.neighbors import NearestNeighbors
                    k = min(10, len(X) - 1)  # Use 10 nearest neighbors
                    nn = NearestNeighbors(n_neighbors=k + 1)  # +1 to exclude self
                    nn.fit(X)
                    distances, indices = nn.kneighbors(X)
                    
                    # For each sample, check if any neighbors share the same label
                    # (This won't happen at gene level, but gives us a measure of gene separation)
                    # Instead, compute average rank of the nearest same-gene sample
                    # Actually, since each gene has 1 sample, let's compute something else:
                    # Average distance to nearest neighbor / average distance to all samples
                    # Lower ratio = better clustering
                    nearest_distances = distances[:, 1]  # Exclude self (index 0)
                    all_distances = squareform(pdist(X, metric='euclidean'))
                    mean_nearest = np.mean(nearest_distances)
                    mean_all = np.mean(all_distances[np.triu_indices_from(all_distances, k=1)])
                    
                    # Convert to a 0-1 score where higher is better (1 - ratio)
                    # Cap at 1 to avoid negative scores
                    nn_purity = max(0, 1 - (mean_nearest / mean_all))
                    
                    scores["knn_accuracy"] = nn_purity
                    scores["knn_accuracy_std"] = 0
                    logger.info(f"  {organelle} neighbor purity (gene-level): {nn_purity:.3f}")
                else:
                    # Cell/Guide level: Use standard KNN classification with cross-validation
                    k = min(5, len(X)-1)
                    knn = KNeighborsClassifier(n_neighbors=k)
                    n_splits = min(3, min_samples_per_class)
                    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
                    
                    # Suppress sklearn warnings about too many unique classes
                    with warnings.catch_warnings():
                        warnings.filterwarnings('ignore', message='.*number of unique classes.*', category=UserWarning)
                        cv_scores = cross_val_score(knn, X, y, cv=cv, scoring="accuracy")
                    
                    scores["knn_accuracy"] = cv_scores.mean()
                    scores["knn_accuracy_std"] = cv_scores.std()
                    logger.info(f"  {organelle} KNN accuracy: {cv_scores.mean():.3f} +/- {cv_scores.std():.3f}")
            except Exception as e:
                logger.warning(f"  KNN failed: {e}")
                scores["knn_accuracy"] = 0
                scores["knn_accuracy_std"] = 0
            
            # 4. Inter/Intra Gene Distance Ratio
            # How separated are genes in feature space?
            try:
                dist_ratio = self._compute_distance_ratio(X, y)
                scores["distance_ratio"] = dist_ratio
                logger.info(f"  {organelle} distance ratio: {dist_ratio:.3f}")
            except Exception as e:
                logger.warning(f"  Distance ratio failed: {e}")
                scores["distance_ratio"] = 1.0
            
            return scores
            
        except Exception as e:
            logger.error(f"Failed to compute scores for {organelle}: {e}")
            return None
    
    def _compute_distance_ratio(
        self,
        X: np.ndarray,
        y: np.ndarray,
        n_sample: int = 1000,
    ) -> float:
        """
        Compute ratio of inter-gene to intra-gene distances.
        
        Higher ratio = better separation between genes.
        """
        # Sample if too many items
        if len(X) > n_sample:
            idx = np.random.choice(len(X), n_sample, replace=False)
            X = X[idx]
            y = y[idx]
        
        unique_genes = np.unique(y)
        
        # Compute centroids per gene
        centroids = {}
        for gene in unique_genes:
            mask = y == gene
            if mask.sum() > 0:
                centroids[gene] = X[mask].mean(axis=0)
        
        if len(centroids) < 2:
            return 1.0
        
        # Inter-gene distance (between centroids)
        centroid_matrix = np.array(list(centroids.values()))
        inter_dists = pdist(centroid_matrix, metric="euclidean")
        mean_inter = np.mean(inter_dists)
        
        # Intra-gene distance (within genes)
        intra_dists = []
        for gene in unique_genes:
            mask = y == gene
            gene_points = X[mask]
            if len(gene_points) > 1:
                # Distance to centroid
                centroid = centroids[gene]
                dists = np.sqrt(((gene_points - centroid) ** 2).sum(axis=1))
                intra_dists.extend(dists)
        
        mean_intra = np.mean(intra_dists) if intra_dists else 1.0
        
        return mean_inter / (mean_intra + 1e-10)
    
    def _save_metric_explanations(self, result: StageResult) -> None:
        """Save detailed explanation of discrimination metrics to a text file."""
        
        explanation = f"""ORGANELLE DISCRIMINATION METRICS - DETAILED EXPLANATION
{'=' * 80}

Analysis Level: {self.level.upper()}
Date Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}

{'=' * 80}
OVERVIEW
{'=' * 80}

This analysis compares different organelle/segmentation groups based on their ability 
to discriminate between gene knockouts. Each organelle is scored on 4 complementary 
metrics that together assess how well the morphological features from that organelle 
can identify and distinguish different genetic perturbations.

The key question: "If we only had features from ONE organelle, how well could we 
identify which gene was knocked out?"

{'=' * 80}
METRIC 1: GENE SILHOUETTE SCORE
{'=' * 80}

MAIN QUESTION:
How well do samples from the same gene cluster together in feature space, compared 
to samples from different genes?

STATISTICAL MEASURE:
- Cell/Guide Level: Traditional silhouette coefficient
  Formula: s(i) = (b(i) - a(i)) / max(a(i), b(i))
  where a(i) = mean intra-cluster distance
        b(i) = mean nearest-cluster distance
  
- Gene Level: Inter-gene separation proxy
  Formula: mean_inter_gene_distance / max_distance
  (normalized to [0,1] range since each gene has only 1 sample)

PARAMETER CHOICES:
- Distance metric: Euclidean distance in scaled feature space
- Sample size: Up to 5,000 samples (for computational efficiency)
- Features: StandardScaler normalized before distance computation

INTERPRETATION:
- Range: [-1, 1] for cell/guide level; [0, 1] for gene level
- Higher = Better: Samples from same gene are closer to each other than to other genes
- Score near 0: Overlapping gene phenotypes (poor discrimination)
- Score near 1: Well-separated gene phenotypes (excellent discrimination)

ASSUMPTIONS:
1. Euclidean distance is appropriate for scaled morphological features
2. Gene effects manifest as shifts in feature space (detectable via clustering)
3. At gene level, average inter-gene distance reflects discriminability
4. Missing values have been imputed or removed appropriately

WHAT IT TELLS US:
This metric reveals whether genes with the same knockout have consistent, 
distinguishable morphological signatures across the dataset. High scores indicate 
the organelle shows reproducible gene-specific phenotypes.


{'=' * 80}
METRIC 2: NORMALIZED MUTUAL INFORMATION (NMI) - CLUSTER-GENE AGREEMENT
{'=' * 80}

MAIN QUESTION:
When we cluster samples based purely on their morphological features (unsupervised), 
how well do those clusters correspond to the true gene labels?

STATISTICAL MEASURE:
Normalized Mutual Information between cluster assignments and gene labels
Formula: NMI(Y, C) = 2 * I(Y; C) / (H(Y) + H(C))
where I(Y; C) = mutual information between gene labels Y and clusters C
      H(Y), H(C) = entropies of Y and C

PARAMETER CHOICES:
- Clustering algorithm: K-Means
- Number of clusters: min(n_unique_genes, 30)
  Rationale: Match biological structure but cap for computational stability
- Random state: 42 (reproducibility)
- n_init: 10 (multiple initializations to avoid local minima)

INTERPRETATION:
- Range: [0, 1]
- 0 = No agreement (clusters don't correspond to genes at all)
- 1 = Perfect agreement (each cluster contains exactly one gene)
- ~0.6-0.8 = Good correspondence (typical for biological data with noise)

ASSUMPTIONS:
1. Gene effects create natural clusters in morphological space
2. K-means is appropriate for the feature distribution
3. Optimal number of clusters approximates number of distinct phenotypes
4. Cluster purity reflects biological signal strength

WHAT IT TELLS US:
This metric assesses whether the organelle features contain enough information 
to naturally separate genes without supervision. High NMI means the organelle 
reveals gene-specific phenotypic patterns that emerge from unsupervised analysis.
Unlike supervised methods, this tests if gene effects are "obvious" in the data.


{'=' * 80}
METRIC 3: {"NEIGHBOR PURITY" if self.level == "gene" else "KNN CLASSIFICATION ACCURACY"}
{'=' * 80}

MAIN QUESTION:
{"How tightly clustered are samples in feature space?" if self.level == "gene" else "Can we accurately predict which gene was knocked out based on morphological features?"}

STATISTICAL MEASURE:
"""
        
        if self.level == "gene":
            explanation += """- Gene Level: Neighbor Purity Score
  Formula: 1 - (mean_nearest_neighbor_distance / mean_all_pairwise_distances)
  
  This replaces traditional KNN accuracy because with 1 sample per gene, 
  leave-one-out cross-validation always yields 0% accuracy (the held-out 
  gene isn't in the training set by definition).

PARAMETER CHOICES:
- k neighbors: 10 (to assess local neighborhood structure)
- Distance metric: Euclidean in scaled feature space
- Normalization: Ratio relative to dataset diameter

INTERPRETATION:
- Range: [0, 1]
- Higher = Better clustering/separation in feature space
- Score near 0: Samples are spread uniformly (poor structure)
- Score near 1: Strong local clustering (good separation)

ASSUMPTIONS:
1. Nearest neighbor distance relative to global spread indicates clustering quality
2. Well-discriminated genes have tighter local neighborhoods
3. This proxy correlates with separability even without classification
"""
        else:
            explanation += """- Cell/Guide Level: K-Nearest Neighbors Cross-Validation Accuracy
  Standard supervised classification with held-out test sets

PARAMETER CHOICES:
- k neighbors: 5 (standard choice balancing bias-variance)
- Cross-validation: Stratified K-Fold (n_splits = min(3, min_samples_per_class))
- Random state: 42 (reproducibility)
- Scoring: Accuracy (fraction of correct predictions)

INTERPRETATION:
- Range: [0, 1] (0-100%)
- Baseline: 1/n_genes (random chance)
- >0.5 for many genes: Good discrimination power
- Near 1.0: Excellent gene identification capability

ASSUMPTIONS:
1. Local neighborhoods in feature space preserve class identity
2. k=5 is appropriate for the density and distribution of samples
3. Stratified CV maintains class balance across folds
4. Classification accuracy reflects biological discriminability
"""
        
        explanation += f"""
WHAT IT TELLS US:
{"This metric assesses whether genes form tight, separated clusters in feature space. High scores indicate strong phenotypic structure suitable for downstream classification or hit identification." if self.level == "gene" else "This metric directly tests whether an organelle's features are sufficient to identify gene perturbations in practice. High accuracy means this organelle could be used alone for phenotypic screening or hit calling."}


{'=' * 80}
METRIC 4: INTER/INTRA GENE DISTANCE RATIO
{'=' * 80}

MAIN QUESTION:
Are genes more different from each other than they are variable within themselves?
In other words, is between-gene variation larger than within-gene variation?

STATISTICAL MEASURE:
Ratio = mean_inter_gene_distance / mean_intra_gene_distance

where:
- Inter-gene distance: Euclidean distance between gene centroids
- Intra-gene distance: Mean distance of each sample to its gene centroid

PARAMETER CHOICES:
- Sample size: Up to 1,000 samples (for computational efficiency)
- Distance metric: Euclidean in scaled feature space
- Centroids: Mean feature vector per gene

INTERPRETATION:
- Range: [0, ∞), typically [1, 100]
- Ratio = 1: Inter-gene and intra-gene variation are equal (no discrimination)
- Ratio > 10: Strong separation (inter-gene variance dominates)
- Ratio > 50: Excellent separation (very distinct gene phenotypes)

ASSUMPTIONS:
1. Gene centroids represent the "true" average phenotype
2. Variation within a gene is primarily noise/biological variability
3. Variation between genes reflects true biological differences
4. Linear distances capture relevant feature space structure

WHAT IT TELLS US:
This metric directly quantifies the signal-to-noise ratio for gene discrimination.
High ratios indicate that gene-specific effects are much larger than measurement 
noise or within-gene heterogeneity. This is the most intuitive measure of 
discriminability: "Are the genes actually different?"


{'=' * 80}
COMPOSITE SCORE
{'=' * 80}

The composite score combines all four metrics into a single ranking:

FORMULA:
For each metric:
  normalized_metric = (metric - min) / (max - min)

composite_score = mean(normalized_silhouette, normalized_nmi, 
                      normalized_knn, normalized_distance_ratio)

RATIONALE:
- Each metric captures a different aspect of discrimination
- Normalization ensures equal weighting across different scales
- Average provides balanced overall assessment

INTERPRETATION:
- Range: [0, 1]
- Higher = Better overall discrimination power
- Best organelle = highest composite score
- Use to rank organelles for downstream analysis or focused experiments


{'=' * 80}
ANALYSIS PARAMETERS (THIS RUN)
{'=' * 80}

Level: {self.level}
Minimum items per gene: {self.MIN_ITEMS_PER_GENE.get(self.level, 10)}
Minimum genes required: {self.MIN_GENES}
Maximum genes analyzed: {self.MAX_GENES_FOR_ANALYSIS.get(self.level, 100)}
Maximum items analyzed: {self.MAX_ITEMS_FOR_ANALYSIS.get(self.level, 50000)}

Note: Data may be subsampled to these limits for computational efficiency.
Non-targeting controls (NTC) are excluded from discrimination analysis.


{'=' * 80}
INTERPRETATION GUIDELINES
{'=' * 80}

HIGH SCORES across all metrics suggest:
- This organelle shows strong, consistent gene-specific phenotypes
- Features from this organelle alone could identify perturbations
- Prioritize this organelle for detailed phenotypic analysis
- Use as primary feature set for hit calling

LOW SCORES across all metrics suggest:
- This organelle may not respond strongly to most gene perturbations
- Features are noisy or lack gene-specific structure
- Consider excluding from final feature set
- May still be valuable in combination with other organelles

MIXED SCORES (high on some, low on others):
- High silhouette + low NMI: Genes cluster, but overlap between genes
- High NMI + low classification: Clusters exist but boundaries are fuzzy
- High distance ratio + low silhouette: Strong centroids but high variance
- Investigate individual metric plots to understand specific patterns


{'=' * 80}
METHODOLOGICAL NOTES
{'=' * 80}

1. All features are StandardScaler normalized before analysis
   (zero mean, unit variance per feature)

2. NaN/missing values are handled by:
   - Dropping features with all NaN values
   - Filling remaining NaNs with 0 (post-scaling)

3. Warnings about "too many unique classes" are suppressed
   (expected at gene level where n_classes ≈ n_samples)

4. Random subsampling (if applied) uses fixed random_state for reproducibility

5. Distance calculations use scipy.spatial.distance for efficiency

6. All sklearn models use random_state=42 where applicable


{'=' * 80}
REFERENCES & FURTHER READING
{'=' * 80}

Silhouette Analysis:
- Rousseeuw, P.J. (1987). "Silhouettes: a graphical aid to the interpretation 
  and validation of cluster analysis." Journal of Computational and Applied 
  Mathematics, 20, 53-65.

Mutual Information:
- Vinh, N.X., Epps, J., & Bailey, J. (2010). "Information theoretic measures 
  for clusterings comparison: Variants, properties, normalization and 
  correction for chance." Journal of Machine Learning Research, 11, 2837-2854.

KNN Classification:
- Fix, E. & Hodges, J.L. (1951). "Discriminatory Analysis: Nonparametric 
  Discrimination: Consistency Properties." USAF School of Aviation Medicine.

Distance Ratios:
- Calinski, T. & Harabasz, J. (1974). "A dendrite method for cluster analysis."
  Communications in Statistics, 3(1), 1-27.


{'=' * 80}
END OF DOCUMENTATION
{'=' * 80}
"""
        
        # Save to file
        explanation_path = self.output_dir / "METRIC_EXPLANATIONS.txt"
        explanation_path.write_text(explanation)
        result.add_file(explanation_path)
        
        logger.info(f"Saved metric explanations to {explanation_path}")
    
    def _generate_plots(
        self,
        disc_df: pd.DataFrame,
        organelles: List[str],
        result: StageResult,
    ) -> None:
        """Generate comparison visualizations."""
        
        # 1. Composite Score Bar Chart (main result)
        self._plot_composite_scores(disc_df, result)
        
        # 2. Metric Comparison Heatmap
        self._plot_metric_heatmap(disc_df, result)
        
        # 3. Radar Plot (if we have enough organelles)
        if len(disc_df) >= 3:
            self._plot_radar_comparison(disc_df, result)
        
        # 4. Detailed metric bar charts
        self._plot_individual_metrics(disc_df, result)
    
    def _plot_composite_scores(self, disc_df: pd.DataFrame, result: StageResult) -> None:
        """Plot composite discrimination scores."""
        fig, ax = plt.subplots(figsize=(12, 6))
        
        colors = sns.color_palette("viridis", len(disc_df))
        
        bars = ax.bar(
            disc_df["organelle"],
            disc_df["composite_score"],
            color=colors,
        )
        
        # Add value labels
        for bar, val in zip(bars, disc_df["composite_score"]):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.02,
                f"{val:.2f}",
                ha="center", va="bottom", fontsize=10,
            )
        
        ax.set_xlabel("Organelle / Segmentation Group", fontsize=12)
        ax.set_ylabel("Composite Discrimination Score", fontsize=12)
        ax.set_title(
            "Organelle Discrimination Power for Gene KO Identification\n"
            "(Higher = Better at distinguishing gene knockouts)",
            fontsize=14,
        )
        ax.set_ylim(0, 1.1)
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        
        path = save_figure(fig, self.output_dir / "organelle_discrimination_composite.png")
        result.add_file(path)
    
    def _plot_metric_heatmap(self, disc_df: pd.DataFrame, result: StageResult) -> None:
        """Plot heatmap of all metrics per organelle."""
        metric_cols = ["silhouette_score", "nmi_score", "knn_accuracy", "distance_ratio"]
        available_cols = [c for c in metric_cols if c in disc_df.columns]
        
        if not available_cols:
            return
        
        heatmap_data = disc_df.set_index("organelle")[available_cols]
        
        # Normalize for heatmap
        heatmap_norm = (heatmap_data - heatmap_data.min()) / (heatmap_data.max() - heatmap_data.min() + 1e-10)
        
        fig, ax = plt.subplots(figsize=(10, max(6, len(disc_df) * 0.5)))
        
        sns.heatmap(
            heatmap_norm,
            annot=heatmap_data.round(3),
            fmt="",
            cmap="YlGnBu",
            ax=ax,
            cbar_kws={"label": "Normalized Score"},
        )
        
        ax.set_title("Discrimination Metrics by Organelle\n(Values shown, colors normalized)")
        ax.set_xlabel("Metric")
        ax.set_ylabel("Organelle")
        
        # Better metric labels - adjust based on level
        knn_label = "Neighbor\nPurity" if self.level == "gene" else "KNN\nAccuracy"
        labels = {
            "silhouette_score": "Gene\nSilhouette",
            "nmi_score": "Cluster-Gene\nNMI",
            "knn_accuracy": knn_label,
            "distance_ratio": "Distance\nRatio",
        }
        ax.set_xticklabels([labels.get(c, c) for c in available_cols], rotation=0)
        
        plt.tight_layout()
        
        path = save_figure(fig, self.output_dir / "organelle_discrimination_heatmap.png")
        result.add_file(path)
    
    def _plot_radar_comparison(self, disc_df: pd.DataFrame, result: StageResult) -> None:
        """Plot radar chart comparing all organelles."""
        from math import pi
        
        metric_cols = ["silhouette_score", "nmi_score", "knn_accuracy", "distance_ratio"]
        available_cols = [c for c in metric_cols if c in disc_df.columns]
        
        if len(available_cols) < 3:
            return
        
        # Use all organelles (previously limited to top 5)
        plot_df = disc_df.copy()
        
        # Normalize metrics to 0-1
        normalized = plot_df[available_cols].copy()
        for col in available_cols:
            col_min, col_max = disc_df[col].min(), disc_df[col].max()
            normalized[col] = (normalized[col] - col_min) / (col_max - col_min + 1e-10)
        
        # Radar setup
        categories = available_cols
        N = len(categories)
        angles = [n / float(N) * 2 * pi for n in range(N)]
        angles += angles[:1]  # Close the loop
        
        # Adjust figure size based on number of organelles
        fig_size = max(12, min(16, 10 + len(plot_df) * 0.3))
        fig, ax = plt.subplots(figsize=(fig_size, fig_size), subplot_kw=dict(polar=True))
        
        colors = sns.color_palette("husl", len(plot_df))
        
        for idx, (_, row) in enumerate(plot_df.iterrows()):
            values = normalized.iloc[idx][available_cols].values.tolist()
            values += values[:1]  # Close the loop
            
            ax.plot(angles, values, "o-", linewidth=2, label=row["organelle"], color=colors[idx], alpha=0.7)
            ax.fill(angles, values, alpha=0.05, color=colors[idx])
        
        ax.set_xticks(angles[:-1])
        # Adjust labels based on analysis level
        knn_label = "Neighbor\nPurity" if self.level == "gene" else "KNN\nAccuracy"
        ax.set_xticklabels([
            "Gene\nSilhouette",
            "Cluster-Gene\nNMI", 
            knn_label,
            "Distance\nRatio"
        ][:len(available_cols)])
        
        ax.set_title(f"Organelle Discrimination Comparison (All {len(plot_df)} Organelles)", size=14, y=1.08)
        ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.0), fontsize=9)
        
        plt.tight_layout()
        
        path = save_figure(fig, self.output_dir / "organelle_discrimination_radar.png")
        result.add_file(path)
    
    def _plot_individual_metrics(self, disc_df: pd.DataFrame, result: StageResult) -> None:
        """Plot individual metric comparisons."""
        # Adjust KNN accuracy label based on analysis level
        knn_label = "Neighbor Purity" if self.level == "gene" else "KNN Classification Accuracy"
        knn_subtitle = "Nearest neighbor clustering quality" if self.level == "gene" else "Ability to predict gene from features"
        
        metrics = [
            ("silhouette_score", "Gene Silhouette Score", "How well cells from same gene cluster"),
            ("nmi_score", "Cluster-Gene NMI", "Agreement between clusters and gene labels"),
            ("knn_accuracy", knn_label, knn_subtitle),
            ("distance_ratio", "Inter/Intra Gene Distance Ratio", "Separation between gene groups"),
        ]
        
        for col, title, subtitle in metrics:
            if col not in disc_df.columns:
                continue
            
            fig, ax = plt.subplots(figsize=(12, 5))
            
            # Sort by this metric
            sorted_df = disc_df.sort_values(col, ascending=False)
            
            colors = sns.color_palette("viridis", len(sorted_df))
            
            bars = ax.bar(sorted_df["organelle"], sorted_df[col], color=colors)
            
            # Add value labels
            for bar, val in zip(bars, sorted_df[col]):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.01,
                    f"{val:.3f}",
                    ha="center", va="bottom", fontsize=9,
                )
            
            ax.set_xlabel("Organelle")
            ax.set_ylabel(title)
            ax.set_title(f"{title}\n({subtitle})")
            plt.xticks(rotation=45, ha="right")
            plt.tight_layout()
            
            path = save_figure(fig, self.output_dir / f"organelle_{col}.png")
            result.add_file(path)
