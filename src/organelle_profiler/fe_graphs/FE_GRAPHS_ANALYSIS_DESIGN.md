# Feature Graphs Analysis Design

## Philosophy

This pipeline follows a **hierarchical analytical flow** organized by:
1. **Level** (cell → guide → gene) - aggregation hierarchy
2. **Stage** (QC → embedding → clustering → differential → summary) - analytical sequence
3. **Chaining** - results from each level inform the next

---

## Output Organization

```
{experiment}/4-features/graphs/
│
├── 1_cell_level/                          # Individual cell analysis (~millions)
│   ├── 1_qc/                              # Quality control
│   │   ├── spatial_drift/                 # Edge effects
│   │   │   ├── well_radial_drift.png
│   │   │   ├── tile_radial_drift.png
│   │   │   └── drift_summary.csv
│   │   ├── batch_effects/                 # Well/plate effects
│   │   │   ├── umap_by_well.png
│   │   │   └── well_mixing_score.csv
│   │   └── feature_quality/               # Feature distributions
│   │       ├── feature_variance.png
│   │       ├── correlation_matrix.png
│   │       └── qc_summary.csv
│   │
│   ├── 2_embedding/                       # Dimensionality reduction
│   │   ├── pca/
│   │   │   ├── variance_explained.png
│   │   │   └── pca_loadings.csv
│   │   ├── umap_all_features/
│   │   │   ├── umap_coordinates.csv
│   │   │   └── umap_basic.png
│   │   └── umap_by_organelle/             # Per-organelle embeddings
│   │       ├── umap_mito.png
│   │       ├── umap_nuclei.png
│   │       └── ...
│   │
│   ├── 3_clustering/                      # Unsupervised clustering
│   │   ├── cluster_assignments/
│   │   │   ├── hdbscan_clusters.csv
│   │   │   ├── leiden_clusters.csv
│   │   │   └── kmeans_clusters.csv
│   │   ├── cluster_quality/
│   │   │   ├── silhouette_scores.png
│   │   │   └── cluster_sizes.png
│   │   ├── cluster_visualization/
│   │   │   ├── umap_hdbscan.png
│   │   │   ├── umap_leiden.png
│   │   │   └── umap_hdbscan_annotated.png
│   │   └── cluster_enrichment/
│   │       ├── gene_enrichment_per_cluster.csv
│   │       └── enriched_gene_highlights/
│   │
│   ├── 4_differential/                    # Comparison vs NTC
│   │   ├── ntc_comparison/
│   │   │   ├── zscore_top_features.png
│   │   │   ├── fold_change_top_features.png
│   │   │   └── differential_stats.csv
│   │   ├── volcano_plots/
│   │   │   ├── volcano_{feature}.png
│   │   │   └── ...
│   │   └── gene_highlights/
│   │       ├── umap_highlight_{gene}.png
│   │       └── ...
│   │
│   └── 5_summary/                         # Level summary
│       ├── cell_level_summary.csv         # Key metrics
│       ├── top_hits_cell_level.csv        # Preliminary hit list
│       └── cell_level_report.html         # Summary report
│
├── 2_guide_level/                         # Guide-aggregated analysis (~thousands)
│   ├── 1_qc/
│   │   ├── guide_cell_counts.png
│   │   ├── guide_consistency/             # Within-gene guide agreement
│   │   │   ├── guide_correlation.png
│   │   │   └── inconsistent_guides.csv
│   │   └── qc_summary.csv
│   │
│   ├── 2_embedding/
│   │   ├── umap_guide.png
│   │   ├── umap_by_gene_effect.png        # Colored by known essentiality
│   │   └── umap_coordinates.csv
│   │
│   ├── 3_clustering/
│   │   ├── cluster_assignments.csv
│   │   ├── umap_clustered.png
│   │   └── cluster_gene_composition.csv
│   │
│   ├── 4_differential/
│   │   ├── guide_vs_ntc_stats.csv
│   │   ├── feature_importance.png
│   │   └── guide_volcano.png
│   │
│   └── 5_summary/
│       ├── guide_level_summary.csv
│       └── flagged_guides.csv             # Inconsistent or low-quality guides
│
├── 3_gene_level/                          # Gene-aggregated analysis (~hundreds)
│   ├── 1_qc/
│   │   ├── gene_cell_counts.png
│   │   ├── gene_guide_counts.png
│   │   └── qc_summary.csv
│   │
│   ├── 2_embedding/
│   │   ├── umap_gene.png
│   │   ├── umap_gene_labeled.png          # All genes labeled
│   │   ├── umap_by_gene_effect.png
│   │   └── umap_coordinates.csv
│   │
│   ├── 3_clustering/
│   │   ├── cluster_assignments.csv
│   │   ├── umap_clustered.png
│   │   └── cluster_pathway_enrichment.csv # If pathway DB available
│   │
│   ├── 4_differential/
│   │   ├── gene_vs_ntc_stats.csv          # Main hit calling table
│   │   ├── feature_volcano/
│   │   │   ├── volcano_{feature}.png
│   │   │   └── ...
│   │   ├── organelle_contribution/
│   │   │   ├── contribution_heatmap.png
│   │   │   └── contribution_scores.csv
│   │   └── gene_rankings.csv              # Ranked by effect size
│   │
│   └── 5_summary/
│       ├── gene_level_summary.csv
│       ├── hit_list.csv                   # Final hit calls
│       └── gene_level_report.html
│
└── 4_integrated/                          # Cross-level integration
    ├── hit_summary/
    │   ├── final_hits.csv                 # Integrated hit list
    │   ├── hit_validation.csv             # Cell→Guide→Gene consistency
    │   └── hit_summary_plots/
    │
    ├── cross_level_correlation/
    │   ├── cell_vs_gene_correlation.png
    │   └── guide_consistency_vs_effect.png
    │
    └── reports/
        ├── full_analysis_report.html
        └── qc_report.html
```

---

## Analytical Flow

### Stage 1: Quality Control

**Purpose:** Identify and flag technical artifacts before biological interpretation.

```
┌─────────────────────────────────────────────────────────────────┐
│                     QC Stage (per level)                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐      │
│  │   Spatial    │    │    Batch     │    │   Feature    │      │
│  │    Drift     │    │   Effects    │    │   Quality    │      │
│  └──────┬───────┘    └──────┬───────┘    └──────┬───────┘      │
│         │                   │                   │               │
│         ▼                   ▼                   ▼               │
│  ┌─────────────────────────────────────────────────────┐       │
│  │              QC Summary & Flags                      │       │
│  │  • Edge cells flagged                                │       │
│  │  • Problematic wells identified                      │       │
│  │  • Low-variance features removed                     │       │
│  └─────────────────────────────────────────────────────┘       │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

**Outputs:**
- `qc_summary.csv` - QC metrics
- `flagged_items.csv` - Items to exclude or flag
- Drift/batch effect visualizations

**Chaining:** QC flags propagate to downstream stages and levels.

---

### Stage 2: Embedding

**Purpose:** Reduce dimensionality for visualization and clustering.

```
┌─────────────────────────────────────────────────────────────────┐
│                   Embedding Stage (per level)                   │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────────┐         ┌──────────────┐                     │
│  │     PCA      │────────▶│    UMAP      │                     │
│  │  (50 PCs)    │         │  (2D viz)    │                     │
│  └──────────────┘         └──────┬───────┘                     │
│                                  │                              │
│                    ┌─────────────┴─────────────┐               │
│                    ▼                           ▼               │
│           ┌──────────────┐           ┌──────────────┐          │
│           │ All Features │           │ Per-Organelle│          │
│           │    UMAP      │           │    UMAPs     │          │
│           └──────────────┘           └──────────────┘          │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

**Outputs:**
- `umap_coordinates.csv` - Saved coordinates
- Embedding saved to AnnData (`obsm['X_umap']`, `obsm['X_pca']`)
- Basic UMAP visualizations

**Chaining:** Embeddings are reused in clustering and differential visualization.

---

### Stage 3: Clustering

**Purpose:** Identify phenotypic subpopulations without labels.

```
┌─────────────────────────────────────────────────────────────────┐
│                   Clustering Stage (per level)                  │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│        ┌──────────────┐  ┌──────────────┐  ┌──────────────┐    │
│        │   HDBSCAN    │  │    Leiden    │  │    KMeans    │    │
│        │ (density)    │  │   (graph)    │  │   (elbow)    │    │
│        └──────┬───────┘  └──────┬───────┘  └──────┬───────┘    │
│               │                 │                 │             │
│               └────────────┬────┴─────────────────┘             │
│                            ▼                                    │
│                  ┌──────────────────┐                          │
│                  │ Cluster Quality  │                          │
│                  │   Assessment     │                          │
│                  └────────┬─────────┘                          │
│                           │                                     │
│               ┌───────────┴───────────┐                        │
│               ▼                       ▼                        │
│     ┌──────────────────┐    ┌──────────────────┐              │
│     │ Gene Enrichment  │    │ Cluster-colored  │              │
│     │  per Cluster     │    │     UMAPs        │              │
│     └──────────────────┘    └──────────────────┘              │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

**Outputs:**
- `cluster_assignments.csv` - Cluster labels per item
- Cluster quality metrics (silhouette, sizes)
- Annotated UMAP visualizations
- Gene enrichment tables

**Chaining:** Cluster enrichment identifies genes that co-localize phenotypically.

---

### Stage 4: Differential Analysis

**Purpose:** Quantify effects relative to NTC controls.

```
┌─────────────────────────────────────────────────────────────────┐
│                 Differential Stage (per level)                  │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                   NTC Comparison                         │   │
│  │  • Z-scores per feature                                  │   │
│  │  • Fold changes                                          │   │
│  │  • Statistical tests (t-test, Welch)                     │   │
│  └─────────────────────────────────────────────────────────┘   │
│                            │                                    │
│              ┌─────────────┼─────────────┐                     │
│              ▼             ▼             ▼                     │
│     ┌────────────┐  ┌────────────┐  ┌────────────┐            │
│     │  Feature   │  │  Volcano   │  │  Organelle │            │
│     │  Rankings  │  │   Plots    │  │Contribution│            │
│     └────────────┘  └────────────┘  └────────────┘            │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

**Outputs:**
- `differential_stats.csv` - Full statistics table
- Volcano plots per feature
- Feature rankings
- Organelle contribution analysis

**Chaining:** Differential statistics are used for hit calling.

---

### Stage 5: Organelle Discrimination (Cell Level Only)

**Purpose:** Compare segmentation groups' ability to discriminate gene knockouts.

**Key Question:** "If I only had features from one organelle, how well could I identify gene effects?"

```
┌─────────────────────────────────────────────────────────────────┐
│            Organelle Discrimination Stage (cell level)          │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  For each organelle feature set:                               │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │                                                          │  │
│  │  ┌────────────┐  ┌────────────┐  ┌────────────┐         │  │
│  │  │   Gene     │  │  Cluster-  │  │    KNN     │         │  │
│  │  │ Silhouette │  │  Gene NMI  │  │ Accuracy   │         │  │
│  │  └─────┬──────┘  └─────┬──────┘  └─────┬──────┘         │  │
│  │        │               │               │                 │  │
│  │        └───────────────┼───────────────┘                 │  │
│  │                        ▼                                 │  │
│  │               ┌────────────────┐                        │  │
│  │               │   Composite    │                        │  │
│  │               │     Score      │                        │  │
│  │               └────────────────┘                        │  │
│  └──────────────────────────────────────────────────────────┘  │
│                                                                 │
│  Output: Ranked organelles by discrimination power              │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

**Metrics:**
- **Gene Silhouette**: How well cells from same gene cluster together
- **Cluster-Gene NMI**: Do unsupervised clusters correspond to gene labels?
- **KNN Accuracy**: Can we predict gene from organelle features alone?
- **Distance Ratio**: Inter-gene vs intra-gene separation

**Outputs:**
- `organelle_discrimination_scores.csv` - All metrics per organelle
- Composite score ranking - Which organelle is best?
- Radar plot comparison - Visual comparison of top organelles

---

### Stage 6: Summary

**Purpose:** Aggregate results and generate reports.

```
┌─────────────────────────────────────────────────────────────────┐
│                    Summary Stage (per level)                    │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │                 Aggregate Statistics                      │  │
│  │  • Total items analyzed                                   │  │
│  │  • Items passing QC                                       │  │
│  │  • Number of clusters                                     │  │
│  │  • Items with significant effects                         │  │
│  └──────────────────────────────────────────────────────────┘  │
│                            │                                    │
│              ┌─────────────┼─────────────┐                     │
│              ▼             ▼             ▼                     │
│     ┌────────────┐  ┌────────────┐  ┌────────────┐            │
│     │  Summary   │  │  Hit List  │  │   Report   │            │
│     │   CSV      │  │ (prelim)   │  │   HTML     │            │
│     └────────────┘  └────────────┘  └────────────┘            │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

**Outputs:**
- Level summary statistics
- Preliminary hit lists
- HTML reports with key visualizations

---

## Cross-Level Chaining

### Cell → Guide Chaining

```
Cell Level Results                    Guide Level Uses
─────────────────                    ────────────────
• Cluster enrichment    ──────────▶  • Which guides cluster together?
• Per-cell QC flags     ──────────▶  • Guide reliability scores
• Feature importance    ──────────▶  • Same features analyzed
```

### Guide → Gene Chaining

```
Guide Level Results                   Gene Level Uses
───────────────────                  ───────────────
• Guide consistency     ──────────▶  • Weight guides by reliability
• Guide differential    ──────────▶  • Aggregate to gene level
• Flagged guides        ──────────▶  • Exclude from gene aggregation
```

### Gene Level Integration

```
All Levels Contribute                 Final Outputs
─────────────────────                ─────────────
• Cell: phenotype       ──────────▶  • Validated hits
• Guide: consistency    ──────────▶  • Confidence scores
• Gene: effect size     ──────────▶  • Final rankings
```

---

## Standard Analysis Suite

Each level runs the **same core analyses** for consistency:

| Analysis | Cell Level | Guide Level | Gene Level |
|----------|------------|-------------|------------|
| UMAP embedding | ✓ | ✓ | ✓ |
| Clustering (HDBSCAN/Leiden) | ✓ | ✓ | ✓ |
| NTC comparison (z-score) | ✓ | ✓ | ✓ |
| Volcano plots | ✓ | ✓ | ✓ |
| Feature importance | ✓ | ✓ | ✓ |
| Organelle contribution | ✓ | ✓ | ✓ |
| Per-organelle UMAPs | ✓ | ✓ | ✓ |

**Level-Specific Additions:**

| Level | Additional Analyses |
|-------|---------------------|
| Cell | Spatial drift, batch effects, gene highlights |
| Guide | Guide consistency, within-gene agreement |
| Gene | Gene labeling, pathway enrichment, final hit calling |

---

## Implementation: Unified Analysis Pipeline

```python
class LevelAnalysisPipeline:
    """Runs the full analysis suite for one level."""
    
    STAGES = [
        ("1_qc", QCStage),
        ("2_embedding", EmbeddingStage),
        ("3_clustering", ClusteringStage),
        ("4_differential", DifferentialStage),
        ("5_summary", SummaryStage),
    ]
    
    def run(self, level: str, upstream_results: dict = None):
        """Run all stages for a level, using upstream results for chaining."""
        results = {}
        for stage_name, StageClass in self.STAGES:
            stage = StageClass(self.data, self.config, level, upstream_results)
            results[stage_name] = stage.run()
        return results
```

```python
class FullPipeline:
    """Runs analysis across all levels with chaining."""
    
    def run(self):
        # Cell level (no upstream)
        cell_results = LevelAnalysisPipeline().run("cell")
        
        # Guide level (uses cell results)
        guide_results = LevelAnalysisPipeline().run("guide", upstream=cell_results)
        
        # Gene level (uses guide results)
        gene_results = LevelAnalysisPipeline().run("gene", upstream=guide_results)
        
        # Integrated analysis
        self.run_integration(cell_results, guide_results, gene_results)
```

---

## Key Metrics Tracked

### Per Level
- Number of items (cells/guides/genes)
- Items passing QC
- Number of clusters found
- Items significantly different from NTC
- Top features driving variance

### Cross Level
- Cell-to-gene consistency scores
- Guide reproducibility metrics
- Final hit confidence

---

## Configuration

```python
@dataclass
class AnalysisPipelineConfig:
    # Stages to run
    run_qc: bool = True
    run_embedding: bool = True
    run_clustering: bool = True
    run_differential: bool = True
    run_summary: bool = True
    
    # Levels to analyze
    levels: List[str] = field(default_factory=lambda: ["cell", "guide", "gene"])
    
    # Cross-level chaining
    chain_results: bool = True
    
    # QC thresholds
    min_cells_per_guide: int = 10
    min_guides_per_gene: int = 2
    
    # Differential thresholds
    zscore_threshold: float = 2.0
    pvalue_threshold: float = 0.05
```
