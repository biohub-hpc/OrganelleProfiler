# Feature Graphs Architecture (v3.0)

## Overview

The `fe_graphs` package provides a modular, hierarchical framework for phenotypic analysis of pooled optical screens. It generates UMAP embeddings, clustering, and statistical visualizations from feature extraction data.

**Key Design Principles:**
1. **Level-based hierarchy**: Cell → Guide → Gene aggregation
2. **Stage-based pipeline**: QC → Embedding → Clustering → Differential → Summary
3. **Cross-level chaining**: Results propagate between levels
4. **Consistent analysis suite**: Same stages run at every level

---

## Architecture

```
┌────────────────────────────────────────────────────────────────────┐
│                      AnalysisOrchestrator                          │
│  Coordinates analysis across all levels with result chaining       │
└─────────────────────────────┬──────────────────────────────────────┘
                              │
          ┌───────────────────┼───────────────────┐
          ▼                   ▼                   ▼
┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐
│  LevelPipeline  │  │  LevelPipeline  │  │  LevelPipeline  │
│    (cell)       │──│    (guide)      │──│    (gene)       │
│                 │  │                 │  │                 │
│  ┌───────────┐  │  │  ┌───────────┐  │  │  ┌───────────┐  │
│  │ QCStage   │  │  │  │ QCStage   │  │  │  │ QCStage   │  │
│  ├───────────┤  │  │  ├───────────┤  │  │  ├───────────┤  │
│  │ Embedding │  │  │  │ Embedding │  │  │  │ Embedding │  │
│  ├───────────┤  │  │  ├───────────┤  │  │  ├───────────┤  │
│  │ Clustering│  │  │  │ Clustering│  │  │  │ Clustering│  │
│  ├───────────┤  │  │  ├───────────┤  │  │  ├───────────┤  │
│  │Differential│ │  │  │Differential│ │  │  │Differential│ │
│  ├───────────┤  │  │  ├───────────┤  │  │  ├───────────┤  │
│  │ Summary   │  │  │  │ Summary   │  │  │  │ Summary   │  │
│  └───────────┘  │  │  └───────────┘  │  │  └───────────┘  │
└─────────────────┘  └─────────────────┘  └─────────────────┘
          │                   │                   │
          └───────────────────┼───────────────────┘
                              ▼
                  ┌─────────────────────┐
                  │ Integrated Analysis │
                  │  (cross-level)      │
                  └─────────────────────┘
```

---

## Directory Structure

```
fe_graphs/
├── __init__.py                      # Package exports
├── fe_graphs_config.py              # Configuration dataclasses
├── fe_graphs_orchestrator.py        # Main entry point
├── fe_graphs_level_pipeline.py      # Per-level pipeline runner
│
├── core/                            # Reusable utilities
│   ├── __init__.py
│   ├── fe_graphs_embedding.py       # UMAP/PCA with GPU support
│   ├── fe_graphs_clustering.py      # HDBSCAN/KMeans/Leiden
│   ├── fe_graphs_cache.py           # Persistent result caching
│   └── fe_graphs_data_loader.py     # Data loading and context
│
├── stages/                          # Analysis stages
│   ├── __init__.py
│   ├── fe_graphs_stage_base.py      # Base stage class
│   ├── fe_graphs_qc_stage.py        # Quality control
│   ├── fe_graphs_embedding_stage.py # Dimensionality reduction
│   ├── fe_graphs_clustering_stage.py# Unsupervised clustering
│   ├── fe_graphs_differential_stage.py # NTC comparison
│   └── fe_graphs_summary_stage.py   # Result aggregation
│
├── plotting/                        # Plotting utilities
│   ├── __init__.py
│   ├── fe_graphs_utils.py           # Common plotting functions
│   ├── fe_graphs_umap_plots.py      # UMAP visualizations
│   ├── fe_graphs_heatmaps.py        # Heatmap functions
│   └── fe_graphs_statistical_plots.py # Volcano, bar charts
│
├── analysis/                        # Legacy analyzers (deprecated)
│   └── ...
│
├── FE_GRAPHS_ARCHITECTURE.md        # This file
└── FE_GRAPHS_ANALYSIS_DESIGN.md     # Analysis flow design
```

---

## Output Organization

```
{experiment}/4-features/graphs/
│
├── 1_cell_level/                     # Individual cell analysis
│   ├── 1_qc/
│   │   ├── batch_effects/            # Well distribution
│   │   └── feature_quality/          # Variance, correlations
│   │
│   ├── 2_spatial_drift/              # Radial position effects (cell level)
│   │   ├── radial_drift_*.png        # Per-feature drift curves
│   │   └── inflection_summary.png    # Edge effect positions
│   │
│   ├── 3_embedding/
│   │   ├── pca/                      # Variance explained
│   │   ├── umap_all_features/        # Main UMAP
│   │   └── umap_by_organelle/        # Per-organelle UMAPs
│   │
│   ├── 4_positive_controls/          # EARLY validation! (all levels)
│   │   ├── positive_controls_all_embeddings.png  # <<< Single canvas!
│   │   ├── cluster_cohesion_scores.csv
│   │   ├── cluster_cohesion_summary.png
│   │   ├── organelle_cohesion_comparison.csv
│   │   ├── organelle_cohesion_comparison.png
│   │   └── cluster_highlights/
│   │       ├── all_clusters_overview.png
│   │       └── cluster_{name}.png
│   │
│   ├── 5_clustering/
│   │   ├── cluster_assignments/      # Labels per method
│   │   ├── cluster_quality/          # Silhouette scores
│   │   ├── cluster_visualization/    # Colored UMAPs
│   │   └── cluster_enrichment/       # Gene enrichment
│   │
│   ├── 6_differential/
│   │   ├── ntc_comparison/           # Z-scores, fold changes
│   │   ├── volcano_plots/            # Per-feature volcanos
│   │   ├── organelle_contribution/   # Per-organelle effects
│   │   └── gene_highlights/          # Top gene UMAPs
│   │
│   ├── 7_organelle_discrimination/   # Cell level only
│   │   ├── organelle_discrimination_scores.csv
│   │   ├── organelle_discrimination_composite.png
│   │   ├── organelle_discrimination_heatmap.png
│   │   └── organelle_discrimination_radar.png
│   │
│   └── 8_summary/
│       ├── cell_level_summary.csv
│       └── top_hits_cell_level.csv
│
├── 2_guide_level/                    # Guide-aggregated
│   └── [same structure as cell]
│
├── 3_gene_level/                     # Gene-aggregated
│   └── [same structure as cell]
│
└── 4_integrated/                     # Cross-level integration
    ├── hit_summary/
    │   └── final_hits.csv
    └── cross_level_correlation/
```

---

## Analysis Stages

### Stage 1: QC (Quality Control)

**Purpose:** Identify technical artifacts before biological interpretation.

| Analysis | Cell | Guide | Gene |
|----------|------|-------|------|
| Spatial drift | ✓ | - | - |
| Batch effects | ✓ | - | - |
| Feature quality | ✓ | ✓ | ✓ |
| Item counts | ✓ | ✓ | ✓ |
| Guide consistency | - | ✓ | - |

**Key Outputs:**
- `qc_summary.csv` - QC metrics
- `spatial_drift/` - Edge effect visualizations
- `guide_consistency/` - Within-gene agreement

---

### Stage 2: Spatial Drift (Cell Level Only)

**Purpose:** Analyze radial position effects (well-effect) that can confound biological signal.

**Methods:**
- Three-segment regression to find inflection points
- Segmented regression (two-segment linear)
- Second derivative / LOWESS smoothing

**Key Outputs:**
- `radial_drift_analysis.png` - Per-feature drift curves
- `inflection_summary.png` - Distribution of edge effect positions

---

### Stage 3: Embedding

**Purpose:** Reduce dimensionality for visualization and downstream analysis.

**Methods:**
- **PCA**: Variance explained analysis (up to 250 components)
- **UMAP**: 2D embedding for visualization

**Per-organelle UMAPs:**
- Separate UMAPs for mitochondria, nuclei, phase, etc.
- Reveals organelle-specific phenotypes

**Key Outputs:**
- `pca/variance_explained.png`
- `pca/pca_loadings.csv`
- `umap_coordinates.csv`
- `umap_by_organelle/`

---

### Stage 4: Positive Controls Validation (All Levels) - EARLY!

**Purpose:** Validate embeddings using known gene functional clusters IMMEDIATELY after embedding.

**Key Question:** "Do genes known to function together cluster together?"

This stage runs right after embedding to provide **early biological validation** of your feature space before clustering.

**Known Clusters (from `chad_positive_controls_v3.yml`):**
- Proteasome subunits (19S, 20S)
- Ribosome subunits (40S, 60S, mitochondrial)
- RNA Polymerase complexes (I, II, III)
- Splicing machinery (Sm ring, LSm ring, SF3B)
- Transport complexes (COPI, TRAPP, SEC61)

**Key Outputs:**
- `positive_controls_all_embeddings.png` - **Single canvas** showing all clusters on all UMAP types
- `cluster_cohesion_scores.csv` - Per-cluster tightness metrics
- `organelle_cohesion_comparison.csv` - Which organelle best captures known biology
- `cluster_highlights/` - Individual cluster highlight plots

---

### Stage 5: Clustering

**Purpose:** Identify phenotypic subpopulations without labels.

**Methods:**
- **HDBSCAN**: Density-based, handles noise
- **KMeans**: Optimal k via elbow method
- **Leiden**: Graph-based community detection

**Gene Enrichment:**
- Fisher's exact test per cluster
- Identifies genes overrepresented in each phenotypic state

**Key Outputs:**
- `cluster_assignments/{method}_clusters.csv`
- `cluster_quality/silhouette_scores.png`
- `cluster_enrichment/gene_enrichment_per_cluster.csv`

---

### Stage 6: Differential Analysis

**Purpose:** Quantify effects relative to NTC controls.

**Metrics:**
- **Z-score**: (perturbed_mean - NTC_mean) / NTC_std
- **Fold change**: perturbed_mean / NTC_mean
- **P-value**: Welch's t-test
- **FDR**: Benjamini-Hochberg correction

**Visualizations:**
- Volcano plots per feature
- Top features by z-score
- Organelle contribution heatmaps

**Key Outputs:**
- `differential_stats.csv` - Full statistics table
- `volcano_plots/`
- `organelle_contribution/`

---

### Stage 7: Organelle Discrimination (Cell Level Only)

**Purpose:** Compare segmentation groups on their ability to discriminate gene KOs.

**Key Question:** "Which organelle best separates different gene knockouts?"

**Metrics computed per organelle:**
- **Gene Silhouette Score**: How well cells from the same gene cluster together
- **Cluster-Gene NMI**: Mutual information between clusters and gene labels
- **KNN Classification Accuracy**: Can we predict gene from organelle features?
- **Distance Ratio**: Inter-gene distance / intra-gene distance

**Key Outputs:**
- `organelle_discrimination_scores.csv` - All metrics per organelle
- `organelle_discrimination_composite.png` - Ranked bar chart
- `organelle_discrimination_heatmap.png` - All metrics comparison
- `organelle_discrimination_radar.png` - Radar plot of top organelles

---

### Stage 8: Summary

**Purpose:** Aggregate results and generate hit lists.

**Outputs:**
- Level summary statistics
- Preliminary hit lists (threshold: |z| > 2, p_adj < 0.05)
- Metrics from all stages

---

## Cross-Level Chaining

Results propagate between levels:

```
Cell Level          Guide Level         Gene Level
──────────          ───────────         ──────────
QC flags      ──▶   Weighted            Final hit
Cluster             aggregation         confidence
enrichment    ──▶   Guide               scores
                    consistency   ──▶   Hit
                                        validation
```

---

## Usage

### Command Line (via launcher)

```bash
# Full pipeline
python -m organelle_profiler.feature_extraction.fe_graphs_v2 ops0094_20251217

# Specific levels
python -m organelle_profiler.feature_extraction.fe_graphs_v2 ops0094 --levels cell gene

# Skip stages
python -m organelle_profiler.feature_extraction.fe_graphs_v2 ops0094 --skip-stages qc
```

### Programmatic

```python
from organelle_profiler.feature_extraction.fe_graphs import AnalysisOrchestrator

# Full pipeline
orchestrator = AnalysisOrchestrator("ops0094_20251217")
result = orchestrator.run()

# Specific levels
result = orchestrator.run(levels=["gene"])

# Skip stages
result = orchestrator.run(skip_stages=["clustering"])

# Access results
for level, level_result in result.level_results.items():
    print(f"{level}: {len(level_result.get_all_files())} files")
```

### Individual Level Pipeline

```python
from organelle_profiler.feature_extraction.fe_graphs import LevelPipeline
from organelle_profiler.fe_graphs.core import DataLoader

# Load data
loader = DataLoader("ops0094_20251217")
data_context = loader.load()

# Run single level
pipeline = LevelPipeline(
    data_context=data_context,
    config=GraphConfig(experiment="ops0094_20251217"),
    level="cell",
)
level_result = pipeline.run()
```

---

## Configuration

### GraphConfig

```python
GraphConfig(
    experiment="ops0094_20251217",
    debug_cell_fraction=None,     # Sample fraction for testing
    use_cuml=True,                # GPU acceleration
    cluster_algo="all",           # "all", "hdbscan", "kmeans", "leiden"
    analysis_mode="all",          # "all", "guide", "gene", "guide_and_gene"
)
```

### AnalysisConfig

```python
AnalysisConfig(
    # Clustering
    min_cluster_size=25,
    kmeans_k_range=(2, 31),
    leiden_resolution=1.0,
    
    # UMAP
    umap_n_neighbors=15,
    umap_min_dist=0.1,
    
    # Differential
    zscore_threshold=2.0,
    pvalue_threshold=0.05,
)
```

---

## Performance Considerations

### GPU Acceleration

- **cuML UMAP**: ~10x faster than sklearn
- **cuML HDBSCAN**: ~5x faster on GPU
- Automatic CPU fallback if GPU unavailable

### Caching

- UMAP embeddings cached to `.npz` files
- Cluster assignments cached alongside embeddings
- Cache invalidated when cell count changes

### Parallelization

- Gene enrichment computed in parallel (joblib)
- Per-organelle UMAPs can run in parallel

---

## Dependencies

Required:
- numpy, pandas, scipy
- scikit-learn
- matplotlib, seaborn
- anndata
- umap-learn
- hdbscan
- leidenalg, igraph

Optional (GPU):
- cuml
- cudf

---

## Migration from v2.0

The v3.0 architecture replaces the analyzer-based approach with a stage-based pipeline:

| v2.0 | v3.0 |
|------|------|
| `CellLevelAnalyzer` | `LevelPipeline(level="cell")` |
| `AggregatedLevelAnalyzer` | `LevelPipeline(level="guide/gene")` |
| `SpatialDriftAnalyzer` | `QCStage` |
| `VolcanoAnalyzer` | `DifferentialStage` |

The original `fe_graphs.py` monolithic script is preserved for backwards compatibility.
