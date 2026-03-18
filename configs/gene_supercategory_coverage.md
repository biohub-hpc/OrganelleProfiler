# Gene Super-Category Coverage Analysis

Panel: 1,009 genes from `annotated_gene_panel_July2025.csv`

Three categorization approaches available via `--sources`:

| Source | Categories | Coverage | Mapping | Description |
|---|---|---|---|---|
| **chad** | 8 | ~23% (244 genes) | Single | CHAD v5 clusters only |
| **chad_boosted** | 8 | ~93% (986 genes) | Single | CHAD + keyword/regex/Harmonizome |
| **reactome_toplevel** | 29 | 78% (788 genes) | Multi | Reactome's own pathway ontology |

---

## 1. `chad` — CHAD v5 Only

CHAD v5 hierarchy covers ~393 manually curated genes, but many are positive controls
not in the 1009-gene panel. Only genes present in both CHAD and the panel get assigned.

| Category | Genes |
|---|---|
| Translation | 93 |
| Gene Expression | 47 |
| Cell Cycle & DNA | 43 |
| Membrane Trafficking | 20 |
| Metabolism | 13 |
| Signaling | 11 |
| Protein Homeostasis | 11 |
| Cytoskeleton & Morphology | 6 |
| **Other (unassigned)** | **818** |

---

## 2. `chad_boosted` — CHAD + Extra Annotations

Starts with CHAD clusters, then fills in gaps using:
1. **Reactome/GO keyword matching** from gene panel CSV annotations (+665 genes)
2. **Gene name regex patterns** like RPL*, COX*, KIF* (+3 genes after keywords)
3. **Harmonizome overrides** for 74 specifically-researched poorly-annotated genes (+74 genes)

| Category | Genes | % of panel |
|---|---|---|
| Gene Expression | 240 | 23.8% |
| Translation | 156 | 15.5% |
| Cell Cycle & DNA | 153 | 15.2% |
| Membrane Trafficking | 141 | 14.0% |
| Metabolism | 89 | 8.8% |
| Cytoskeleton & Morphology | 73 | 7.2% |
| Signaling | 70 | 6.9% |
| Protein Homeostasis | 64 | 6.3% |
| **Other (unassigned)** | **76** | **7.5%** |

### Boost breakdown

| Step | Cumulative assigned | Added |
|---|---|---|
| CHAD clusters | 244 | 244 |
| + Reactome/GO keywords | 909 | +665 |
| + Regex patterns | 912 | +3 |
| + Harmonizome overrides | 986 | +74 |

### Harmonizome overrides

74 genes validated via Harmonizome REST API functional descriptions and cross-checked
against Enrichr (GO_Biological_Process_2023, KEGG_2021, Reactome_2022). Applied only
to genes not covered by the first three steps. See `_HARMONIZOME_OVERRIDES` dict in
`gene_supercategories.py` for the full list.

### Remaining unassigned (~76 genes)

Includes ~23 genes that are genuinely uncharacterized (ANKRD20A3, CBWD1-3, MROH6, etc.)
plus ~53 genes that fell through all annotation steps.

---

## 3. `reactome_toplevel` — Reactome's Own 29 Categories

Uses Reactome's top-level biological process hierarchy directly. Genes can belong to
**multiple categories** (multi-mapping, avg 2.9 categories/gene).

- **788/1009 genes mapped (78.1%)**
- **221 genes unmapped** (no NCBI ID in Reactome, or no pathway annotation)
- **29 Reactome top-level categories**

| Reactome Category | Genes |
|---|---|
| Metabolism of proteins | 271 |
| Disease | 267 |
| Signal Transduction | 191 |
| Immune System | 178 |
| Metabolism of RNA | 167 |
| Metabolism | 163 |
| Gene expression (Transcription) | 152 |
| Developmental Biology | 145 |
| Cellular responses to stimuli | 129 |
| Vesicle-mediated transport | 106 |
| Cell Cycle | 103 |
| DNA Repair | 52 |
| Transport of small molecules | 46 |
| Hemostasis | 39 |
| Organelle biogenesis and maintenance | 36 |
| Cell-Cell communication | 33 |
| DNA Replication | 31 |
| Autophagy | 31 |
| Chromatin organization | 29 |
| Neuronal System | 27 |
| Programmed Cell Death | 24 |
| Protein localization | 23 |
| Circadian clock | 19 |
| Sensory Perception | 17 |
| Reproduction | 14 |
| Extracellular matrix organization | 13 |
| Muscle contraction | 12 |
| Drug ADME | 4 |
| Digestion and absorption | 1 |

### Data files

Pre-computed panel-specific files at:
- `/hpc/projects/icd.fast.ops/configs/ontologies/reactome/top_level_pathways.tsv`
- `/hpc/projects/icd.fast.ops/configs/ontologies/reactome/panel_gene_top_level_pathways.tsv`
- `/hpc/projects/icd.fast.ops/configs/ontologies/reactome/panel_gene_reactome_categories.tsv`

---

## Comparison

| Property | chad | chad_boosted | reactome_toplevel |
|---|---|---|---|
| **Categories** | 8 | 8 | 29 |
| **Coverage** | 23% | 93% | 78% |
| **Mapping** | Single | Single | Multi (avg 2.9/gene) |
| **Category sizes** | 6-93 | 64-240 | 1-271 |
| **Maintenance** | Auto (CHAD YAML) | Manual (YAML + code) | Auto (Reactome DB) |
| **Radar readability** | 8 axes, sparse | 8 axes, full | 29 axes, detailed |

### When to use which

- **chad**: Quick look using only manually curated CHAD clusters. Low coverage means many
  genes excluded — best for focused analysis on well-characterized biology.
- **chad_boosted** (default): High-coverage 8-axis radar. Best for comparing reporters at
  a high level with near-complete gene coverage.
- **reactome_toplevel**: Detailed 29-category biological profiling using Reactome's own
  ontology. Multi-mapping captures biological reality. Best for characterizing what specific
  biology a reporter sees.

---

## Implementation

```bash
# Default: chad_boosted (8 categories, ~93% coverage)
--sources chad_boosted

# CHAD only (8 categories, ~23% coverage)
--sources chad

# Reactome top-level (29 categories, 78% coverage, multi-mapped)
--sources reactome_toplevel
```

Each source saves to `sources_{name}/` so runs don't overwrite each other.

---

## Other Databases Investigated

### PANTHER Protein Classes (23 top-level categories)
- Downloaded to `/hpc/projects/icd.fast.ops/configs/ontologies/panther/`
- Coverage: 728/1009 (72.2%) with symbol rescue via dep_map_gene_name
- Categories are structural/functional (e.g., "kinase", "transporter") not process-based
- Not integrated — less suitable for biological process radar plots

### Harmonizome High-Level Categories
- No built-in high-level system (only hgncRootFamilies: 309 families, incomplete)
- Used instead as per-gene overrides in chad_boosted (74 genes)

### Enrichr
- Tested as validation source — high agreement with Harmonizome
- Not integrated standalone — Reactome top-level is better for database-driven approach
