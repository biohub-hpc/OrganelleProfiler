# Gene Super-Category Coverage

Panel: ~1,009 genes from `annotated_gene_panel_July2025.csv`

---

## The 8 CHAD-derived categories — origin and design

The 8 categories used by `chad` and `chad_boosted` are **manually defined buckets
designed to match what OPS live-cell reporters can actually detect**. They were not
derived from CHAD — rather, they were designed first by the team to cover the major
biological axes visible in an OPS screen (organelle identity, trafficking, metabolic
state, proliferation, etc.), and then CHAD v5 cluster names were mapped into them.

CHAD v5 is a hierarchy of manually curated gene sets representing known protein
complexes and pathways (positive controls used in OPS screening). Each CHAD cluster
has a name (e.g. "Proteasome", "mTOR", "DNA Replication"). The 8-category
design assigns each named CHAD cluster to the most appropriate bucket:

| Category | What it captures | Example CHAD clusters mapped in |
|---|---|---|
| **Translation** | Ribosomes, tRNA, translation initiation | ribosome 40s/60s, tRNA synthetase, eIF2 |
| **Gene Expression** | Transcription, splicing, RNA processing | RNA Polymerase, Spliceosome, mediator |
| **Cell Cycle & DNA** | Cell cycle, replication, repair, mitosis | DNA Replication, Replication fork, MuvB |
| **Signaling** | Kinase cascades, receptor signaling | mTOR, KRAS |
| **Membrane Trafficking** | COPI/COPII, SNARE, dynein, endosomes | GOLGI to ER Transport, SRP, dynein-dynactin |
| **Metabolism** | ETC, mitochondria, lipid/amino acid metabolism | electronic transport chain, mitochondria protein import |
| **Protein Homeostasis** | Proteasome, ubiquitin, ER quality control | Proteasome, Ufmylation 60s |
| **Cytoskeleton & Morphology** | Actin, focal adhesion, cell shape | Focal adhesion |

The mapping is defined in `gene_supercategory_mapping.yaml`. Each category entry
lists the CHAD cluster names that belong to it — this is the authoritative source
of which clusters map where.

---

## How genes end up in a category — the 4-pass pipeline

### `chad` (pass 1 only, ~24% coverage)

Only genes that are explicit members of a named CHAD cluster get assigned. The
CHAD YAML is traversed recursively; any gene found under a cluster listed in the
category's `chad_clusters` is assigned to that category. First assignment wins
(no gene gets two categories).

### `chad_boosted` (all 4 passes, ~98% coverage)

Runs the same pass 1, then applies 3 additional passes **only to genes still
unassigned**. First match across all passes wins — more curated evidence always
takes priority over less curated.

**Pass 1 — CHAD clusters** (245 genes)
Same as above. Only genes in explicitly named CHAD clusters.

**Pass 2 — Reactome/GO keyword matching** (+664 genes)
Each gene in the panel CSV has `In_REACT_pathways` and `In_go_pathways` columns —
free-text concatenations of all Reactome pathway names and GO terms that gene
belongs to. These are lowercased and scored against each category's
`pathway_keywords` list (e.g. "ubiquitin", "vesicle", "mitochondri"). The
category with the most keyword hits wins if score ≥ 1. This is where **66% of
the panel** gets assigned — the bulk of OPS genes have Reactome/GO annotations
even if they don't appear in CHAD.

**Pass 3 — Gene name regex** (+3 genes)
Applies compiled regular expressions (e.g. `^RPL\d`, `^PSM[ABCD]`, `^NDUF`)
to gene names for genes that somehow have no pathway annotation. Only assigns
3 genes in practice — catches well-named gene families that are poorly annotated
(e.g. some mitoribosomal genes).

**Pass 4 — Harmonizome overrides** (+74 genes)
A hardcoded dict (`_HARMONIZOME_OVERRIDES` in `gene_supercategories.py`) of
74 specific genes manually assigned by querying the Harmonizome REST API and
cross-checking against Enrichr (GO_Biological_Process_2023, KEGG_2021,
Reactome_2022). Applied to genes that pass through all 3 earlier steps without
assignment — mostly poorly characterized or recently renamed genes.

**Remaining unassigned: 76 genes**
~23 are genuinely uncharacterized (ANKRD20A3, CBWD1-3, MROH6...), ~53 have
annotations that don't match any category keyword.

---

## Coverage comparison — all 4 sources

| Source | Categories | n genes assigned | Coverage | Cats/gene | Mapping |
|---|---|---|---|---|---|
| `chad` | 8 | 245 | 24% | 1.0 | single |
| `chad_boosted` | 8 | 986 | 98% | 1.0 | single |
| `reactome_toplevel` | 29 | 788 | 78% | 2.95 avg (max 19) | multi |
| `reactome_cell_biology` | 17 | 732 | 73% | 2.17 avg (max 15) | multi |

---

## Per-category gene counts

### chad / chad_boosted (8 categories)

| Category | chad | chad_boosted |
|---|---|---|
| Translation | 94 | 156 |
| Gene Expression | 47 | 240 |
| Cell Cycle & DNA | 43 | 153 |
| Membrane Trafficking | 20 | 141 |
| Metabolism | 13 | 89 |
| Signaling | 11 | 70 |
| Protein Homeostasis | 11 | 64 |
| Cytoskeleton & Morphology | 6 | 73 |
| **Other (unassigned)** | **818** | **76** |

`chad` is heavily biased toward Translation/Gene Expression because ribosomal and
spliceosomal complexes are well-represented in CHAD. `chad_boosted` balances the
categories substantially — Cytoskeleton goes from 6 → 73 genes once
actin/tubulin/focal-adhesion GO terms are included.

### chad_boosted pass breakdown

| Pass | Genes assigned | Cumulative |
|---|---|---|
| 1. CHAD clusters | 245 | 245 |
| 2. Reactome/GO keyword matching | +664 | 909 |
| 3. Gene name regex | +3 | 912 |
| 4. Harmonizome overrides | +74 | 986 |

### reactome_toplevel (29 categories, multi-mapped)

| Category | n genes | cell_bio ✓ |
|---|---|---|
| Metabolism of proteins | 271 | ✓ |
| Disease | 267 | ✗ |
| Signal Transduction | 191 | ✓ |
| Immune System | 178 | ✗ |
| Metabolism of RNA | 167 | ✓ |
| Metabolism | 163 | ✓ |
| Gene expression (Transcription) | 152 | ✓ |
| Developmental Biology | 145 | ✗ |
| Cellular responses to stimuli | 129 | ✓ |
| Vesicle-mediated transport | 106 | ✓ |
| Cell Cycle | 103 | ✓ |
| DNA Repair | 52 | ✓ |
| Transport of small molecules | 46 | ✓ |
| Hemostasis | 39 | ✗ |
| Organelle biogenesis and maintenance | 36 | ✓ |
| Cell-Cell communication | 33 | ✓ |
| DNA Replication | 31 | ✓ |
| Autophagy | 31 | ✓ |
| Chromatin organization | 29 | ✓ |
| Neuronal System | 27 | ✗ |
| Programmed Cell Death | 24 | ✓ |
| Protein localization | 23 | ✓ |
| Circadian clock | 19 | ✗ |
| Sensory Perception | 17 | ✗ |
| Reproduction | 14 | ✗ |
| Extracellular matrix organization | 13 | ✗ |
| Muscle contraction | 12 | ✗ |
| Drug ADME | 4 | ✗ |
| Digestion and absorption | 1 | ✗ |

`reactome_cell_biology` retains the 17 marked ✓. See
`reactome_cell_biology_filter.md` for full exclusion rationale.

---

## Key design differences between the systems

| | chad/chad_boosted | reactome_toplevel/cell_bio |
|---|---|---|
| **Category origin** | Manually designed to match OPS biology | Reactome's own top-level hierarchy |
| **Assignment** | Single best category (first match wins) | Multi-mapping (gene belongs to all matching categories) |
| **Keyword source** | Gene's own Reactome+GO annotations from panel CSV | Pathway membership pre-computed from Reactome DB |
| **Granularity** | 8 broad buckets, OPS-focused | 17–29 finer process categories |
| **Unassigned** | 76 genes (7%) | 221–277 genes (22–27%) |
| **Maintenance** | Manual YAML + hardcoded overrides | Automatic from Reactome DB |

---

## Data sources

- CHAD v5: `/hpc/projects/icd.ops/configs/gene_clusters/chad_positive_controls_v5_hierarchy.yml`
- Gene panel + annotations: `/hpc/projects/intracellular_dashboard/ops/configs/annotated_gene_panel_July2025.csv`
- Reactome pre-computed: `/hpc/projects/icd.fast.ops/configs/ontologies/reactome/panel_gene_reactome_categories.tsv`
- Category mapping config: `gene_supercategory_mapping.yaml`
- Cell biology filter: `reactome_cell_biology_filter.md`
- Harmonizome overrides: `_HARMONIZOME_OVERRIDES` dict in `ops_utils/analysis/gene_supercategories.py`
