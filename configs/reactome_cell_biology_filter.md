# Reactome Top-Level: Cell Biology Filter

Defines which of the 29 Reactome top-level categories are included in the
`reactome_cell_biology` source. The goal is to retain categories directly
relevant to the biology captured by live-cell OPS reporters (organelle
morphology, trafficking, metabolism, stress responses, cell cycle) and
exclude categories that reflect tissue/organism-level biology not observable
in a cultured cell line screen.

Gene counts are from the OPS ~1009-gene panel (multi-mapped, so genes can
appear in multiple categories).

---

## Included (17)

| Category | n genes | Reason |
|---|---|---|
| Autophagy | 31 | Core organelle quality-control process directly observable via live reporters |
| Cell Cycle | 103 | Fundamental cell biology; strong OPS phenotype via nuclear/cytoskeletal reporters |
| Cellular responses to stimuli | 129 | Broad stress/signaling responses — captures reporter-relevant perturbation biology |
| Chromatin organization | 29 | Nuclear architecture directly visible in OPS; histone/chromatin regulators |
| DNA Repair | 52 | Nuclear stress response; large category with clear OPS phenotypes |
| DNA Replication | 31 | S-phase biology; strong overlap with cell cycle reporter phenotypes |
| Gene expression (Transcription) | 152 | Core regulatory machinery; many OPS-active perturbations |
| Metabolism | 163 | Mitochondrial, lipid, and metabolic reporters directly measure this |
| Metabolism of proteins | 271 | Proteasome, glycosylation, folding — core to ER/Golgi/lysosome reporter biology |
| Metabolism of RNA | 167 | RNA processing machinery; large category with real OPS signal |
| Organelle biogenesis and maintenance | 36 | Directly targeted by OPS reporter design — highest biological relevance |
| Programmed Cell Death | 24 | Apoptosis/ferroptosis visible via mitochondrial and membrane reporters |
| Protein localization | 23 | Targeting and sorting machinery; directly relevant to trafficking reporters |
| Signal Transduction | 191 | Kinase/GPCR/mTOR cascades well-represented in OPS panel |
| Transport of small molecules | 46 | Ion channels, transporters — relevant to organelle homeostasis reporters |
| Vesicle-mediated transport | 106 | COPI/COPII/SNARE/endosomal trafficking — core to multiple OPS reporters |
| Cell-Cell communication | 33 | Secretory pathway, gap junctions — partially relevant; borderline but included |

---

## Excluded (12)

| Category | n genes | Reason |
|---|---|---|
| Circadian clock | 19 | Tissue-level oscillator; not observable in asynchronous cultured cell screens |
| Developmental Biology | 145 | Organism/tissue patterning (Wnt, Notch, Hedgehog morphogens); not OPS-relevant biology |
| Digestion and absorption | 1 | Gut epithelium physiology; 1 gene — not relevant |
| Disease | 267 | Catch-all disease annotation — not a biological process, inflates gene counts artificially |
| Drug ADME | 4 | Pharmacokinetic category; not relevant to cell biology screen |
| Extracellular matrix organization | 13 | ECM remodeling in tissue context; minimal OPS signal in 2D culture |
| Hemostasis | 39 | Platelet/coagulation biology; not expressed/relevant in OPS cell lines |
| Immune System | 178 | Immune cell-specific biology; large but not relevant to epithelial OPS cell lines |
| Muscle contraction | 12 | Sarcomere biology; not relevant |
| Neuronal System | 27 | Neuron-specific biology; not relevant to OPS cell lines |
| Reproduction | 14 | Germ cell/meiosis biology; not relevant |
| Sensory Perception | 17 | Sensory neuron biology; not relevant |

---

## Summary

- **Included**: 17 categories, covers the core cell biology observable in OPS
- **Excluded**: 12 categories, all tissue/organism-specific or non-biological catch-alls
- Disease and Developmental Biology are the largest excluded categories (267 and 145 genes)
  but both are annotation artifacts in a cell line context rather than true biological processes
