# benchmarking

Code and reproducibility package for the paper:

**Benchmarking Posted Quotes under Transaction-Price Opacity: A Computational Market Approach for Data Asset Markets**

## Overview

This repository provides an open-source reproducibility package for the main empirical outputs of the paper, including:

- descriptive statistics and supplier-anchoring evidence,
- benchmark-construction results under KFold and supplier cold-start settings,
- Appendix A residual diagnostics for the Gaussian working approximation,
- consolidated spreadsheet outputs for Tables 1–5 and Appendix B1–B4.

The project implements a computational market-approach benchmarking framework for posted quotes in opaque data-asset markets. The workflow combines:

1. descriptive evidence on supplier anchoring,
2. knowledge-graph-grounded multimodal representations,
3. global benchmark estimation,
4. local comparable retrieval with KNN and GapTrim,
5. Bayesian reconciliation and predictive-interval reporting.

## Repository structure

```text
benchmarking/
├── README.md
├── run_paper_all_from_raw_standalone_with_excel.py
├── appendix_residual_diagnostics.py
├── price_files/
│   └── dataproduct_industry_analysis_list_format_anymous.xlsx
├── export_neo4j/
│   └── api_data_snapshot.csv
└── paper_result_final/
    ├── appendix_residual_diagnostics/
    │   ├── A1_Residuals_KFold.csv
    │   ├── A2_Residuals_GroupKFold.csv
    │   ├── Residual_Diagnostics_KFold.png
    │   ├── Residual_Diagnostics_GroupKFold.png
    │   ├── Residual_Diagnostics_KFold.pdf
    │   ├── Residual_Diagnostics_GroupKFold.pdf
    │   ├── Residual_Normality_Summary.csv
    │   ├── Appendix_Table_A1_Gaussian_Working_Approximation.csv
    │   └── Residual_Diagnostics.xlsx
    ├── paper_results_figures/
    │   └── [selected final figures used in the paper]
    └── paper_results_main_appendix/
        └── All_Tables_1_5_and_Appendix_B_1_4.xlsx
```

## Main files

### `run_paper_all_from_raw_standalone_with_excel.py`
Main end-to-end script for reproducing the paper’s core outputs. It:

- reads the anonymized descriptive and benchmarking input files,
- generates the main descriptive statistics and figures,
- runs the multimodal benchmarking pipeline,
- evaluates KFold and Group KFold performance,
- exports the main and appendix tables,
- writes a consolidated Excel workbook for Tables 1–5 and Appendix B1–B4.

### `appendix_residual_diagnostics.py`
Standalone script for Appendix A. It:

- rebuilds a paper-facing Appendix Table A1 from cached diagnostic outputs by default,
- exports an Excel workbook collecting the appendix residual-diagnostic results,
- optionally re-estimates the global prior model under KFold and Group KFold,
- collects pooled out-of-fold residuals,
- generates residual histograms and normal Q–Q plots,
- computes moment diagnostics and formal normality statistics.

## Input files

### `price_files/dataproduct_industry_analysis_list_format_anymous.xlsx`
An anonymized descriptive-input workbook used for descriptive statistics and supplier-anchoring visualization.

### `export_neo4j/api_data_snapshot.csv`
A processed and anonymized snapshot of the data-asset listing graph used for the main benchmarking pipeline.

## Output files

### Main paper tables
`paper_result_final/paper_results_main_appendix/All_Tables_1_5_and_Appendix_B_1_4.xlsx`

Contains the consolidated outputs for:

- Tables 1–5
- Appendix B1–B4

### Appendix A diagnostics
`paper_result_final/appendix_residual_diagnostics/`

Contains:

- pooled residual files for KFold and Group KFold,
- residual histogram and Q–Q plot figures,
- machine-readable normality summaries,
- a paper-facing Appendix Table A1,
- an Excel workbook collecting the appendix residual-diagnostic results.

## How to run

### Environment
Recommended Python version: **3.10+**

Main dependencies include:

- numpy
- pandas
- matplotlib
- scipy
- scikit-learn
- networkx
- gensim
- openpyxl

Install dependencies with your preferred environment manager (for example, `pip`, `conda`, or `venv`).

### Run the main paper pipeline
```bash
python run_paper_all_from_raw_standalone_with_excel.py
```

This script reproduces the main descriptive outputs, the benchmark-construction results, and the consolidated Excel workbook for Tables 1–5 and Appendix B1–B4.

### Run Appendix A residual diagnostics
```bash
python appendix_residual_diagnostics.py
```

By default, this command rebuilds Appendix Table A1 and the appendix diagnostic workbook from cached diagnostic outputs that are already included in the repository.

### Optional full recomputation of Appendix A
```bash
python appendix_residual_diagnostics.py --recompute
```

This optional mode re-estimates the global prior model from the anonymized graph snapshot, regenerates pooled out-of-fold residuals, and rewrites the Appendix A figures and summary files.

## Reproducibility scope

This repository is designed as a **reproducibility package**. It includes the key scripts, anonymized processed inputs, selected final figures, and the final spreadsheet outputs needed to reproduce the paper’s reported empirical results.

It does **not** include the full raw scraping pipeline, original marketplace identifiers, or unprocessed source records.

## Why some data are anonymized or encrypted

The original source materials are derived from platform-disclosed marketplace records and may still involve supplier-related identifiers, descriptive traces, and commercially sensitive information that could facilitate re-identification or reverse mapping to original entities.

For this reason, the repository releases only a **reproducibility dataset** based on processed and anonymized files. Certain source materials are anonymized, partially withheld, or encrypted for the following reasons:

1. **Data protection and privacy risk**  
   Even when direct identifiers are removed, raw marketplace records may still contain enough structured and textual detail to enable re-identification when combined with external information.

2. **Data-security and compliance considerations**  
   Public redistribution of raw platform records on a global code-hosting platform may create compliance risks under applicable data-security, cybersecurity, and personal-information rules, especially where cross-border public dissemination is involved.

3. **Commercial sensitivity and platform governance**  
   The raw source records may contain supplier-level and product-level details that remain commercially sensitive, even if they were originally observable on platform pages.

4. **Reproducibility–compliance balance**  
   The goal of this repository is to support paper-level reproducibility without exposing unnecessary source details. The released files are therefore sufficient to reproduce the reported tables, figures, and benchmarking logic, while reducing legal, ethical, and commercial risk.

## Notes

- This repository is intended for academic reproducibility and research transparency.
- It should not be interpreted as releasing complete original platform data.
- Users should conduct their own compliance review before redistributing derivative datasets or raw records.
- This repository does not provide legal advice.

## Citation

If you use this repository, please cite the associated paper.

## License

Please add the license you intend to use for the code and data package.
