# Senegal Yield - Climate Extremes

Machine learning framework based on Random Forest to assess the impact of climate extremes on crop yield anomalies in Senegal using CMIP6 projections and SHAP analysis.

## Overview
## Overview

This repository contains the scripts used to assess the relationships between climate extremes and crop yield anomalies in Senegal using a Random Forest framework.

The workflow includes:

1. Yield data preprocessing and LOESS-based anomaly computation
2. Crop-specific MIRCA2000 mask construction
3. Extraction of observational and CMIP6 climate data
4. Computation of climate-extreme and moisture indices
5. Construction of crop-specific department-year datasets
6. Random Forest modelling
7. Leave-One-Year-Out (LOYO) temporal validation
8. SHAP-based interpretation
9. Out-of-domain (OOD) analysis of future predictors
10. Sensitivity analyses for PET formulation, predictor dependence, missing yield records, and CMIP6 ensemble robustness
## Final Random Forest predictors

The final Random Forest models use seven climatic predictors:

- TXx
- TNx
- RX5day
- R95pTOT
- CDD
- PRtot
- SPEI3

SPEI3 is calculated using Hargreaves PET in the reference analysis. A Makkink-based SPEI3 is used as a sensitivity experiment.

## Study area

Senegal (24 departments across Sahelian, central, and southern agro-ecological zones)

## Model

- Random Forest: captures non-linear relationships and interactions between climate variables
- SHAP: used to interpret model outputs and quantify the contribution of climatic drivers

## Climate scenarios

- Historical: 1986–2013
- Near future: 2036–2065
- Far future: 2071–2100
- Scenarios: SSP245 and SSP585
- ## Data availability

The high-resolution bias-corrected CMIP6 dataset used in this study is publicly available at:
https://doi.org/10.14768/5c360e73-3337-48e6-8eaa-2295203d86c3

MIRCA2000 crop-distribution data are available at:
https://doi.org/10.5281/zenodo.7422506

Departmental crop-yield data were obtained from DAPSA, Senegal:
https://www.dapsa.gouv.sn/

## Requirements

- Python (pandas, numpy, xarray, sklearn, shap)
- CDO (Climate Data Operators)
- Bash environment

## Note

This repository contains the final version of the analysis based exclusively on a Random Forest model.

LASSO regression was tested during preliminary analysis but is not included in the final framework.

## Author

Dr Alioune Badara Sarr

## Code availability

All scripts are available in this repository.
