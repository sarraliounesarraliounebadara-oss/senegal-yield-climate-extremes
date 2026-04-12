# senegal-yield-climate-extremes
Machine learning framework (Random Forest and LASSO) to assess the impact of climate extremes on crop yield anomalies in Senegal using CMIP6 projections and SHAP analysis.
# Senegal Yield - Climate Extremes

This repository contains scripts to analyze the impact of climate extremes on crop yield anomalies in Senegal using machine learning models.

## Overview

The workflow includes:

1. Yield data processing and anomaly computation (LOESS)
2. Crop mask construction using MIRCA data
3. Climate data extraction (CHIRPS, ERA5, CMIP6)
4. Computation of climate extreme indices (TXx, TX35, RX1day, CDD, R95pTOT, PRtotal, SPEI3)
5. Dataset construction for machine learning
6. Machine learning models:
   - Random Forest (RF)
   - LASSO regression
7. Model evaluation using Leave-One-Year-Out (LOYO)
8. SHAP analysis for model interpretability

## Study area

Senegal (24 departments across Sahelian, central and southern agro-ecological zones)

## Models

- Random Forest: non-linear relationships and interactions
- LASSO: linear sparse model for comparison
- SHAP: feature attribution and interpretation

## Climate scenarios

- Historical: 2000–2013
- Near future: 2036–2065
- Far future: 2071–2100
- Scenarios: SSP245 and SSP585

## Requirements

- Python (pandas, numpy, xarray, sklearn, shap)
- CDO (Climate Data Operators)
- Bash environment

## Author

Alioune Sarr

## Code availability

All scripts are available in this repository.
