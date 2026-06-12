# Senegal Yield - Climate Extremes

Machine learning framework based on Random Forest to assess the impact of climate extremes on crop yield anomalies in Senegal using CMIP6 projections and SHAP analysis.

## Overview

This repository contains scripts to analyze the impact of climate extremes on crop yield anomalies in Senegal using a machine learning approach.

The workflow includes:

1. Yield data processing and anomaly computation (LOESS)
2. Crop mask construction using MIRCA data
3. Climate data extraction (ARC2 merged with ANACIM station observations, CHIRTS and CMIP6)
4. Computation of climate extreme indices (TXx, TNx, RX1day,RX5day, CDD, R95pTOT, PRtotal, SPEI3)
5. Dataset construction for machine learning
6. Random Forest modeling
7. Model evaluation using Leave-One-Year-Out (LOYO)
8. SHAP analysis for model interpretability

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
