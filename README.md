# ML_Empty_Housing_ES

## Predicting and Interpreting the Percentage of Empty Homes in Spanish Municipalities Using Machine Learning

---

### Project Objective

This project aims to develop a supervised machine learning model to predict the **percentage of empty homes in Spanish municipalities** using public demographic, geographic, and socioeconomic data.

Beyond achieving accurate predictions, a second key goal is **interpretability**: understanding which variables contribute most to housing vacancy, and providing data-driven insights into regional depopulation patterns.

---

### Dataset

The dataset was built by combining multiple public data sources, primarily from the Spanish National Statistics Institute (INE) and Datos.gob.es. After cleaning and merging at the municipal level, the dataset includes:

- Total number of homes, occasional use, median energy consumption.
- Demographics by age and gender.
- Residential mobility indicators.
- Unemployment by sector, gender, and age group.
- Geographic variables: latitude, longitude, area, perimeter, elevation.
- Province and autonomous community (region).

The final dataset contains **2823 municipalities** and **45 variables**, with no missing values.

The target variable is: `%_casas_vacias` (percentage of empty homes).

A sample file is provided in the `src/data_sample/` folder.

---

### Technical Approach

The project is fully implemented in Python using libraries such as `pandas`, `scikit-learn`, `xgboost`, `catboost`, `matplotlib`, `seaborn`, `shap`, and `joblib`.

Key steps of the process:

1. Exploratory Data Analysis (EDA) and multicollinearity assessment.
2. Categorical encoding (One-Hot and Target Encoding).
3. Feature selection manually and using SelectFromModel, RFE, RFECV, and SFS.
4. Model comparison: Ridge, Random Forest, XGBoost, CatBoost.
5. Hyperparameter optimization via GridSearchCV and cross-validation.
6. Model interpretability analysis using coefficients, feature_importance and SHAP.
7. Export of the final model in `.joblib` format.

The most accurate model was Ridge Regression, while the most interpretable model was an optimized CatBoost trained on a reduced set of 12 selected features.

---

### Repository Structure

ML_Empty_Housing_ES/ │ ├── README.md ├── src/ │ ├── data_sample/ # Sample dataset │ ├── img/ # Visualizations and figures │ ├── models/ # Final exported model (.joblib) │ ├── notebooks/ # Exploratory and development notebooks │ ├── results_notebook/ # Final notebook (reproducible version) │ └── utils/ # Custom functions and scripts


---

### Project Presentation

The project includes a PowerPoint presentation prepared for a 7-minute video pitch, which outlines:

- The problem and real-world context.
- Technical methodology and key decisions.
- Model comparison and final results.
- Insights derived from model interpretation.

---

### Author
 
Author: [Your Name Here]  
Year: 2025

