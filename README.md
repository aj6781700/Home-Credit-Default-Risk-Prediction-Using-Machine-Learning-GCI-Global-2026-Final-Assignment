# Home Credit Default Risk Prediction

A machine learning solution developed for the **GCI Global 2026 In-class Competition** to predict the probability of loan repayment difficulties (`TARGET`) using customer financial and demographic data.

## Overview
The goal of this project is to estimate the probability of default for each applicant in the test set. Predictions are evaluated using the **ROC-AUC** (Area Under the Receiver Operating Characteristic Curve) metric.

## Methodology
* **Data Preprocessing:** Aligned training and test schemas, handled categorical missing values with explicit labels, and preserved numeric missing values for native gradient boosting handling.
* **Feature Engineering:** Created domain-specific financial ratios and time-based features, including:
  * Age and employment duration (in years)
  * Credit-to-income, annuity-to-income, and credit-to-annuity ratios
  * Goods-price-to-credit ratio and children-per-family count
* **Modeling & Validation:** Trained a `CatBoostClassifier` using a stratified 80/20 train-validation split (`Logloss` objective, `AUC` evaluation metric) with early stopping and a fixed random seed for reproducibility.
* **Final Submission:** Retrained the model on the full training set using the optimal iteration count to generate probability predictions in the required `SK_ID_CURR, TARGET` submission format.

## Tech Stack
* **Language:** Python
* **Libraries:** `pandas`, `numpy`, `scikit-learn`, `catboost`

## How to Run
1. Place `train.csv`, `test.csv`, and `sample_submission.csv` in the data directory.
2. Install required dependencies:
   ```bash
   pip install pandas numpy scikit-learn catboost
   ```
3. Run the Jupyter/Colab notebook end-to-end to train the model and generate `submission.csv`.

## Acknowledgments & References
* Developed independently using prior machine learning knowledge, **GCI Global 2026** course materials, and publicly available baseline notebooks/documentation on Kaggle and Google (without the use of Generative AI).
* [Kaggle – Home Credit Default Risk Competition](https://kaggle.com)
