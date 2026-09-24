# Concrete Compressive Strength Prediction

Predicting the compressive strength of concrete (MPa) from its mix design and curing age, using classical machine learning and deep learning.

Lab-testing a concrete mix takes days to weeks. A reliable model lets engineers screen mix designs before any samples are cast.

## Results (held-out test set, 201 mixes)

| Model | MAE (MPa) | RMSE (MPa) | R² |
|---|---|---|---|
| **XGBoost (tuned)** | **2.59** | **4.11** | **0.943** |
| Gradient Boosting | 2.69 | 4.12 | 0.943 |
| Keras DNN | 2.94 | 4.31 | 0.938 |
| Random Forest | 3.52 | 4.95 | 0.918 |
| Linear Regression | 5.68 | 7.38 | 0.817 |
| Mean baseline | 13.97 | 17.30 | 0.000 |

88% of test predictions fall within ±5 MPa of the lab-measured value.

## Approach

1. **EDA:** distributions, correlations, strength vs. age, and Abrams' law (water/cement ratio).
2. **Cleaning:** removed 25 duplicate rows so the same mix can't appear in both the training and test sets. Kept the real extreme values instead of deleting them.
3. **Feature engineering:** water/cement ratio, water/binder ratio, total binder, superplasticizer/binder, aggregate/binder, log(age).
4. **Modelling:** a mean baseline and linear regression, then Random Forest, Gradient Boosting, and XGBoost with 5-fold cross-validation, plus a `RandomizedSearchCV` search over 40 XGBoost settings.
5. **Deep learning:** a Keras MLP (128-64-32) with a `Normalization` layer, dropout, Adam, early stopping, and LR scheduling.
6. **Explainability:** SHAP shows that curing age and water/binder ratio drive most predictions, which matches concrete engineering theory.
7. **Deployment-ready:** the model is saved with `joblib`, and a `predict_strength()` function scores new mixes.

## Dataset

The [UCI Concrete Compressive Strength dataset](https://archive.ics.uci.edu/dataset/165/concrete+compressive+strength) (I-Cheng Yeh, 1998): 1,030 samples, 8 input features, and 1 target.

## Run it

**Google Colab (easiest):** open the notebook in [Colab](https://colab.research.google.com) and click **Runtime → Run all**. The first cell installs any missing packages and downloads the dataset if it isn't there.

**Locally:**

```bash
pip install -r requirements.txt
jupyter notebook Concrete_Strength_Prediction.ipynb
```
