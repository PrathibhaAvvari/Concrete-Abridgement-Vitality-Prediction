# Concrete Abridgement Vitality Prediction

Predicting the **compressive strength of concrete** (MPa) from its mix ingredients and age, using a hand-built Keras neural network and an AutoML model built with AutoKeras.

---

## Overview

Concrete is the most important material in civil engineering, and its compressive strength is a highly nonlinear function of its age and ingredients. Testing strength in a lab takes days or weeks. This project trains regression models that estimate strength directly from the mix design, so engineers can screen mixes before pouring and curing test samples.

The notebook covers the full workflow:

1. Data analysis
2. Data preprocessing (outlier removal)
3. Feature engineering (feature selection)
4. Model building with a Keras Sequential network
5. Model building with AutoKeras (AutoML)
6. An additional experiment: predicting concrete **age** from the mix

---

## Dataset

`concrete_data.csv` contains **1,030 samples** and **9 numeric columns**, with no missing values. Each row is a concrete mixture whose compressive strength was measured in a laboratory at a specific age.

| Column | Unit | Range |
|---|---|---|
| `cement` | kg/m³ | 102.0 – 540.0 |
| `blast_furnace_slag` | kg/m³ | 0.0 – 359.4 |
| `fly_ash` | kg/m³ | 0.0 – 200.1 |
| `water` | kg/m³ | 121.8 – 247.0 |
| `superplasticizer` | kg/m³ | 0.0 – 32.2 |
| `coarse_aggregate` | kg/m³ | 801.0 – 1145.0 |
| `fine_aggregate ` | kg/m³ | 594.0 – 992.6 |
| `age` | days | 1 – 365 |
| `concrete_compressive_strength` | MPa | 2.33 – 82.6 (**target**) |

> Note: the `fine_aggregate ` column name has a trailing space in the CSV.

This is the well-known *Concrete Compressive Strength* dataset (I-Cheng Yeh), available from the UCI Machine Learning Repository and on Kaggle.

---

## Project Structure

```
.
├── ConcreteAbridgementVitalityPrediction.ipynb   # Main notebook
├── concrete_data.csv                             # Dataset (add this yourself)
├── structured_data_regressor/                    # Created by AutoKeras during training
└── README.md
```

---

## Methodology

### 1. Data Analysis
The notebook inspects the data with `describe()`, `info()` and `isna()`, plots a correlation heatmap, draws pairwise scatter plots coloured by strength, and uses box plots to look for outliers in every feature.

### 2. Preprocessing: Outlier Removal
Rows outside these bounds are dropped:

- `blast_furnace_slag < 350`
- `122 < water < 246`
- `superplasticizer < 25`
- `age < 150`

After filtering, about **952 rows** remain.

### 3. Feature Engineering
Based on the correlation analysis, the model uses five input features:

`cement`, `fly_ash`, `water`, `superplasticizer`, `age`

The columns `blast_furnace_slag`, `coarse_aggregate` and `fine_aggregate` are dropped.

### 4. Train/Test Split
The data is split 70/30 (`test_size=0.3`, `random_state=42`), which gives 666 training rows and 286 test rows.

### 5. Model A: Keras Sequential Network

```
Dropout(0.1)
Dense(100, activation='relu')
Dropout(0.7)
Dense(5, activation='tanh')
Dropout(0.2)
Dense(1)                       # regression output
```

The network is compiled with the `rmsprop` optimizer, `mse` loss and `mae` metric, and trained for 100 epochs with `batch_size=1`.

### 6. Model B: AutoKeras StructuredDataRegressor
[AutoKeras](https://autokeras.com/), an AutoML system from the DATA Lab at Texas A&M University, searches for a good architecture automatically (`max_trials=3`). The best model is exported with `reg.export_model()`, and it includes built-in normalization and categorical-encoding layers.

### 7. Extra Experiment: Age Prediction
A second AutoKeras regressor tries to predict `age` from the four remaining mix features (strength and age are excluded from the inputs).

---

## Results

| Model | Target | Test MSE | Test MAE |
|---|---|---|---|
| Keras Sequential | Compressive strength | 276.60 | **13.15 MPa** |
| AutoKeras | Compressive strength | *not printed in notebook* | — |
| AutoKeras | Age | 834.67* | — |

\* The notebook prints this value as "MAE", but `reg.evaluate()` returns `[loss, mae]`, and the loss is MSE. So `834.67` is actually the **MSE** (an RMSE of about 28.9 days). See *Known Issues* below.

The age model performs poorly. This is expected, because age cannot be reliably inferred from mix proportions alone: the same mix is tested at many different ages.

---

## Getting Started

### Prerequisites
- Python 3.10
- Jupyter Notebook or JupyterLab

### Installation

```bash
git clone <your-repo-url>
cd <repo-folder>

pip install pandas numpy matplotlib seaborn scikit-learn tensorflow==2.13.0 autokeras==1.1.0
```

> The notebook also installs `keras-tuner` from GitHub (`1.0.2rc1`). Recent versions of `autokeras` pull in a compatible `keras-tuner` automatically, so this step is usually unnecessary.

### Running

1. Download `concrete_data.csv` and place it somewhere accessible.
2. Launch Jupyter:
   ```bash
   jupyter notebook ConcreteAbridgementVitalityPrediction.ipynb
   ```
3. Run the cells in order. When prompted, enter the path to the CSV file:
   ```
   Enter the path name: ./concrete_data.csv
   ```

---

## Tech Stack

- **Data handling:** pandas, NumPy
- **Visualization:** Matplotlib, Seaborn
- **Machine learning:** scikit-learn (train/test split)
- **Deep learning:** TensorFlow / Keras
- **AutoML:** AutoKeras, Keras Tuner

---

## Known Issues & Future Improvements

- **The feature-selection code is missing.** The cell that drops `blast_furnace_slag`, `coarse_aggregate` and `fine_aggregate` appears to have been cleared, although its effect is visible in the outputs. Re-add it for reproducibility:
  ```python
  df = df.drop(["blast_furnace_slag", "coarse_aggregate", "fine_aggregate "], axis=1)
  ```
- **Metric labelling:** `mae, _ = reg.evaluate(...)` actually unpacks the loss (MSE). Use `_, mae = reg.evaluate(...)` instead.
- **No feature scaling** for the Keras Sequential model. Adding `StandardScaler` or a `Normalization` layer should improve convergence and accuracy.
- **Heavy dropout (0.7)** and `batch_size=1` make training slow and noisy. Tuning these, or adding early stopping, is likely to help.
- **Interactive `input()`** for the file path blocks automated runs. Consider a config variable or a command-line argument.
- The nested scatter-plot loop generates 81 figures. A `sns.pairplot` would be more compact.
- Add additional metrics (R², RMSE) and a predicted-vs-actual plot for easier comparison between models.
- Compare against classical baselines (Linear Regression, Random Forest, XGBoost), which perform very well on this dataset.

---

## Acknowledgements

- Dataset: I-Cheng Yeh, *Concrete Compressive Strength*, UCI Machine Learning Repository.
- [AutoKeras](https://autokeras.com/), DATA Lab, Texas A&M University.

