<div align="center">

# 🏠 Bina.az Title Deed Classification

**Predicting whether a property listing has an official title deed (`Çıxarış`)**

[![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?logo=scikitlearn&logoColor=white)](https://scikit-learn.org/)
[![pandas](https://img.shields.io/badge/pandas-150458?logo=pandas&logoColor=white)](https://pandas.pydata.org/)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mahammadhajiiyev/bina-az-price-prediction/blob/main/Bina_az_sale_project.ipynb)

</div>

---

## 📌 Overview

[Bina.az](https://bina.az) is Azerbaijan's largest real estate marketplace. Whether a property comes with an official **title deed extract (`Çıxarış`)** strongly affects its legal status, mortgage eligibility and price.

This project builds a **binary classifier** that predicts from listing details alone whether a property **has (`var`)** or **does not have (`yoxdur`)** a title deed. The final **AdaBoost** model reaches a **weighted F1 score of 0.918** on the test set and **0.935 in cross-validation**.

## 📊 Dataset

| | |
|---|---|
| **Source** | [Kaggle — Bina.az sale project](https://www.kaggle.com/datasets/sehriyarmemmedli/binaaz-sale-project1) |
| **Size** | 100,775 listings × 51 columns |
| **Target** | `Çıxarış` — title deed: `var` (yes) / `yoxdur` (no) |
| **Class balance** | 79% `var` · 21% `yoxdur` — **imbalanced** |

Key features: price, price per m², area (`Sahə`), land area (`Torpaq sahəsi`), number of rooms (`Otaq sayı`), floor and total floors, property category, location and listing type (VIP, featured, agency).

## 🔧 Workflow

1. **Data cleaning**
   - Removed scraping metadata, URLs, descriptions and coordinates
   - Dropped columns with too many missing values (e.g. `products_label` — 28% missing)
   - Split the `Mərtəbə` field (e.g. `5/9`) into separate `floor` and `total_floor` features
   - Removed `İpoteka` (mortgage) to **prevent data leakage** — it is closely tied to having a title deed
2. **Preprocessing** — scikit-learn `Pipeline` + `ColumnTransformer` (imputation, scaling, one-hot encoding)
3. **Handling imbalance** — `class_weight` in tree models and **F1 score** as the main metric instead of plain accuracy
4. **Modeling & tuning** — four model families compared; the Decision Tree tuned with `RandomizedSearchCV`

## 🏆 Results

| Model | Test accuracy | Weighted F1 | Notes |
|---|:---:|:---:|---|
| Logistic Regression (baseline) | 0.792 | 0.776 | Struggles with the minority class |
| Decision Tree | 0.887 | 0.888 | `max_depth=46`, `min_samples_leaf=3` · CV F1 = 0.908 |
| Bagging (500 trees) | 0.808 | 0.773 | No gain over a single tree |
| **AdaBoost** ✅ | **0.919** | **0.918** | 200 estimators · **CV F1 = 0.935** |

### Better detection of the minority class

| Correctly identified | Baseline (LogReg) | Decision Tree |
|---|:---:|:---:|
| Properties **without** a deed (`yoxdur`) | 1,493 / 4,279 (35%) | **3,271 / 4,279 (76%)** |
| Properties **with** a deed (`var`) | 14,465 / 15,876 (91%) | **14,603 / 15,876 (92%)** |

Tree-based models more than **double the recall on properties without a title deed** — the harder and more valuable case to detect.

## 🚀 How to run

1. Click the **Open in Colab** badge above.
2. Upload your Kaggle API key (`kaggle.json`) when prompted — the dataset downloads automatically.
3. Run all cells.

To run locally:

```bash
pip install pandas numpy scikit-learn matplotlib seaborn kaggle
jupyter notebook Bina_az_sale_project.ipynb
```

## 📁 Project structure

```
bina-az-price-prediction/
├── Bina_az_sale_project.ipynb   # data cleaning, modeling, evaluation
└── README.md
```

## 🔮 Future improvements

- Try gradient boosting libraries (LightGBM, XGBoost, CatBoost)
- Use the listing description text as a feature (TF-IDF or embeddings)
- Resampling techniques (e.g. SMOTE) for the minority class
- Explain predictions with SHAP values

## 👤 Author

**Mahammad Hajiyev** · [@mahammadhajiiyev](https://github.com/mahammadhajiiyev)
