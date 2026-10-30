# Bina.az Real Estate Classification

A machine learning project built on real estate listings from [Bina.az](https://bina.az), Azerbaijan's largest property marketplace.

## Dataset

[Kaggle — Bina.az sale project](https://www.kaggle.com/datasets/sehriyarmemmedli/binaaz-sale-project1) (`house_sale.csv`). The notebook downloads it automatically via the Kaggle API.

## Workflow

1. **Data cleaning** — handling missing values, parsing listing fields, feature engineering.
2. **Preprocessing** — scikit-learn `Pipeline` + `ColumnTransformer`.
3. **Modeling** (evaluated with F1 score)
   - Base model
   - Decision Tree Classifier (with class weights for imbalanced data)
   - Bagging (500 decision trees)
   - **Boosting (AdaBoost) — best model**
4. Hyperparameter tuning with randomized search.

## How to run

Open `Bina_az_sale_project.ipynb` in Google Colab, upload your `kaggle.json` when prompted, and run the cells in order.

```bash
pip install pandas numpy scikit-learn matplotlib seaborn kaggle
```

## Technologies

Python, pandas, NumPy, scikit-learn, Matplotlib, Seaborn
