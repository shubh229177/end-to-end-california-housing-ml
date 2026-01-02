#  End-to-End Machine Learning Project: California Housing Price Prediction

This repository contains a **complete end-to-end Machine Learning project** based on **Chapter 2 of *Hands-On Machine Learning with Scikit-Learn, Keras & TensorFlow (2nd Edition)* by Aurélien Géron**.

The goal of this project is to predict **median house values in California districts** using real-world data and industry-standard ML practices.

---

##  Project Highlights

* ✅ End-to-end ML workflow (data → model → evaluation)
* ✅ Stratified train-test split to avoid data leakage
* ✅ Data preprocessing with Pipelines & ColumnTransformer
* ✅ Multiple models: Linear Regression, Decision Tree, Random Forest
* ✅ Cross-validation & hyperparameter tuning using GridSearchCV
* ✅ Final evaluation on a held-out test set
* ✅ All exploratory graphs saved as image files (production-ready)

---

##  Project Structure

```
end-to-end-california-housing-ml/
│
├── housing.py                # Main Python script (run end-to-end)
├── README.md                 # Project documentation
│
├── datasets/
│   └── housing/
│       ├── housing.csv
│       └── housing.tgz
│
├── images/
│   └── housing/
│       ├── housing_histograms.png
│       ├── housing_geo_scatter.png
│       └── housing_price_population.png
```

---

##  Machine Learning Problem

* **Type**: Supervised Learning
* **Task**: Regression
* **Target Variable**: `median_house_value`
* **Performance Metric**: RMSE (Root Mean Squared Error)
* **Learning Mode**: Batch Learning

---

##  Data Visualization

All exploratory data analysis (EDA) graphs are automatically saved to the `images/housing/` directory:

*  Feature histograms
*  Geographic scatter plot (longitude vs latitude)
*  House price visualization weighted by population

These plots help understand data distribution, correlations, and spatial trends.

---

##  Models Used

1. **Linear Regression** – baseline model
2. **Decision Tree Regressor** – captures nonlinear patterns
3. **Random Forest Regressor** – best-performing model

Model performance is evaluated using **10-fold cross-validation**.

---

##  Hyperparameter Tuning

* Technique: **GridSearchCV**
* Tuned parameters:

  * `n_estimators`
  * `max_features`
  * `bootstrap`

The best model is selected based on cross-validated RMSE.

---

##  Final Evaluation

The final model is evaluated **only once** on the test set to estimate real-world performance.

This ensures:

* No data leakage
* Reliable generalization estimate

---

##  How to Run the Project

###  Clone the repository

```bash
git clone https://github.com/your-username/end-to-end-california-housing-ml.git
cd end-to-end-california-housing-ml
```

###  Install dependencies

```bash
pip install numpy pandas matplotlib scikit-learn
```

###  Run the project

```bash
python housing.py
```

All datasets will be downloaded automatically, and graphs will be saved in the `images/` folder.

---

##  Reference

* Aurélien Géron — *Hands-On Machine Learning with Scikit-Learn, Keras & TensorFlow (2nd Edition)*
* Official GitHub Repository: `ageron/handson-ml2`

---

##  Author

**Shubham Thakur**
Machine Learning Enthusiast | Student | Aspiring Data Scientist

---

##  If you like this project

Give it a star  on GitHub — it helps a lot and motivates further learning!
