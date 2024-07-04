# ML-Binary-Classification-with-a-Bank-Churn-Dataset
### GitHub README

```markdown
# Predictive Modeling for Bank Customer Churn

## Overview

This project focuses on predicting bank customer churn using machine learning techniques. The process includes data preprocessing, exploratory data analysis, feature engineering, and the application of sophisticated machine learning models.

## Table of Contents
- [Project Structure](#project-structure)
- [Data Preprocessing](#data-preprocessing)
- [Exploratory Data Analysis](#exploratory-data-analysis)
- [Feature Engineering](#feature-engineering)
- [Model Building](#model-building)
- [Results](#results)
- [Dependencies](#dependencies)
- [Usage](#usage)
- [License](#license)

## Project Structure

```
BankChurnPrediction/
│
├── data/
│   ├── train.csv
│   └── test.csv
│
├── notebooks/
│   ├── EDA.ipynb
│   ├── Preprocessing.ipynb
│   ├── FeatureEngineering.ipynb
│   └── Modeling.ipynb
│
├── src/
│   ├── preprocessing.py
│   ├── eda.py
│   ├── feature_engineering.py
│   └── modeling.py
│
└── README.md
```

## Data Preprocessing

Data preprocessing involved several key steps:

1. **Reading Data:**
   - Loaded training and test datasets.
   - Initial data inspection using `head`, `info`, and `describe`.

2. **Column Dropping:**
   - Removed non-essential columns: `CustomerId`, `id`, `Surname`.

   ```python
   def drop_col(data):
       data = data.drop(columns=["CustomerId", "id", "Surname"], axis=1)
       return data
   ```

3. **Label Encoding:**
   - Encoded categorical variables: `Geography` and `Gender`.

   ```python
   from sklearn.preprocessing import LabelEncoder

   def Encoding_data(data):
       la = LabelEncoder()
       col = ["Geography", "Gender"]
       for i in col:
           data[i] = la.fit_transform(data[i])
       return data
   ```

4. **Feature Scaling:**
   - Applied Min-Max Scaling to continuous features.

   ```python
   from sklearn.preprocessing import MinMaxScaler

   minimax = MinMaxScaler(feature_range=(0, 1))
   data_train[["CreditScore", "Balance", "EstimatedSalary", "Age"]] = minimax.fit_transform(data_train[["CreditScore", "Balance", "EstimatedSalary", "Age"]])
   data_test[["CreditScore", "Balance", "EstimatedSalary", "Age"]] = minimax.transform(data_test[["CreditScore", "Balance", "EstimatedSalary", "Age"]])
   ```

## Exploratory Data Analysis

Exploratory Data Analysis (EDA) included:

- **Histograms:** Visualized feature distributions.
- **Correlation Heatmap:** Analyzed relationships between variables.

```python
import matplotlib.pyplot as plt
import seaborn as sns

data_train.hist(figsize=(15, 8))
plt.figure(figsize=(20, 20))
sns.heatmap(data_train.corr(), cmap="Blues", annot=True, fmt="0.1f", square=True)
```

## Feature Engineering

To handle class imbalance in the target variable (`Exited`):

- **Over-Sampling with SMOTE:** Used SMOTE to balance the dataset.

```python
from imblearn.over_sampling import SMOTE

X = data_train.drop(columns="Exited", axis=1)
Y = data_train["Exited"]

smote = SMOTE()
new_x, new_y = smote.fit_resample(X, Y)
data_train = pd.concat([new_x, new_y], axis=1)
```

## Model Building

Developed models to predict customer churn:

1. **Random Forest Classifier:**
   - Built a Random Forest model.

   ```python
   from sklearn.ensemble import RandomForestClassifier

   model_RF = RandomForestClassifier()
   model_RF.fit(x_train, y_train)
   ```

2. **AdaBoost Classifier:**
   - Implemented AdaBoost with a Decision Tree base estimator.

   ```python
   from sklearn.ensemble import AdaBoostClassifier
   from sklearn.tree import DecisionTreeClassifier

   model_AD = AdaBoostClassifier(estimator=DecisionTreeClassifier(max_depth=1000, 
                                                                  min_samples_split=5,
                                                                  min_samples_leaf=3),
                                 n_estimators=100,
                                 learning_rate=0.01)

   model_AD.fit(x_train, y_train)
   ```

## Results

- **Random Forest:** Achieved strong performance in predicting customer churn.
- **AdaBoost:** Enhanced prediction accuracy with an ensemble approach.

## Dependencies

- **Python 3.x**
- **Pandas**
- **NumPy**
- **Matplotlib**
- **Seaborn**
- **Scikit-learn**
- **Imbalanced-learn**
- **XGBoost**

**Installation:**

```bash
pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn xgboost
```

## Usage

1. Clone the repository:

    ```bash
    git clone https://github.com/your-username/BankChurnPrediction.git
    ```

2. Navigate to the project directory:

    ```bash
    cd BankChurnPrediction
    ```

3. Explore the Jupyter notebooks in the `notebooks/` directory for detailed steps on data preprocessing, EDA, feature engineering, and model building.

4. Run the scripts in the `src/` directory to preprocess data and build models.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

Feel free to contribute to the project by opening issues or submitting pull requests!
```

---
