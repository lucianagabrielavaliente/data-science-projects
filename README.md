# Data Science Projects
This repository contains a collection of data science projects using Python. Each project is designed to explore and apply data analysis, machine learning, and visualization techniques. 

----

# Health Insurance Cost Prediction Project

This repository contains scripts and data for predicting health insurance costs using machine learning models. The dataset used (`insurance.csv`) includes information about individuals, such as age, sex, BMI, number of children, smoking habit, and region.

## Repository Structure

- **insurance.csv**: Dataset containing health insurance data.
- **data-analysis.ipynb**: Jupyter Notebook for data exploration and analysis.
- **machine-learning.ipynb**: Jupyter Notebook for machine learning model training and evaluation.

## Data Analysis

The data analysis notebook (`data-analysis.ipynb`) includes various exploratory data analysis (EDA) tasks:

- Loading and initial inspection of the dataset.
- Checking for missing values (`NaN`).
- Removing duplicates from the dataset.
- Exploring categorical variables like sex, smoker status, and region.
- Calculating correlations with the insurance charges variable.
- Visualizing data distributions and relationships using seaborn plots.

### Key Insights

- The dataset includes data on **1,338 individuals**.
- Features such as **smoker status** have a strong positive correlation with insurance charges (`0.787`), indicating higher charges for smokers.
- Visualization includes histograms and scatterplots showing relationships between features and insurance charges.

## Machine Learning

The machine learning notebook (`machine-learning.ipynb`) focuses on training and evaluating predictive models:

- Preprocessing data by encoding categorical variables using `LabelEncoder` and scaling numerical features with `StandardScaler`.
- Splitting the dataset into training and testing sets.
- Training three different models:
  - **Linear Regression**: Predicting insurance costs based on scaled features.
  - **Support Vector Regressor (SVR)**: Hyperparameter tuning using GridSearchCV.
  - **Random Forest Regressor**: Hyperparameter tuning using GridSearchCV.
- Evaluating model performance using Mean Absolute Error (MAE) and Root Mean Squared Error (RMSE).

### Results

- **Linear Regression**:
  - MAE: 4317.67
  - RMSE: 6195.90

- **SVR**: 
  - MAE: 8146.55
  - RMSE: 12523.41

- **Random Forest Regressor** (Best Parameters selected):
  - MAE: 2689.36
  - RMSE: 4699.82

### Conclusion

The Random Forest Regressor outperformed other models with lower MAE and RMSE, suggesting it is the most suitable model for predicting insurance costs based on the provided features.

## Usage

1. Ensure Python 3.x is installed along with necessary libraries (`pandas`, `scikit-learn`, `numpy`).
2. Open and run the Jupyter Notebooks (`data-analysis.ipynb` and `machine-learning.ipynb`) to explore data and train models.
3. Adjust parameters or add new features as needed for further analysis or model improvement.
