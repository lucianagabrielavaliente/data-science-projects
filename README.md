# Data Science Projects
This repository contains a collection of data science projects using Python. Each project is designed to explore and apply data analysis, machine learning, and visualization techniques. 

----

# Health Insurance Cost Prediction Project

This directory contains scripts and data for predicting health insurance costs using machine learning models. The dataset used (`insurance.csv`) includes information about individuals, such as age, sex, BMI, number of children, smoking habit, and region.

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

----

# Booking Cancelation Classification Project

This directory contains scripts and notebooks for analyzing booking data (`booking.csv`) and implementing machine learning models for classification tasks.

## Repository Structure

- **booking.csv**: Raw data file containing booking information.
- **data-analysis.ipynb**: Jupyter Notebook for data exploration and analysis.
- **classification.ipynb**: Jupyter Notebook for training and evaluating classification models.

## Data Exploration and Preprocessing:
   - Load and inspect the dataset.
   - Handle missing values and duplicates (none found in this dataset).
   - Visualize data distributions and correlations.

## Data Analysis
   - Visualize distributions of categorical variables (`no_of_children`, `type_of_meal_plan`, `room_type_reserved`, etc.).
   - Explore correlations between features and the target variable (`booking_status`).

## Machine Learning

The machine learning notebook (`classification.ipynb`) focuses on training and evaluating predictive models:

- Preprocessing data by encoding categorical variables using `LabelEncoder` and scaling numerical features with `StandardScaler`.
- Splitting the dataset into training and testing sets.
- Training three different models:
  - **Logistic Regression**: A linear model for binary classification that estimates the probability of a binary outcome. It is simple and fast but may not capture complex relationships in the data.
  - **K Nearest Neighbors (KNN)**: A non-parametric algorithm that classifies data points based on the majority class among its k-nearest neighbors. It is simple and effective but can be computationally expensive with large datasets.
  - **Random Forest**: An ensemble learning method that builds multiple decision trees and merges them to get a more accurate and stable prediction. It is robust to overfitting and handles non-linear relationships well.
- Evaluating model performance using accuracy score.

### Results

- **Logistic Regression**:
  - Accuracy Score: 0.6636

- **K Nearest Neighbors (KNN)**:
  - Accuracy Score: 0.7175

- **Random Forest**:
  - Accuracy Score: 0.6913

## Prediction for New Data
   - Use of the best performing model (KNN) to predict if a new customer will cancel their booking.

## Conclusion

Based on the accuracy scores, the K Nearest Neighbors (KNN) model performs best for predicting booking status in this dataset.

## Usage

To replicate the analysis and models:

1. Ensure you have Python and necessary libraries installed (`pandas`, `numpy`, `scikit-learn`, `matplotlib`, `seaborn`).
2. Open and run `data-analysis.ipynb` for data exploration and preprocessing.
3. Open and run `classification.ipynb` for training and evaluating machine learning models.