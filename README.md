# Gold Price Prediction using Machine Learning (Python)
This repository hosts a machine learning project, developed as part of a group, aimed at predicting historical gold prices. The project utilizes various economic indicators and implements a comprehensive data analysis pipeline, from preprocessing and exploratory analysis to outlier detection, model application, and rigorous evaluation, to identify the most effective predictive model.

# 📝 Project Overview
This project explores the application of machine learning techniques to a real-world financial forecasting problem. By analyzing the relationships between gold prices and other key economic indicators, it demonstrates a systematic approach to building and validating predictive models for time-series data.

# 🎯 Objectives
To import and conduct initial analysis of historical gold prices and related economic indicators.

To perform robust data preprocessing, including handling missing values and ensuring correct data types.

To execute comprehensive Exploratory Data Analysis (EDA) to understand data distributions, correlations, and relationships.

To identify and handle outliers to improve model robustness.

To apply and evaluate various machine learning regression models for gold price prediction.

To determine the most effective model based on performance metrics and visual assessment.

# 📊 Dataset
The dataset (gld_price_data.csv) contains historical daily information on gold prices and several relevant economic indicators from 2008 onwards:

Date: The date in MM/dd/yyyy format.

SPX: The Standard and Poor's 500 index (S&P 500).

GLD: Gold price (the target variable).

USO: The United States Oil Fund ® LP (USO), an exchange-traded security tracking crude oil prices.

SLV: Silver price.

EUR/USD: Euro to US dollar exchange ratio.

# ⚙️ Methodology & Workflow
The project follows a structured machine learning workflow implemented entirely in Python:

Data Import and Initial Analysis:

Imported the gld_price_data.csv dataset using pandas.

Performed initial data inspection with .head(), .info(), and .describe() to understand data types, check for null values, and get summary statistics.

# Data Preprocessing:

Handled and verified the absence of missing values using .isna().sum().

Ensured all columns were of appropriate data types; specifically, the 'Date' column was converted to datetime format for time-series operations.

Exploratory Data Analysis (EDA):

Calculated and visualized a correlation matrix using seaborn.heatmap to identify relationships between gold prices and other indicators.

Utilized pair plots and scatter plots to visually explore data distributions and inter-variable relationships.

Leveraged Sweetviz for automated EDA to gain quick, insightful visualizations and summaries.

# Outlier Detection:

Applied a percentile-based method to detect and manage outliers within the dataset.

Visualized data distributions and outlier presence using KDE (Kernel Density Estimate) charts, highlighting outliers for clear identification.

# Feature Selection and Data Partition:

Selected relevant economic indicators (SPX, USO, SLV, EUR/USD) as features and 'GLD' (Gold price) as the target variable.

Split the dataset into training (80%) and testing (20%) sets using sklearn.model_selection.train_test_split to prepare for model training and unbiased evaluation.

# Model Application:

Implemented and trained four distinct machine learning regression models:

Random Forest Regressor: Chosen for its ensemble learning capabilities, robustness against overfitting, and strong predictive power.

Decision Tree Regressor: A simpler, interpretable model used to understand feature importance and decision rules.

K-Nearest Neighbors (KNN) Regressor: A non-parametric method suitable for capturing local data patterns.

Linear Regression: A foundational model used as a baseline for performance comparison.

# Model Evaluation:

Evaluated each model's performance using a suite of metrics:

Mean Absolute Error (MAE)

Mean Squared Error (MSE)

Root Mean Square Error (RMSE)

R-squared Score (R²)

Explained Variance Score

Visualized actual vs. predicted gold prices for each model to provide a qualitative assessment of predictive accuracy.

# 📈 Key Findings & Solution
The Random Forest Regressor emerged as the most effective model for predicting gold prices in this dataset. It consistently demonstrated superior prediction accuracy and lower error rates (e.g., highest R-squared, lowest MAE) compared to the other models tested. This detailed comparative analysis provided valuable insights into the strengths and limitations of different machine learning approaches for financial time-series prediction.

# 🛠️ Technologies & Tools
Python: Primary programming language.

Pandas: For data manipulation and analysis.

NumPy: For numerical operations.

Matplotlib: For creating static, animated, and interactive visualizations.

Seaborn: For drawing attractive and informative statistical graphics.

Scikit-learn: For machine learning model implementation (train-test split, RandomForestRegressor, DecisionTreeRegressor, KNeighborsRegressor, LinearRegression, metrics).

Sweetviz: For automated Exploratory Data Analysis (EDA) and visualization.

# 🏃 How to Run the Project
Clone the repository:

git clone https://github.com/Sametcan99/Gold-Price-Prediction-ML-Python.git

Navigate to the project directory:

cd Gold-Price-Prediction-ML-Python

Ensure you have the dataset: Place the gld_price_data.csv file in the same directory as the Jupyter Notebook.

Install necessary libraries:

pip install numpy pandas matplotlib seaborn scikit-learn sweetviz

Open and run the Jupyter Notebook:

jupyter notebook gold-price-prediction.ipynb

Execute the cells sequentially to replicate the analysis, model training, and evaluation.

# 📞 Contact
Feel free to connect with me for any questions or collaborations:

LinkedIn: https://www.linkedin.com/in/sametcan-kandemirt/

Email: kandemirsametcan99@gmail.com
