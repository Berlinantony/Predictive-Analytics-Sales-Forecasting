# Predictive Analytics Using Historical Sales Data

## Project Overview

This project analyzes historical sales data and builds a predictive model to forecast future sales trends using Machine Learning techniques. The project uses Linear Regression to identify patterns in monthly sales and predict future performance.

## Objective

* Analyze historical sales data
* Aggregate sales on a monthly basis
* Build a Linear Regression model
* Forecast future sales for the next 12 months
* Visualize actual, predicted, and future sales trends

## Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Scikit-learn

## Dataset

The dataset contains historical sales records, including order dates and sales values. Monthly sales were calculated by grouping sales data based on order dates.

## Project Workflow

1. Load and preprocess the dataset
2. Convert order dates into datetime format
3. Aggregate sales data by month
4. Create numerical features for model training
5. Train a Linear Regression model
6. Evaluate model performance using MAE and R² Score
7. Forecast sales for the next 12 months
8. Visualize results using line charts

## Model Evaluation

The model performance was evaluated using:

* Mean Absolute Error (MAE)
* R² Score

## Future Forecast

The trained model predicts sales trends for the next 12 months based on historical patterns.

## Project Structure

Predictive-Analytics-Sales-Forecasting/

├── train.csv

├── predictive_analytics.py

├── README.md

├── requirements.txt

├── screenshots/

└── report.pdf

## How to Run

1. Install required libraries:
   pip install pandas numpy matplotlib scikit-learn

2. Update the dataset path in the Python file.

3. Run:
   python predictive_analytics.py

## Output

* Model evaluation metrics
* Future sales forecast
* Sales trend visualization chart

## Author

Berlin M

B.Tech Computer Science Engineering specialization with (AI & ML)
