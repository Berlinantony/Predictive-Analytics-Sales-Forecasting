import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

# ==========================
# Load Dataset
# ==========================

df = pd.read_csv(
    r"C:\Users\berli\Downloads\archive\train.csv"
)

# ==========================
# Convert Date Column
# ==========================

df['Order Date'] = pd.to_datetime(
    df['Order Date'],
    dayfirst=True
)

# ==========================
# Monthly Sales Aggregation
# ==========================

monthly_sales = df.groupby(
    pd.Grouper(key='Order Date', freq='ME')
)['Sales'].sum().reset_index()

# ==========================
# Create Feature
# ==========================

monthly_sales['Month_Number'] = range(len(monthly_sales))

# ==========================
# Define Features and Target
# ==========================

X = monthly_sales[['Month_Number']]
y = monthly_sales['Sales']

# ==========================
# Train Model
# ==========================

model = LinearRegression()

model.fit(X, y)

# ==========================
# Predictions
# ==========================

predictions = model.predict(X)

# ==========================
# Model Evaluation
# ==========================

mae = mean_absolute_error(y, predictions)
r2 = r2_score(y, predictions)

print("=" * 50)
print("MODEL EVALUATION")
print("=" * 50)

print(f"MAE      : {mae:.2f}")
print(f"R2 Score : {r2:.4f}")

# ==========================
# Future Forecast
# ==========================

future_months = pd.DataFrame({
    'Month_Number': range(
        len(monthly_sales),
        len(monthly_sales) + 12
    )
})

future_predictions = model.predict(future_months)

print("\n" + "=" * 50)
print("FUTURE SALES FORECAST")
print("=" * 50)

for i, value in enumerate(future_predictions, start=1):
    print(f"Month {i}: {value:.2f}")

# ==========================
# Future Dates
# ==========================

future_dates = pd.date_range(
    start=monthly_sales['Order Date'].max()
          + pd.offsets.MonthEnd(1),
    periods=12,
    freq='ME'
)

# ==========================
# Visualization
# ==========================

plt.figure(figsize=(14, 7))

# Actual Sales
plt.plot(
    monthly_sales['Order Date'],
    y,
    marker='o',
    label='Actual Sales'
)

# Predicted Sales
plt.plot(
    monthly_sales['Order Date'],
    predictions,
    marker='o',
    label='Predicted Sales'
)

# Future Forecast
plt.plot(
    future_dates,
    future_predictions,
    '--',
    marker='o',
    label='Future Forecast'
)

plt.title(
    'Predictive Analytics Using Historical Sales Data',
    fontsize=14
)

plt.xlabel('Date')
plt.ylabel('Sales')

plt.legend()
plt.grid(True)

plt.tight_layout()

plt.show()