import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, accuracy_score, f1_score

company = 'HON'
start_date = '2022-01-01'
end_date = '2022-12-31'

stock_data = yf.download(company, start=start_date, end=end_date)  # Historical Data Collection

# Visualize the actual stock prices over time
# Closing price: end of day value(traders (interest rates and inflation) and investors (commodities being bought))
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8))
ax1.plot(stock_data['Close'], label='Actual Prices', linestyle='-') # Plots stock_data (closing prices) over tiem 
ax1.set_title(f'{company} Stock Prices Over Time') 
ax1.set_xlabel('Date')
ax1.set_ylabel('Closing Prices')
ax1.legend(loc='lower left')

stock_data['Days'] = np.arange(1, len(stock_data) + 1)   # Evenly Spaced Values for Usage
X = stock_data[['Days', 'Open', 'High', 'Low']]  # Multilinear Regression (Multitude of variables) : Opening Prices, High Prices : Low Prices
y = stock_data['Close'] 

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)  # arbitutuary value (ensure values are consistent)

scalar = StandardScaler()  

X_train_scaled = scalar.fit_transform(X_train)  # Standardise data for best practice
X_test_scaled = scalar.transform(X_test)

model = LinearRegression()
model.fit(X_train_scaled, y_train)  # Train the model

y_pred = model.predict(X_test_scaled) # Prediction

# Visualize the regression line
ax2.scatter(X_test['Days'], y_test, color='blue', label='Actual Prices')
ax2.scatter(X_test['Days'], y_pred, color='red', label='Predicted Prices')
ax2.set_title(f'{company} Stock Prices Prediction (Multilinear Regression)')
ax2.set_xlabel('Days')
ax2.set_ylabel('Closing Prices')
ax2.legend(loc='lower left')

plt.tight_layout()

plt.show()

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)  # If low then accuracy is good (amount of error in model) Actual - predicted = Error^2 = ANS/Total
print(f"Mean Squared Error: {mse:.2f}")   # (squared diff between predicted and actual)
                                                # sensitivity to outliers value

