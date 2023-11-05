import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Specify the stock symbol and the date range
company = 'GOOGL'
start_date = '2022-01-01'
end_date = '2022-12-31'

# Fetch historical stock prices
stock_data = yf.download(company, start=start_date, end=end_date)  # Fetches stocks between start-end date

# Visualize the actual stock prices over time
plt.plot(stock_data['Close'], label='Actual Prices')  # Closing Price
plt.title(f'{company} Stock Prices Over Time')
plt.xlabel('Date')
plt.ylabel('Closing Prices')
plt.legend()
plt.show()

# Extract target and organises the data
stock_data['Days'] = np.arange(1, len(stock_data) + 1)
print(stock_data)
X = stock_data[['Days']]
y = stock_data['Close']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) # 20% as test set and 80% as training set
                                                                                        # 42 is the seed                                            

# Initialize and train the linear regression model
model = LinearRegression()
model.fit(X_train, y_train) # Dats and Closing Stock Prices

# Make predictions on the test set
y_pred = model.predict(X_test)

# Visualize the regression line
plt.plot(X_test, y_test, label='Actual Prices')
plt.plot(X_test, y_pred, color='red', linewidth=3, label='Predicted Prices')
plt.title(f'{company} Stock Prices Prediction')
plt.xlabel('Days')
plt.ylabel('Closing Prices')
plt.legend()
plt.show()

# Evaluate the model
mse = mean_squared_error(y_test, y_pred) # Measure of the model's accuracy, where a lower MSE indicates better performance.
print(f"Mean Squared Error: {mse:.2f}")


# Linear Regression: Predictions of a variable on another variable