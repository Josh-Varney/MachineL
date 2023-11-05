# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Generate some sample data
np.random.seed(42)
days = np.arange(1, 101, 1)
prices = 50 + 2 * days + np.random.normal(0, 10, size=len(days))

# Create a DataFrame
data = pd.DataFrame({'Days': days, 'Prices': prices})

# Visualize the data
plt.scatter(data['Days'], data['Prices'], label='Actual Prices')
plt.title('Stock Prices Over Time')
plt.xlabel('Days')
plt.ylabel('Prices')
plt.legend()
plt.show()

# Split the data into features (X) and target (y)
X = data[['Days']]
y = data['Prices']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Visualize the regression line
plt.scatter(X_test, y_test, label='Actual Prices')
plt.plot(X_test, y_pred, color='red', linewidth=3, label='Predicted Prices')
plt.title('Stock Prices Prediction')
plt.xlabel('Days')
plt.ylabel('Prices')
plt.legend()
plt.show()

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse:.2f}")
