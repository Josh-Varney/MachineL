from datetime import timedelta
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error
from sklearn.impute import SimpleImputer
from sklearn.ensemble import IsolationForest

train = pd.read_csv(r'C:\Users\Jrv12\Desktop\Data Science\BA Rating Prediction\airline_review.csv')
train_df = pd.DataFrame(train)

mapping_dict = {'Verified': 1, 'Not Verified': 0, pd.NaT: 0}
train_df['trip_verified'] = train_df['trip_verified'].map(mapping_dict)
train_df['trip_verified'] = train_df['trip_verified'].fillna(0)

columns_to_fill = ['aircraft', 'seat_type', 'route', 'date_flown']
train_df[columns_to_fill] = train_df[columns_to_fill].fillna('Unknown')
train_df['traveller_type'] = train_df['traveller_type'].fillna('Leisure')

drop_list = ['place', 'content', 'cabin_staff_service', 'food_beverages', 'entertainment', 'ground_service', 'value_for_money']
train_df.drop(columns=drop_list, inplace=True)

train_df.dropna(subset=['date', 'date_flown'], inplace=True) # Drop missing values

train_df['date'] = pd.to_datetime(train_df['date'], errors='coerce')
train_df['date_flown'] = pd.to_datetime(train_df['date_flown'], errors='coerce')

one_month_threshold = timedelta(days=30)
train_df = train_df[(train_df['trip_verified'] == 1) & (train_df['date_flown'] - train_df['date'] < one_month_threshold)]  # Time difference of a month

numeric_columns = train_df.select_dtypes(include=[np.number]).columns   # Numerical columns to detect outlier

imputer = SimpleImputer(strategy='most_frequent')   # Impute missing values with most frequent
data_imputed = pd.DataFrame(imputer.fit_transform(train_df[numeric_columns]), columns=numeric_columns)

outlier_detector = IsolationForest(contamination=0.1)      # Apply Isolation Forest for outlier detection
remove_outliers = outlier_detector.fit_predict(data_imputed)

train_df = train_df[remove_outliers != -1] # Outliers equal to -1

# Linear Regression Model
X = train_df[['seat_comfort']] # Add more features as needed
Y = train_df['rating']

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Linear Regression Model
model = LinearRegression()
model.fit(X_train, Y_train)

y_pred = model.predict(X_test)

mse = mean_squared_error(Y_test, y_pred)
print(f'Mean Squared Error: {mse}')

# Scatter plot for actual vs. predicted values
plt.figure(figsize=(10, 6))
sns.scatterplot(x=Y_test.values.flatten(), y=y_pred.flatten(), color='blue', alpha=0.5)
plt.plot([Y_test.min(), Y_test.max()], [Y_test.min(), Y_test.max()], linestyle='--', color='red', linewidth=2)  # Add a diagonal line for reference
plt.title('Actual vs. Predicted Ratings')
plt.xlabel('Actual Ratings')
plt.ylabel('Predicted Ratings')
plt.show()

# Assuming 'date_flown' is in datetime format
plt.figure(figsize=(12, 6))
sns.lineplot(x='date_flown', y='rating', data=train_df, marker='o')
plt.title('Ratings Over Time')
plt.xlabel('Date Flown')
plt.ylabel('Rating')
plt.xticks(rotation=40)  # Increase readability
plt.show()
