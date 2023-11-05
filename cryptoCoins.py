from pycoingecko import CoinGeckoAPI
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

cg = CoinGeckoAPI()

def get_historical_data(coin_id, vs_currency, start_date, end_date):
    start_timestamp = int(start_date.timestamp())
    end_timestamp = int(end_date.timestamp())
    historical_data = cg.get_coin_market_chart_range_by_id(id=coin_id, vs_currency=vs_currency, from_timestamp=start_timestamp, to_timestamp=end_timestamp)
    prices = historical_data['prices']
    timestamps = [datetime.utcfromtimestamp(timestamp[0]/1000).strftime('%Y-%m-%d %H:%M:%S') for timestamp in prices]
    # Converts into seconds and converts into the following format
    return timestamps, prices

# Define the coin id and currency
coin_id = 'bitcoin'
vs_currency = 'gbp'

coin_id2 = 'ethereum'

# Define the start and end date (in UNIX timestamp format)
start_date = datetime(2009, 1, 1)
end_date = datetime(2023, 1, 5)

timestamps, prices = get_historical_data(coin_id2, vs_currency, start_date, end_date)           
                
days = np.arange(1, len(timestamps) + 1)
X = days.reshape(-1, 1)
Y = [price[1] for price in prices]

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=52)

model = LinearRegression()
model.fit(X_train, Y_train)

Y_pred = model.predict(X_test)

plt.figure(figsize=(10,6))
plt.plot(X_test, Y_test, label='Actual Prices', linestyle='-', marker='o', color='blue', alpha=0.7)
plt.plot(X_test, Y_pred, color='red', linewidth=2, label='Predicted Prices', marker='o', linestyle='--')
plt.title(f'{coin_id.capitalize()} Price Prediction')
plt.xlabel('Days')
plt.ylabel('Prices')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)
plt.xticks(rotation=45, ha='right')
plt.show()