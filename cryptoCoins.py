from pycoingecko import CoinGeckoAPI
from datetime import datetime

cg = CoinGeckoAPI()

# Define the coin id and currency
coin_id = 'bitcoin'
vs_currency = 'gbp'

coin_id2 = 'ethereum'

# Define the start and end date (in UNIX timestamp format)
start_date = datetime(2023, 1, 1)
end_date = datetime(2023, 1, 5)

start_timestamp = int(start_date.timestamp())
end_timestamp = int(end_date.timestamp())

# Get historical market chart data
historical_data = cg.get_coin_market_chart_range_by_id(id=coin_id, vs_currency=vs_currency, from_timestamp=start_timestamp, to_timestamp=end_timestamp)

# Extracting prices and timestamps
prices = historical_data['prices']
timestamps = [datetime.utcfromtimestamp(timestamp[0]/1000).strftime('%Y-%m-%d %H:%M:%S') for timestamp in prices]

# Print the results
for timestamp, price in zip(timestamps, prices):
    print(f"Date: {timestamp}, Price: {price[1]} {vs_currency}")
