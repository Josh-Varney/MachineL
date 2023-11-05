from pycoingecko import CoinGeckoAPI
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np

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
start_date = datetime(2023, 1, 1)
end_date = datetime(2023, 1, 5)

timestamps, prices = get_historical_data(coin_id, vs_currency, start_date, end_date)           
                
days = np.arange(1, len(timestamps) + 1)
X = days.reshape(-1, 1)
print(prices)