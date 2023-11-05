import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from alpha_vantage.timeseries import TimeSeries
import datetime as dt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM

def fetch_data(symbol, start_date, end_date, api_key):
    try:
        ts = TimeSeries(key=api_key, output_format='pandas')
        data, _ = ts.get_daily(symbol=symbol, outputsize='full')
        data = data[start_date:end_date]
        return data
    except Exception as e:
        print(f"Failed to download data for {symbol} from Alpha Vantage: {e}")
        return None

def preprocess_data(data, prediction_date):
    if data is not None and not data.empty:
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(data['4. close'].values.reshape(-1, 1))

        x_train, y_train = [], []

        for x in range(prediction_date, len(scaled_data)):
            x_train.append(scaled_data[x - prediction_date : x, 0])
            y_train.append(scaled_data[x, 0])

        x_train, y_train = np.array(x_train), np.array(y_train)
        x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

        return x_train, y_train, scaler
    else:
        return None, None, None

def build_model(input_shape):
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50))
    model.add(Dropout(0.2))
    model.add(Dense(units=1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def train_model(model, x_train, y_train, epochs, batch_size):
    model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size)
    return model

def make_predictions(model, model_inputs, prediction_date, scaler):
    x_test = np.array(model_inputs)
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

    predicted_prices = model.predict(x_test)
    predicted_prices = scaler.inverse_transform(predicted_prices.reshape(-1, 1))

    return predicted_prices

def plot_results(actual_prices, predicted_prices):
    plt.plot(actual_prices, color="black", label='Company Price')
    plt.plot(predicted_prices, color="green")
    plt.title('Company Shared Price')
    plt.xlabel('Time')
    plt.ylabel('Company Shared Price')
    plt.legend()
    plt.show()

# Main script
company_symbol = 'MSFT'
start_date_train = dt.datetime(2022, 1, 1)
end_date_train = dt.datetime(2023, 1, 1)
prediction_date_train = 60
api_key = 'D90RS993YTOYAWE9'  # Replace with your actual API key

data_train = fetch_data(company_symbol, start_date_train, end_date_train, api_key)

if data_train is not None and not data_train.empty:
    x_train, y_train, scaler = preprocess_data(data_train, prediction_date_train)

    model = build_model(input_shape=(x_train.shape[1], 1))
    model = train_model(model, x_train, y_train, epochs=25, batch_size=32)

    # Test Data
    start_date_test = dt.datetime(2020, 1, 1)
    end_date_test = dt.datetime.now()
    prediction_date_test = 60

    data_test = fetch_data(company_symbol, start_date_test, end_date_test, api_key)

    if data_test is not None and not data_test.empty:
        actual_prices_test = data_test['4. close'].values

        total_dataset = pd.concat((data_train['4. close'], data_test['4. close']), axis=0)
        model_inputs_test = total_dataset[len(total_dataset) - len(data_test) - prediction_date_test :].values
        predicted_prices_test = make_predictions(model, model_inputs_test, prediction_date_test, scaler)

        # Plot Test Predictions
        plot_results(actual_prices_test, predicted_prices_test)
else:
    print(f"Data not available for {company_symbol} from Alpha Vantage. Please check if the symbol is valid or update your API key.")
