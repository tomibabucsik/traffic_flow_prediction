import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def load_and_preprocess(path, seq_length=48):
    df = pd.read_csv(path)
    df['date_time'] = pd.to_datetime(df['date_time'], dayfirst=True)
    df = df.sort_values('date_time')
    df.set_index('date_time', inplace=True)

    traffic_data = df[['traffic_volume']]
    scaler = MinMaxScaler()
    traffic_data_scaled = scaler.fit_transform(traffic_data)

    def create_sequences(data, seq_len):
        xs, ys = [], []
        for i in range(len(data) - seq_len):
            xs.append(data[i:i+seq_len])
            ys.append(data[i+seq_len])
        return np.array(xs), np.array(ys)

    X, y = create_sequences(traffic_data_scaled, seq_length)
    return X, y, scaler