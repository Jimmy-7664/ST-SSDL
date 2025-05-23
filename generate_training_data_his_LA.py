from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse
import numpy as np
import os
import pandas as pd
import datetime


def generate_graph_seq2seq_io_data_with_his(data, x_offsets, y_offsets):
    """
    Generate samples from
    :param df:
    :param x_offsets:
    :param y_offsets:
    :return:
    # x: (epoch_size, input_length, num_nodes, input_dim)
    # y: (epoch_size, output_length, num_nodes, output_dim)
    """

    num_samples = data.shape[0]
    x, y = [], []
    # t is the index of the last observation.
    min_t = abs(min(x_offsets))
    max_t = abs(num_samples - abs(max(y_offsets)))  # Exclusive
    for t in range(min_t, max_t):
        x_t = data[t + x_offsets, ...]
        y_t = data[t + y_offsets, ...]
        x.append(x_t)
        y.append(y_t)
    x = np.stack(x, axis=0)
    y = np.stack(y, axis=0)
    return x, y


def generate_train_val_test(args):
    df = pd.HDFStore(args.traffic_df_filename)['/df']
    timeslots = pd.date_range('2012-03-01 00:00:00', '2012-06-27 23:55:00', freq='5min')
    print(len(timeslots))
    days = pd.date_range('2012-03-01 00:00:00', '2012-06-27 23:55:00', freq='1D')
    print(len(days), 0.8*len(days))
    train_days = pd.date_range('2012-03-01 00:00:00', '2012-06-03 23:55:00', freq='1D')
    print(len(train_days))
    print(len(timeslots), timeslots[0], timeslots[-1])

    data = df.stack().reset_index()
    data.columns = ['timestamp', 'sensorid', 'speed']

    data['timeinday'] = (data['timestamp'].values - data['timestamp'].values.astype("datetime64[D]")) / np.timedelta64(1, "D")

    data['timestamp'] = pd.to_datetime(data['timestamp'])
    data['weekdaytime'] = data['timestamp'].dt.weekday * 288 + (data['timestamp'].dt.hour * 60 + data['timestamp'].dt.minute)//5

    df_train = data[(data.timestamp >= '2012-03-01 00:00:00') & (data.timestamp <= '2012-06-03 23:55:00')]
    df_train = df_train[['sensorid', 'weekdaytime', 'speed']]

    def get_mean_without_null_values(data, null_val=0.):
        return data[data != null_val].mean()
    df_train_avg = df_train.groupby(['sensorid', 'weekdaytime']).aggregate({'speed': get_mean_without_null_values}).reset_index()
    data_his = pd.merge(data, df_train_avg, on=['sensorid', 'weekdaytime'], suffixes=(None, '_y'))

    sensorid_list = df.columns
    timeslices = df.index
    mux = pd.MultiIndex.from_product([timeslices, sensorid_list], names=['timestamp', 'sensorid'])
    data_his = data_his.set_index(['timestamp', 'sensorid']).reindex(mux).reset_index()
    data_his = data_his[['timestamp', 'sensorid', 'speed', 'timeinday', 'speed_y']]

    print(np.array_equal(data_his['speed'].values.reshape(-1, 207), df.values))
    
    data_his = data_his[['speed', 'timeinday', 'speed_y']].values
    data_his = data_his.reshape(-1, 207, 3)
    
    # 0 is the latest observed sample.
    x_offsets = np.sort(
        np.concatenate((np.arange(-11, 1, 1),))
    )
    # Predict the next one hour
    y_offsets = np.sort(np.arange(1, 13, 1))
    # x: (num_samples, input_length, num_nodes, input_dim)
    # y: (num_samples, output_length, num_nodes, output_dim)
    x, y = generate_graph_seq2seq_io_data_with_his(
        data_his,
        x_offsets=x_offsets,
        y_offsets=y_offsets
    )

    print("x shape: ", x.shape, ", y shape: ", y.shape)

    num_samples = x.shape[0]
    num_test = round(num_samples * 0.2)
    num_train = round(num_samples * 0.7)
    num_val = num_samples - num_test - num_train

    # train
    x_train, y_train = x[:num_train], y[:num_train]
    # val
    x_val, y_val = (
        x[num_train: num_train + num_val],
        y[num_train: num_train + num_val],
    )
    # test
    x_test, y_test = x[-num_test:], y[-num_test:]

    for cat in ["train", "val", "test"]:
        _x, _y = locals()["x_" + cat], locals()["y_" + cat]
        print(cat, "x: ", _x.shape, "y:", _y.shape)
        np.savez_compressed(
            os.path.join(args.output_dir, "%shis.npz" % cat),
            x=_x,
            y=_y,
            x_offsets=x_offsets.reshape(list(x_offsets.shape) + [1]),
            y_offsets=y_offsets.reshape(list(y_offsets.shape) + [1]),
        )


def main(args):
    print("Generating training data")
    generate_train_val_test(args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, choices=['METRLA'], default='METRLA', help='which dataset to run')
    parser.add_argument("--output_dir", type=str, default="METRLA/", help="Output directory.")
    parser.add_argument("--traffic_df_filename", type=str, default="METRLA/metr-la.h5", help="Raw traffic readings.")
    args = parser.parse_args()
    args.output_dir = f'{args.dataset}/'
    if args.dataset == 'METRLA':
        args.traffic_df_filename = f'{args.dataset}/metr-la.h5'
    main(args)
