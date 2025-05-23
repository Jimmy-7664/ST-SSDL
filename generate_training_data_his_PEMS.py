from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse
import numpy as np
import os
import pandas as pd
from datetime import datetime,timedelta


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
    # read data
    dataset_name = args.dataset
    data = np.load(f"{args.traffic_df_filename}")["data"]
    data = data[..., 0]
    df = pd.DataFrame(data)
    print("raw time series shape: {0}".format(data.shape))
    num_nodes={
        "PEMS04": 307,
        "PEMS07": 883,
        "PEMS08": 170,
    }
    start_dates = {
        "PEMS04": "2018-01-01 00:00:00",
        "PEMS07": "2017-05-01 00:00:00",
        "PEMS08": "2016-07-01 00:00:00",
    }
    end_dates = {
        "PEMS04": "2018-02-28 23:55:00",
        "PEMS07": "2017-08-06 23:55:00",
        "PEMS08": "2016-08-31 23:55:00",
    }
    train_dates = {
        "PEMS04": "2018-02-16 23:55:00",
        "PEMS07": "2017-07-17 23:55:00",
        "PEMS08": "2016-08-18 23:55:00",
    }
    timeslots = pd.date_range(start_dates[dataset_name], end_dates[dataset_name], freq='5min')
    df.insert(0, 'timeslots', timeslots)
    df_new = df.set_index('timeslots', drop=True, append=False, inplace=False, verify_integrity=False)
    data = df_new.stack().reset_index()
    data.columns = ['timestamp', 'sensorid', 'flow']
    data['timeinday'] = (data['timestamp'].values - data['timestamp'].values.astype("datetime64[D]")) / np.timedelta64(
        1, "D")
    data['timestamp'] = pd.to_datetime(data['timestamp'])
    data['weekdaytime'] = data['timestamp'].dt.weekday * 288 + (
                data['timestamp'].dt.hour * 60 + data['timestamp'].dt.minute) // 5

    df_train = data[(data.timestamp >= start_dates[dataset_name]) & (data.timestamp <= train_dates[dataset_name])]
    df_train = df_train[['sensorid', 'weekdaytime', 'flow']]

    def get_mean_without_null_values(data, null_val=0.):
        return data[data != null_val].mean()

    df_train_avg = df_train.groupby(['sensorid', 'weekdaytime']).aggregate(
        {'flow': get_mean_without_null_values}).reset_index()

    data_his = pd.merge(data, df_train_avg, on=['sensorid', 'weekdaytime'], suffixes=(None, '_y'))

    sensorid_list = df_new.columns
    timeslices = df_new.index
    mux = pd.MultiIndex.from_product([timeslices, sensorid_list], names=['timestamp', 'sensorid'])
    data_his = data_his.set_index(['timestamp', 'sensorid']).reindex(mux).reset_index()
    data_his = data_his[['timestamp', 'sensorid', 'flow', 'timeinday', 'flow_y']]

    print(np.array_equal(data_his['flow'].values.reshape(-1, num_nodes[dataset_name]), df_new.values))

    data_his = data_his[['flow', 'timeinday', 'flow_y']].values
    data_his = data_his.reshape(-1, num_nodes[dataset_name], 3)

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
    num_train = round(num_samples * 0.6)
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
    print("Train Samples:", x_train.shape)
    print("Val Samples:", x_val.shape)
    print("Test Samples:", x_test.shape)

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
    parser.add_argument('--dataset', type=str, choices=['PEMS04','PEMS07', 'PEMS08'], default='PEMS08', help='which dataset to run')
    parser.add_argument("--output_dir", type=str, default="PEMS04/", help="Output directory.")
    parser.add_argument("--traffic_df_filename", type=str, default="{}/{}.npz", help="Raw traffic readings.")
    args = parser.parse_args()
    args.output_dir = f'{args.dataset}/'
    args.traffic_df_filename = args.traffic_df_filename.format(args.dataset, args.dataset)
    main(args)
