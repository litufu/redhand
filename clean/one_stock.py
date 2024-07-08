import sys
import os
import sqlite3
import pandas as pd
import numpy as np
from pathlib import Path
sys.path.append(os.getcwd())
from sktime.classification.deep_learning.inceptiontime import InceptionTimeClassifier
from sktime.classification.kernel_based import RocketClassifier
from stock_dataset_new import get_one_stock_data_from_sqlite,compute_data_y,split_data


if __name__ == '__main__':
    # x: [n_instances, n_dimensions, series_length]
    # y: 1D iterable, of shape [n_instances]
    # ts_code = "000001.SZ"
    # path="inceptiontime"
    start_date = "20000101"
    end_date = "20231231"
    fh = 10
    step_length = 10
    freq = "d"
    clean = True
    conn = sqlite3.connect(r"D:\redhand\clean\data\tushare_db\tushare.db")
    df_index = pd.read_sql("select * from df_index;", conn)
    # D:\redhand\clean\data\model\inceptiontime_000001.SZ.zip

    df_ts_codes = pd.read_sql("select distinct ts_code from stock_all", conn)
    # network = InceptionTimeClassifier(n_epochs=20, verbose=True)
    clf = RocketClassifier(num_kernels=500,rocket_transform="multirocket",use_multivariate="yes")
    # network = network.load_from_path(Path(r"D:\redhand\clean\data\modelinceptiontime_000019.zip"))
    X_trains, y_trains, X_valids, y_valids = None,None,None,None
    for key, ts_code in enumerate(np.unique(df_ts_codes["ts_code"])):
        print(key,ts_code)
        if key % 20 == 0 and X_trains is not None:
            clf.fit(X_trains, y_trains)
            print(clf.score(X_valids, y_valids))
            clf.save(r"D:\redhand\clean\data\model\rocket_{}".format(ts_code[:6]))
        df = get_one_stock_data_from_sqlite(conn, ts_code, start_date, end_date)
        if len(df) < 2000:
            pass
        else:
            df_data = compute_data_y(df, df_index, fh, freq,clean)
            X_train_tmp, y_train_tmp, X_valid_tmp, y_valid_tmp = split_data(df_data, step_length)

            if X_trains is None:
                X_trains = X_train_tmp
                y_trains = y_train_tmp
                X_valids = X_valid_tmp
                y_valids = y_valid_tmp
            else:
                X_trains = np.concatenate((X_trains, X_train_tmp), axis=0)
                y_trains = np.concatenate((y_trains, y_train_tmp), axis=0)
                X_valids = np.concatenate((X_valids, X_valid_tmp), axis=0)
                y_valids = np.concatenate((y_valids, y_valid_tmp), axis=0)


    # network.fit(X_trains, y_trains)
    # print(network.score(X_valids, y_valids))
    # network.save(path)