import numpy as np
import pandas as pd
import torch
import sqlite3

from tsai.all import *
from torch.utils.data import Dataset
from datetime import datetime
from sklearn.preprocessing import OneHotEncoder
from download_data import transform_data, categorize
from MYTT.apply_mytt import indicatior
from constants import TOTAL_TRAIN_DATA_LENGTH, TOTAL_TEST_DATA_LENGTH, X_TRAIN_PATH, Y_TRAIN_PATH, X_VALID_PATH, \
    Y_VALID_PATH

start_date = "20000101"
end_date = "20231231"
conn = sqlite3.connect(r"D:\redhand\clean\data\tushare_db\tushare.db",check_same_thread=False)

def transform_date(origin, contain_split=True):
    '''
    将%Y-%m-%d转换成%Y%m%d
    '''
    if contain_split:
        return (datetime.strptime(str(origin), "%Y-%m-%d")).strftime('%Y%m%d')
    else:
        return (datetime.strptime(str(origin), "%Y%m%d")).strftime('%Y-%m-%d')


def get_one_stock_data_from_sqlite(ts_code, start_date, end_date, is_merge_index,table="stock_all"):
    start_date = transform_date(start_date, False)
    end_date = transform_date(end_date, False)
    df_stock = pd.read_sql(
        '''SELECT * FROM  {} where ts_code='{}' AND trade_date>='{}' AND trade_date<='{}';'''.format(table,ts_code,
                                                                                                            start_date,
                                                                                                            end_date),
        con=conn)  # 把数据库的入口传给它 # 简单明了的 sql 语句
    df_stock["trade_date"] = df_stock["trade_date"].apply(
        lambda x: (datetime.strptime(x, "%Y-%m-%d")).strftime('%Y%m%d'))
    df_stock.dropna(inplace=True)
    df_stock["trade_date"] = df_stock["trade_date"].astype(int)
    if is_merge_index:
        df_merge = df_stock.merge(df_index, how="left", suffixes=["_stock", "_index"], on="trade_date")
    else:
        df_merge = df_stock.copy(deep=True)
    return df_merge

def get_data_length(data_length, step_length, fh):
    total_length = data_length - fh - step_length
    valid_length = int(total_length * 0.2)
    train_length = total_length - valid_length
    return train_length, valid_length


def save_stock_to_db(df_stock_basic):
    start = False
    for key, ts_code in enumerate(list(df_stock_basic["ts_code"])):
        print(ts_code)
        # if ts_code == "001379.SZ":
        #     start = True
        if start:
            df = get_one_stock_data_from_sqlite(ts_code, start_date, end_date, False)
            if len(df) < 20:
                continue
            # 股票按照交易日降序排列
            df = df.sort_values(by="trade_date", ascending=True, ignore_index=True)
            # 添加股票技术指标
            df = indicatior(df)
            df.dropna(inplace=True)
            # print(df)
            df.to_sql("stock_daily", conn, if_exists="append", index=False)


def compute_data_y(df,fh,df_index):
    df = indicatior(df)
    df = df.merge(df_index, how="left", suffixes=["_stock", "_index"], on="trade_date")
    df = df.drop(labels=["ts_code", "trade_date", "ts_code_sh_index", "ts_code_sz_index"], axis=1)
    df["next_n_close"] = df["close"].shift(-fh)
    df.dropna(inplace=True)
    df["next_n_pct_chg"] = ((df["next_n_close"] - df["close"]) / df["close"]) * 100
    # 将涨幅分类为0到8，进行9分类
    df_data = df.copy()
    df["target"] = df["next_n_pct_chg"].apply(categorize)
    # 进行数据归一化
    df_data = (df_data - df_data.mean()) / df_data.std()
    df_data["target"] = df["target"]
    df_data = df_data.drop(labels=["next_n_close", "next_n_pct_chg"], axis=1)
    return df_data

def split_data(df,step_length):
    columns = df.columns.tolist()
    # X：所有数据都是X,包括股票的信息和指数的信息
    get_x = columns[:-1]
    # y的值:columns的索引：使用股票的pct_chg
    get_y = "target"
    _, X_train, y_train, X_valid, y_valid = transform_data(df, step_length, get_x, get_y, horizon=1,
                                                           stride=1,
                                                           start=0, seq_first=True)
    if y_train is None:
        return None, None, None, None
    if (np.any(np.isnan(X_train))) or (np.any(np.isnan(y_train))) or (np.any(np.isnan(X_valid))) or (np.any(
            np.isnan(y_valid))):
        return None, None, None, None
    return X_train, y_train, X_valid, y_valid


def find_loc(num, number_list):
    '''
    从一个列表中，找出一个大于左边，小于右边数对应的左索引和右索引
    :param num:
    :param number_list:
    :return:
    '''
    for key, value in enumerate(number_list):
        if num <= value:
            return key, key + 1
    raise Exception("在{}中未找到{}的索引".format(number_list, num))



class StockPytorchDatasetNew(Dataset):
    '''
    这个是一个pytorch Dataset,也可以用于keras data,需要将y转换为one_hot形式
    原理是：将每一支股票数据保存为npy格式，然后建立一张统计表，统计每个npy包含的sample个数
    包含训练和测试数据集

    '''

    def __init__(self, one_hot=False, train=True, keras_data=False, step_length=100, fh=15):
        '''

        :param annotations_file:记录x_train_path,y_train_path,x_valid_path,y_valid_path,train_data_length,
                                test_data_length,total_train_data_length,total_test_data_length
        :param one_hot:如果是keras model需要one_hot，如果是pytorch不需要one_hot
        :param train:生成训练集还是测试集
        '''
        self.train = train
        self.one_hot = one_hot
        self.keras_data = keras_data
        self.step_length = step_length
        self.fh = fh
        self.conn = conn
        self.df_index = pd.read_sql("select * from df_index;", conn)
        self.df_stock_basic = pd.read_sql("select * from stock_basic;", conn)





    def get_length(self):
        '''

        :param step_length:
        :param fh:
        :return:
        '''
        ts_codes = []
        total_train_data_lengths = []
        total_test_data_lengths = []
        train_data_lengths = []
        test_data_lengths = []
        total_train_data_length = 0
        total_test_data_length = 0
        for key, ts_code in enumerate(list(self.df_stock_basic["ts_code"])):
            print(ts_code)
            df = get_one_stock_data_from_sqlite(ts_code, start_date, end_date, False)
            if len(df) < self.step_length + self.fh + 1:
                continue
            df_data = compute_data_y(df,self.fh,self.df_index)
            X_train, y_train, X_valid, y_valid = split_data(df_data,self.step_length)
            if y_train is None:
                continue
            ts_codes.append(ts_code)
            train_data_lengths.append(len(y_train))
            test_data_lengths.append(len(y_valid))
            total_train_data_length += len(y_train)
            total_train_data_lengths.append(total_train_data_length)
            total_test_data_length += len(y_valid)
            total_test_data_lengths.append(total_test_data_length)
            print(len(y_train), len(y_valid))
        df_record = pd.DataFrame({"ts_code": ts_codes,
                                  "train_data_length": train_data_lengths,
                                  "test_data_length": test_data_lengths,
                                  "total_train_data_length": total_train_data_lengths,
                                  "total_test_data_length": total_test_data_lengths,
                                  })
        print(df_record)
        df_record.to_sql("step_length_{}_fh_{}".format(self.step_length, self.fh), self.conn, index=False)

    def get_records(self):
        try:
            df = pd.read_sql("select * from step_length_{}_fh_{}".format(self.step_length, self.fh), self.conn)
        except pd.errors.DatabaseError:
            self.get_length()
            df = pd.read_sql("select * from step_length_{}_fh_{}".format(self.step_length, self.fh), self.conn)
        return df

    def __len__(self):
        self.df_records = self.get_records()
        if self.train:
            return self.df_records[TOTAL_TRAIN_DATA_LENGTH].max()
        else:
            return self.df_records[TOTAL_TEST_DATA_LENGTH].max()

    def get_data(self, idx, total_data_length_cl, train):
        self.df_records = self.get_records()
        search_len = idx + 1
        pos_start, pos_end = find_loc(search_len, self.df_records[total_data_length_cl])
        if pos_start == 0:
            data_pos = idx
        else:
            data_pos = idx - self.df_records[total_data_length_cl].iloc[pos_start - 1]
        ts_code = self.df_records["ts_code"].iloc[pos_start]
        df = get_one_stock_data_from_sqlite(ts_code, start_date, end_date, False)
        df_data = compute_data_y(df,self.fh,self.df_index)
        X_train, y_train, X_valid, y_valid = split_data(df_data,self.step_length)
        if train:
            x = X_train[data_pos]
            y = y_train[data_pos]
        else:
            x = X_valid[data_pos]
            y = y_valid[data_pos]

        if self.keras_data:
            x = x.transpose((1, 0))
        if self.one_hot:
            y = y.reshape(-1, 1)
            onehot_encoder = OneHotEncoder(categories=[[0, 1, 2, 3, 4, 5, 6, 7, 8]], sparse_output=False)
            y = onehot_encoder.fit_transform(y)
            y = np.squeeze(y)
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.long)

    def __getitem__(self, idx):
        if self.train:
            x, y = self.get_data(idx, TOTAL_TRAIN_DATA_LENGTH, self.train)
            return x, y
        else:
            x, y = self.get_data(idx, TOTAL_TEST_DATA_LENGTH, self.train)
            return x, y


def delete_table(conn,tablename):
    cursor = conn.cursor()
    try:
        cursor.execute("DROP TABLE {}".format(tablename))
    except Exception as e:
        print(e)


def new_fit(step_length,fh):
    import keras
    from torch.utils.data import DataLoader
    from sktime.classification.deep_learning.inceptiontime import InceptionTimeClassifier
    # 创建pytorch 数据集，y onehot为True
    # conn = sqlite3.connect(r"D:\redhand\clean\data\tushare_db\tushare.db",check_same_thread=False)
    train_data = StockPytorchDatasetNew(True, True, True, step_length, fh)
    train_dataloader = DataLoader(train_data, batch_size=64, shuffle=False)
    test_data = StockPytorchDatasetNew(True, False, True, step_length, fh)
    test_dataloader = DataLoader(test_data, batch_size=64, shuffle=False)
    # 从sktime创建inceptiontime模型
    network = InceptionTimeClassifier(verbose=True,depth=9)
    # 模型初始化，
    model_ = network.build_model((step_length, 96), 9)
    model_save_path = r"D:\redhand\clean\data\state_dict\inceptiontime_new_{}_{}.keras".format(step_length,fh)
    # 模型参数初始化
    # model_ = keras.saving.load_model(model_save_path)
    # 开始训练
    csv_logger = keras.callbacks.CSVLogger(r"D:\redhand\clean\data\log\inceptiontime_log_{}_{}.csv".format(step_length,fh), separator=",", append=True)
    batch_print_callback = keras.callbacks.LambdaCallback(
        on_train_batch_end=lambda batch, logs: print("batch:{};logs:{}".format(batch, logs)))
    model_checkpoint_callback = keras.callbacks.ModelCheckpoint(
        filepath=model_save_path,
        monitor='val_accuracy',
        mode='max',
        save_weights_only=False,
        save_freq=50,
        save_best_only=False)
    reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=50, min_lr=0.0001)
    remot = keras.callbacks.RemoteMonitor(
        root="http://121.41.21.130:9000",
        path="/publish/epoch/end/",
        field="data",
        headers=None,
        send_as_json=False,
    )
    history = model_.fit(train_dataloader, epochs=100, validation_data=test_dataloader,
                         callbacks=[csv_logger, batch_print_callback,
                                    model_checkpoint_callback,
                                    reduce_lr,remot])
    print(history.history)
    # 保存参数
    # loaded_model = keras.saving.load_model(r"D:\redhand\project\data\state_dict\inceptiontime.keras")


if __name__ == '__main__':
    new_fit(100, 15)
    # # save_stock_to_db()
    # from torch.utils.data import DataLoader
    #
    # conn_sqlite = sqlite3.connect(r"D:\redhand\clean\data\tushare_db\tushare.db")
    # df_index = pd.read_sql("select * from df_index;", conn_sqlite)
    # # delete_table(conn_sqlite, "step_length_100_fh_15")
    # train_data = StockPytorchDatasetNew(conn_sqlite, True, True, True, 100, 15)
    # train_dataloader = DataLoader(train_data, batch_size=64, shuffle=False)
    # test_data = StockPytorchDatasetNew(conn_sqlite, True, False, True, 100, 15)
    # test_dataloader = DataLoader(test_data, batch_size=64, shuffle=False)
    # # # for key, (x_train, y_train) in enumerate(train_dataloader):
    # # #     print(key)
    # # #     print("------------------------------------")
    # # #     print(x_train)
    # # #     print(y_train)
    # # #     if key > 3:
    # # #         break
    # #
    # # df = pd.read_sql("select * from step_length_100_fh_15",conn)
    # # print(df)
    # train_features, train_labels = next(iter(train_dataloader))
    # test_features, test_labels = next(iter(test_dataloader))
    # # #
    # print(train_features)
    # print(f"Train Feature batch shape: {train_features.size()}")
    # print(f"Train Labels batch shape: {train_labels.size()}")
    # print(train_labels)
    # print(f"Test Labels batch shape: {test_features.size()}")
    # print(f"Test Labels batch shape: {test_labels.size()}")
    #
    # df = get_one_stock_data_from_sqlite("000001.SZ", start_date, end_date, False)
    # df_data = compute_data_y(df, 15, df_index)
    # X_train, y_train, X_valid, y_valid = split_data(df_data, 100)
    # X_train = X_train.transpose((0, 2, 1))
    # print(X_train[0:64])
    # print(y_train[0:64])
    # # get_length()
