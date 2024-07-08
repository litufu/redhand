import sqlite3
import os
import torch
from datetime import datetime
from tsai.all import *
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from stock_dataset_new import split_data, find_loc
from constants import TOTAL_TRAIN_DATA_LENGTH, TOTAL_TEST_DATA_LENGTH, CATEGORIES


def get_distribution(conn, symbol, fh, table="futures"):
    '''
    获取数据分布
    :param df: dataframe
    :param col_name:
    :return:
    '''
    print(symbol)
    df = pd.read_sql(
        '''SELECT * FROM  {} where symbol='{}' limit 1000;'''.format(table, symbol),
        con=conn)  # 把数据库的入口传给它 # 简单明了的 sql 语句
    df.dropna(inplace=True)
    df = df.sort_values(by="trade_date", ascending=True, ignore_index=True)
    # 计算未来fh天的收盘价
    df["next_n_close"] = df["close"].shift(-fh)
    # 删除最后的fh天的信息，因为无法取得未来fh天的收盘价
    df.dropna(inplace=True)
    # 计算未来fh天的股票变动比例
    df["next_n_pct_chg"] = ((df["next_n_close"] - df["close"]) / df["close"]) * 100
    # 计算未来fh天的股票变动方向
    df_res = df["next_n_pct_chg"].describe(percentiles=[.33, .66])
    df_res = df_res.to_frame().T
    df_res["symbol"] = symbol
    df_res["fh"] = fh
    print(df_res)
    df_res.to_csv(r'D:\期货\res_1.csv', mode='a', header=False, index=False)


def get_all_distribution(conn, table="futures"):
    '''
    获取所有股票的分布
    :param conn:
    :param table:
    :return:
    '''
    symbols = pd.read_sql('''SELECT DISTINCT symbol FROM {}'''.format(table), con=conn)['symbol'].tolist()
    for symbol in symbols:
        for fh in range(1, 11):
            get_distribution(conn, symbol, fh, table)


def get_files_by_suffix(dir1, dir2, suffix):
    '''
    获取路径下面后缀的文件
    :param dir:
    :param suffix:
    :return:
    '''
    files_res = []
    file_paths_res1 = []
    file_paths_res2 = []
    for root, dirs, files in os.walk(dir1):
        for file in files:
            if file.endswith(suffix):
                file_path1 = os.path.join(root, file)
                file_paths_res1.append(file_path1)
                file_path2 = os.path.join(dir2, file)
                file_paths_res2.append(file_path2)
                files_res.append(file)

    return files_res, file_paths_res1, file_paths_res2


def get_date(datetime_str):
    '''
    将日期时间转换为日期格式
    :param datetime_str: 日期时间字符串
    :return:
    '''
    dt_obj = datetime.strptime(datetime_str, '%Y-%m-%d %H:%M:%S')
    dt_obj_formatted = dt_obj.strftime('%Y-%m-%d')

    return dt_obj_formatted


def futures_trade_date():
    files, file_paths1, file_paths2 = get_files_by_suffix(r'D:\期货\futures_5', r'D:\期货\换月表\换月表', '.csv')
    for key, file in enumerate(files):
        path1 = file_paths1[key]
        path2 = file_paths2[key]
        df1 = pd.read_csv(path1)
        df1['trade_date'] = df1['datetime'].apply(get_date)
        # print(df1.head())
        df2 = pd.read_csv(path2)
        # print(df2.head())
        df_merge = df1.merge(df2, on='trade_date', how='inner')
        # if df_merge.columns.values.tolist() == ['datetime','open','high','low','close','volume','position','trade_date','symbol']:
        #     pass
        # else:
        #     print(file)
        # print(df_merge.columns.values)
        conn = sqlite3.connect(r"D:\futures\futures.db")
        df_merge.to_sql('futures', conn, if_exists='append', index=False)


def categorize_future_change(a, fh=1, freq="m"):
    '''
    将涨跌幅分类
    	平均值项:25%	平均值项:75%
    1	-8.00%	8.00%
    2	-10.00%	11.00%
    3	-13.00%	12.00%
    4	-14.00%	14.00%
    5	-15.00%	16.00%
    6	-16.00%	17.00%
    7	-17.00%	18.00%
    8	-18.00%	20.00%
    9	-19.00%	21.00%
    10	-20.00%	23.00%


    :param a: 浮点数，代表涨跌幅
    :param fh: 未来期间，根据未来不同的时间来计算涨幅分类
    :param freq: 未来的频率
    :return:
    '''
    # 5分钟涨跌幅分类
    m_setting = {
        1: (-8.00, 8.00),
        2: (-10.00, 11.00),
        3: (-13.00, 12.00),
        4: (-14.00, 14.00),
        5: (-15.00, 16.00),
        6: (-16.00, 17.00),
        7: (-17.00, 18.00),
        8: (-18.00, 20.00),
        9: (-19.00, 21.00),
        10: (-20.00, 23.00)
    }
    num = 0
    # 如果是15分钟，一天是4个小时，一小时是4个15分钟，一天是16个15分钟
    if freq == "m":
        min = m_setting[fh][0]
        max = m_setting[fh][1]
    elif freq == "d":
        scale = fh / 5
    elif freq == "y":
        scale = fh * 10
    else:
        raise Exception("无法识别的频率freq,必须为'm'或者‘d'")

    if a < min / 100:
        num = 0
    elif min / 100 <= a < max / 100:
        num = 1
    elif a >= max / 100:
        num = 2
    return num


def get_one_future_data_from_sqlite(conn, symbol, table="futures"):
    '''
    从数据库获取指定日期间的股票信息
    :param symbol: 期货代码
    :param table: 从数据库查询那张表
    :return: 合并指数后的数据
    '''
    # 先获取一下股票信息，检查一下其日期格式，如果是%Y%m%d则不用转换，如果是
    df_future = pd.read_sql(
        '''SELECT * FROM  {} where symbol='{}' ;'''.format(table, symbol),
        con=conn)  # 把数据库的入口传给它 # 简单明了的 sql 语句
    df_future.dropna(inplace=True)

    return df_future


def compute_future_y(df, fh, freq):
    '''
    计算股票指标
    如果是日线则合并指数信息
    如果是分钟线不合并指数信息
    :param df:股票信息
    :param fh:未来几天
    :param freq:日线d,分钟线m
    :return:处理后的数据
    '''
    if freq == "d":
        df = df.sort_values(by="trade_date", ascending=True, ignore_index=True)
        df = df.drop(labels=["trade_date", "symbol"], axis=1)
    elif freq == "m":
        df = df.sort_values(by="datetime", ascending=True, ignore_index=True)
        df = df.drop(labels=["datetime", "trade_date", "symbol"], axis=1)
    # 计算未来fh天的收盘价
    df["next_n_close"] = df["close"].shift(-fh)
    # 删除最后的fh天的信息，因为无法取得未来fh天的收盘价
    df.dropna(inplace=True)
    # 计算未来fh天的股票变动比例
    df["next_n_pct_chg"] = ((df["next_n_close"] - df["close"]) / df["close"]) * 100
    # 复制一份数据到df_data
    df_data = df.copy()
    # 进行数据归一化
    df_data = (df_data - df_data.mean()) / df_data.std()
    # 将涨幅分类为0到8，进行9分类
    df["target"] = df["next_n_pct_chg"].apply(categorize_future_change, args=(fh, freq))
    # 将分类结果复制到df_data
    df_data["target"] = df["target"]
    # 将未来几天收盘价和涨幅从预测数据中删除
    df_data = df_data.drop(labels=["next_n_close", "next_n_pct_chg"], axis=1)
    return df_data


class FuturePytorchDatasetNew(Dataset):
    '''
    这个是一个pytorch Dataset,也可以用于keras data,需要将y转换为one_hot形式
    原理是：将每一支股票数据保存为npy格式，然后建立一张统计表，统计每个npy包含的sample个数
    包含训练和测试数据集

    '''

    def __init__(self, conn, one_hot=False, train=True, keras_data=False, step_length=24, fh=1, freq="m"):

        '''

        :param conn: 数据库连接
        :param one_hot: 是否将y转化为one_hot形式
        :param train: 是否取训练集
        :param keras_data: 是否转化为keras形式的数据，即n_samples,n_steps,n_vars
        :param step_length: 过去多少时间
        :param fh: 需要预测的未来多少时间
        :param freq: 频率，d:代表是日线数据，m:代表是分钟数据
        '''
        self.conn = conn
        self.one_hot = one_hot
        self.train = train
        self.keras_data = keras_data
        self.step_length = step_length
        self.fh = fh
        self.freq = freq

    def get_length(self):
        '''
        获取要训练和测试的数据集长度
        ts_code,train_data_length,test_data_length,total_train_data_length,total_test_data_length
        '''
        # 股票代码
        ts_codes = []
        # 训练数据总长度
        total_train_data_lengths = []
        # 测试数据总长度
        total_test_data_lengths = []
        # 训练数据长度
        train_data_lengths = []
        # 测试数据长度
        test_data_lengths = []
        # 总的训练数据长度
        total_train_data_length = 0
        # 总的测试数据长度
        total_test_data_length = 0

        df_ts_codes = pd.read_sql("select symbol from futures", self.conn)

        for key, ts_code in enumerate(np.unique(df_ts_codes["symbol"])):
            # 遍历所有的股票代码
            print(ts_code)
            # 获取股票日线或者分钟线数据
            df = get_one_future_data_from_sqlite(self.conn, ts_code)
            # 如果在开始时间和截止时间内，该股票的数据小于历史回顾和未来预测期数据，那么该股票将没有可测试的数据，因此不测试该股票
            if len(df) < self.step_length + self.fh + 1:
                continue
            # 计算股票指标并合并指数
            df_data = compute_future_y(df, self.fh, self.freq)
            print(df_data.shape)
            # 生成训练和测试数据集
            X_train, y_train, X_valid, y_valid = split_data(df_data, self.step_length)
            # 如果生成的数据为None
            if y_train is None:
                continue
            print(X_train.shape, y_train.shape, X_valid.shape, y_valid.shape)
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
        df_record.to_sql("step_length_{}_fh_{}".format(self.step_length, self.fh), self.conn, index=False,
                         if_exists="replace")

    def get_records(self):
        '''
        获取数据记录,如果没有的话，重新计算
        :return:
        '''
        try:
            df = pd.read_sql("select * from step_length_{}_fh_{}".format(self.step_length, self.fh), self.conn)
            if len(df) < 2:
                self.get_length()
                df = pd.read_sql("select * from step_length_{}_fh_{}".format(self.step_length, self.fh), self.conn)
        except pd.errors.DatabaseError:
            self.get_length()
            df = pd.read_sql("select * from step_length_{}_fh_{}".format(self.step_length, self.fh), self.conn)
        return df

    def __len__(self):
        '''
        获取数据长度
        :return:
        '''

        self.df_records = self.get_records()
        if self.train:
            return self.df_records[TOTAL_TRAIN_DATA_LENGTH].max()
        else:
            return self.df_records[TOTAL_TEST_DATA_LENGTH].max()

    def get_data(self, idx, total_data_length_cl):
        '''
        根据idx从数据集中获取数据
        :param idx:
        :param total_data_length_cl:
        :param train:
        :return:
        '''
        # 获取idx在那只股票的什么位置
        self.df_records = self.get_records()
        search_len = idx + 1
        pos_start, pos_end = find_loc(search_len, self.df_records[total_data_length_cl])
        if pos_start == 0:
            data_pos = idx
        else:
            data_pos = idx - self.df_records[total_data_length_cl].iloc[pos_start - 1]
        # 获取股票信息
        ts_code = self.df_records["ts_code"].iloc[pos_start]
        df = get_one_future_data_from_sqlite(self.conn, ts_code)
        df_data = compute_future_y(df, self.fh, self.freq)
        X_train, y_train, X_valid, y_valid = split_data(df_data, self.step_length)
        # 获取期货对应位置的数据
        if self.train:
            x = X_train[data_pos]
            y = y_train[data_pos]
        else:
            x = X_valid[data_pos]
            y = y_valid[data_pos]
        # 将获取的位置数据进行转换
        if self.keras_data:
            x = x.transpose((1, 0))
        if self.one_hot:
            y = y.reshape(-1, 1)
            onehot_encoder = OneHotEncoder(categories=[CATEGORIES], sparse_output=False)
            y = onehot_encoder.fit_transform(y)
            y = np.squeeze(y)
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.long)

    def __getitem__(self, idx):
        if self.train:
            x, y = self.get_data(idx, TOTAL_TRAIN_DATA_LENGTH)
            return x, y
        else:
            x, y = self.get_data(idx, TOTAL_TEST_DATA_LENGTH)
            return x, y


def delete_table(conn, tablename):
    '''
    删除数据表
    :param conn:
    :param tablename:
    :return:
    '''
    cursor = conn.cursor()
    try:
        cursor.execute("DROP TABLE {}".format(tablename))
    except Exception as e:
        print(e)


def new_fit(settings, load_model=False, batch_size=64, dataset=FuturePytorchDatasetNew):
    '''
    根据配置信息进行训练
    :param settings: 配置信息
    :return:
    '''
    import keras
    from torch.utils.data import DataLoader
    from sktime.classification.deep_learning.inceptiontime import InceptionTimeClassifier
    db = settings["db"]
    conn = sqlite3.connect(db, check_same_thread=False)
    step_length = settings["step_length"]
    fh = settings["fh"]
    freq = settings["freq"]
    n_vars = settings["n_vars"]
    depth = settings["depth"]
    if settings["model"] == "lstmfcn":
        network = LSTMFCNClassifier(verbose=True, n_epochs=100,batch_size=batch_size)
        model_save_path = r"D:\redhand\clean\data\state_dict\lstmfcn_future_{}_{}_{}_{}.keras".format(title, freq,
                                                                                                       step_length, fh)
    elif settings["model"] == "inceptiontime":
        # 从sktime创建inceptiontime模型
        network = InceptionTimeClassifier(verbose=True, depth=depth)
        model_save_path = r"D:\redhand\clean\data\state_dict\inceptiontime_future_{}_{}_{}_{}.keras".format(title, freq,
                                                                                                     step_length, fh)
    else:
        return None
    # 创建pytorch 数据集，y onehot为True

    train_data = dataset(conn, True, True, True, step_length, fh, freq)
    train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=False)
    test_data = dataset(conn, True, False, True, step_length, fh, freq)
    test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=False)
    # 从sktime创建inceptiontime模型
    # network = InceptionTimeClassifier(verbose=True, depth=depth)
    # 模型初始化，
    model_ = network.build_model((step_length, n_vars), len(CATEGORIES))
    # model_save_path = r"D:\redhand\clean\data\state_dict\inceptiontime_future_{}_{}_{}.keras".format(freq, step_length,
    #                                                                                                  fh)
    # 模型参数初始化
    # model_save_path = r"/clean/data/state_dict/inceptiontime_new_150_15_bak.keras"
    if load_model:
        model_ = keras.saving.load_model(model_save_path)
    # 开始训练
    csv_logger = keras.callbacks.CSVLogger(
        r"D:\redhand\clean\data\log\inceptiontime_log_future_{}_{}_{}.csv".format(freq, step_length, fh), separator=",",
        append=True)
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
    history = model_.fit(train_dataloader, epochs=100, validation_data=test_dataloader,
                         callbacks=[csv_logger, batch_print_callback,
                                    model_checkpoint_callback,
                                    reduce_lr])
    print(history.history)
    # 保存参数
    # loaded_model = keras.saving.load_model(r"D:\redhand\project\data\state_dict\inceptiontime.keras")


if __name__ == '__main__':
    # futures_trade_date()
    settings_minute_10_10 = {
        "db": r"D:\futures\futures.db",
        "table": "futures",
        "step_length": 10,  # 16*20
        "fh": 10,  # 16*5
        "freq": "m",  # d代表day,m代表minute
        "n_vars": 6,
        "depth": 6,
        "model": "lstmfcn",  # lstmfcn,inceptiontime
    }
    new_fit(settings_minute_10_10,load_model=False)
    # conn = sqlite3.connect(r"D:\futures\futures.db", check_same_thread=False)
    # get_all_distribution(conn)
