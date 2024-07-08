import numpy as np
import pandas as pd
import torch
import sqlite3
import keras
from tsai.all import *
from torch.utils.data import Dataset
from datetime import datetime
from sklearn.utils import check_random_state
from sklearn.preprocessing import OneHotEncoder
from MYTT.apply_mytt import indicatior
from constants import TOTAL_TRAIN_DATA_LENGTH, TOTAL_TEST_DATA_LENGTH, X_TRAIN_PATH, Y_TRAIN_PATH, X_VALID_PATH, \
    CATEGORIES


def get_stock_distribution(conn, ts_code, fh, table="stock_all"):
    '''
    获取数据分布
    :param df: dataframe
    :param col_name:
    :return:
    '''
    print(ts_code)
    df = pd.read_sql(
        '''SELECT * FROM  {} where ts_code='{}' limit 1000;'''.format(table, ts_code),
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
    df_res["symbol"] = ts_code
    df_res["fh"] = fh
    print(df_res)
    df_res.to_csv(r'D:\stock\res_2.csv', mode='a', header=False, index=False)


def get_stock_all_distribution(conn, table="stock_all"):
    '''
    获取所有股票的分布
    :param conn:
    :param table:
    :return:
    '''
    ts_codes = pd.read_sql('''SELECT DISTINCT ts_code FROM {}'''.format(table), con=conn)['ts_code'].tolist()
    for key, ts_code in enumerate(ts_codes):
        if key % 3 != 0:
            continue
        if key > 1000:
            return
        for fh in range(1, 11):
            get_stock_distribution(conn, ts_code, fh, table)


def categorize(a, fh=5, freq="d"):
    '''
    将涨跌幅分类
    区间	类别
    1	 -0.90 	 0.90
    2	 -1.40 	 1.30
    3	 -1.80 	 1.70
    4	 -2.10 	 1.90
    5	 -2.40 	 2.20
    6	 -2.70 	 2.40
    7	 -2.90 	 2.50
    8	 -3.10 	 2.70
    9	 -3.40 	 2.80
    10	 -3.60 	 2.90


    :param a: 浮点数，代表涨跌幅
    :param fh: 未来期间，根据未来不同的时间来计算涨幅分类
    :param freq: 未来的频率
    :return:
    '''
    scale = 1
    num = 0
    # 5分钟涨跌幅分类
    d_setting = {
        1: (-0.9, 0.9),
        2: (-1.4, 1.3),
        3: (-1.8, 1.7),
        4: (-2.1, 1.9),
        5: (-2.4, 2.2),
        6: (-2.7, 2.4),
        7: (-2.9, 2.5),
        8: (-3.1, 2.7),
        9: (-3.4, 2.8),
        10: (-3.6, 2.9)
    }
    me_setting = {
        1: (-5, 5),
        2: (-10, 10),
        3: (-15, 15),
        4: (-20, 20),
        5: (-25, 25),
        6: (-30, 30),
        7: (-35, 35),
        8: (-40, 40),
        9: (-45, 45),
        10: (-50, 50)
    }
    # 如果是15分钟，一天是4个小时，一小时是4个15分钟，一天是16个15分钟
    if freq == "m":
        scale = fh / (5 * 16)
    elif freq == "d":
        min = d_setting[fh][0]
        max = d_setting[fh][1]
    elif freq == "y":
        scale = fh * 10
    elif freq == "me":
        min = me_setting[fh][0]
        max = me_setting[fh][1]
    else:
        raise Exception("无法识别的频率freq,必须为'm'或者‘d'")

    if a < min:
        num = 0
    elif min <= a < max:
        num = 1
    elif a >= max:
        num = 2
    return num


def transform_date(origin, contain_split=True):
    '''
    将%Y-%m-%d转换成%Y%m%d
    '''
    if contain_split:
        return (datetime.strptime(str(origin), "%Y-%m-%d")).strftime('%Y%m%d')
    else:
        return (datetime.strptime(str(origin), "%Y%m%d")).strftime('%Y-%m-%d')


def check_date_format(conn, table):
    '''
    检查日期格式是否正确，是否为%Y%m%d的格式
    :param conn:
    :param table:
    :return:
    '''
    df_stock = pd.read_sql(
        '''SELECT * FROM  {} LIMIT 3;'''.format(table), con=conn)  # 把数据库的入口传给它 # 简单明了的 sql 语句
    trade_date = df_stock["trade_date"].iloc[1]
    if "-" in trade_date:
        return False
    else:
        return True


def get_one_stock_data_from_sqlite(conn, ts_code, start_date, end_date, table="stock_all"):
    '''
    从数据库获取指定日期间的股票信息
    :param df_index: 索引
    :param ts_code: 股票代码
    :param start_date: 开始时间格式%Y%m%d
    :param end_date: 结束时间格式%Y%m%d
    :param is_merge_index: 是否合并指数
    :param table: 从数据库查询那张表
    :return: 合并指数后的数据
    '''
    # 先获取一下股票信息，检查一下其日期格式，如果是%Y%m%d则不用转换，如果是
    if check_date_format(conn, table):
        df_stock = pd.read_sql(
            '''SELECT * FROM  {} where ts_code='{}' AND trade_date>='{}' AND trade_date<='{}';'''.format(table, ts_code,
                                                                                                         start_date,
                                                                                                         end_date),
            con=conn)  # 把数据库的入口传给它 # 简单明了的 sql 语句
    else:
        start_date = transform_date(start_date, False)
        end_date = transform_date(end_date, False)
        df_stock = pd.read_sql(
            '''SELECT * FROM  {} where ts_code='{}' AND trade_date>='{}' AND trade_date<='{}';'''.format(table, ts_code,
                                                                                                         start_date,
                                                                                                         end_date),
            con=conn)  # 把数据库的入口传给它 # 简单明了的 sql 语句
        df_stock["trade_date"] = df_stock["trade_date"].apply(
            lambda x: (datetime.strptime(x, "%Y-%m-%d")).strftime('%Y%m%d'))
    df_stock.dropna(inplace=True)

    return df_stock


def get_data_length(data_length, step_length, fh):
    '''
    此函数暂时不用
    根据数据长度，计算训练数据和测试数据的长度，
    :param data_length: 股票数据长度
    :param step_length: 历史回顾长度
    :param fh: 未来展望长度
    :return: 训练数据长度，测试数据长度
    '''
    total_length = data_length - fh - step_length
    valid_length = int(total_length * 0.2)
    train_length = total_length - valid_length
    return train_length, valid_length


def save_stock_to_db(conn, df_stock_basic,table_name):
    '''
    此函数暂时不用，
    此函数是将合并指数后，计算完指标后的股票数据临时保存到数据库中，不用每次都计算一次股票指标
    :param df_index:
    :param df_stock_basic:
    :return:
    '''
    start = False
    for key, ts_code in enumerate(list(df_stock_basic["ts_code"])):
        print(ts_code)
        # if ts_code == "001379.SZ":
        #     start = True
        if start:
            df = get_one_stock_data_from_sqlite(conn, ts_code, start_date, end_date,table_name)
            if len(df) < 20:
                continue
            # 股票按照交易日降序排列
            df = df.sort_values(by="trade_date", ascending=True, ignore_index=True)
            # 添加股票技术指标
            df = indicatior(df)
            df.dropna(inplace=True)
            # print(df)
            df.to_sql("stock_daily", conn, if_exists="append", index=False)


def compute_data_y(df, df_index, fh, freq, clean):
    '''
    计算股票指标
    如果是日线则合并指数信息
    如果是分钟线不合并指数信息
    :param df:股票信息
    :param df_index:指数信息
    :param fh:未来几天
    :param freq:日线d,分钟线m
    :return:处理后的数据
    '''
    if (freq == "d") or (freq == "me"):
        df = df.sort_values(by="trade_date", ascending=True, ignore_index=True)
        df["trade_date"] = df["trade_date"].astype(int)
        if clean:
            df = df.drop(labels=["ts_code", "trade_date"], axis=1)
        else:
            df = indicatior(df)
            df = df.merge(df_index, how="left", suffixes=["_stock", "_index"], on="trade_date")
            df = df.drop(labels=["ts_code", "trade_date", "ts_code_sh_index", "ts_code_sz_index"], axis=1)
    elif freq == "m":
        df = df.sort_values(by=["trade_date", "trade_time"], ascending=True, ignore_index=True)
        if clean:
            df = df.drop(labels=["ts_code", "trade_date", "trade_time", "adj"], axis=1)
        else:
            df = indicatior(df)
            df = df.drop(labels=["ts_code", "trade_date", "trade_time", "adj"], axis=1)
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
    df["target"] = df["next_n_pct_chg"].apply(categorize, args=(fh, freq))
    # 将分类结果复制到df_data
    df_data["target"] = df["target"]
    # 将未来几天收盘价和涨幅从预测数据中删除
    df_data = df_data.drop(labels=["next_n_close", "next_n_pct_chg"], axis=1)
    return df_data


def transform_data(df, window_length, get_x, get_y, horizon=1, stride=1, start=0, seq_first=True):
    '''
    将Dataframe的时序数据，转换为（samples,varilbes,steps)类型的数据，拆分成训练集和测试集，同时返回tsai的dataloader
    :param df:要处理的数据，DataFrame格式
    :param window_length:window_length is usually selected based on prior domain knowledge or by trial and error
    :param get_x:Indicates which are the columns that contain the x data.
    :param get_y:In multivariate time series, you must indicate which is/are the y columns
    :param horizon:0 means y is taken from the last time stamp of the time sequence (default = 1)
    :param stride:None for non-overlapping (stride = window_length) (default = 1). This depends on how often you want to predict once the model
    :param start:use all data since the first time stamp (default = 0)
    :return:TSDataloader，X_train,y_train,X_valid,y_valid
    '''

    X, y = SlidingWindow(window_length, stride=stride, start=start, get_x=get_x, get_y=get_y, horizon=horizon,
                         seq_first=seq_first)(df)
    if (y is None) or (len(y) < 20):
        return None, None, None, None, None
    splits = get_splits(y, valid_size=.2, stratify=True, random_state=23, shuffle=False, show_plot=False)
    X_train = X[splits[0]]
    y_train = y[splits[0]]
    X_valid = X[splits[1]]
    y_valid = y[splits[1]]

    # tfms = [None, [Categorize()]]
    # dsets = TSDatasets(X, y, tfms=tfms, splits=splits)
    # batch_tfms = TSStandardize()
    # dls = get_ts_dls(X, y, splits=splits, tfms=tfms, batch_tfms=batch_tfms, bs=[64, 128])
    dls = ""
    return dls, X_train, y_train, X_valid, y_valid


def split_data(df, step_length, get_x=None):
    '''
    将股票数据分类为训练数据和测试数据
    :param df: 处理过之后的股票信息
    :param step_length: 过去的长度
    :return:
    '''
    if get_x is None:
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

    def __init__(self, conn, start_date, end_date, one_hot=False, train=True, keras_data=False, step_length=100, fh=15,
                 freq="d", clean=False, table="stock_all"):
        '''

        :param conn: 数据库连接
        :param one_hot: 是否将y转化为one_hot形式
        :param train: 是否取训练集
        :param keras_data: 是否转化为keras形式的数据，即n_samples,n_steps,n_vars
        :param step_length: 过去多少时间
        :param fh: 需要预测的未来多少时间
        :param freq: 频率，d:代表是日线数据，m:代表是分钟数据me代表月度数据
        :param clean: if ture 不包含index和指标数据，else 包含index和指标数据
        :param table: 数据库表名
        '''
        self.conn = conn
        self.start_date = start_date
        self.end_date = end_date
        self.one_hot = one_hot
        self.train = train
        self.keras_data = keras_data
        self.step_length = step_length
        self.fh = fh
        self.freq = freq
        self.table = table
        self.clean = clean
        self.title = "clean" if clean else "all"
        # 获取指数数据
        self.df_index = None if clean else pd.read_sql("select * from df_index;", conn)
        # 获取股票基本信息
        self.df_stock_basic = None if clean else pd.read_sql("select * from stock_basic;", conn)

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

        df_ts_codes = pd.read_sql("select ts_code from {}".format(self.table), self.conn)

        for key, ts_code in enumerate(np.unique(df_ts_codes["ts_code"])):
            # 遍历所有的股票代码
            print(ts_code)
            # 获取股票日线或者分钟线数据
            df = get_one_stock_data_from_sqlite(self.conn, ts_code, self.start_date, self.end_date, self.table)
            # 如果在开始时间和截止时间内，该股票的数据小于历史回顾和未来预测期数据，那么该股票将没有可测试的数据，因此不测试该股票
            if len(df) < self.step_length + self.fh + 1:
                continue
            # 计算股票指标并合并指数
            df_data = compute_data_y(df, self.df_index, self.fh, self.freq, self.clean)
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
        df_record.to_sql("step_length_{}_fh_{}_{}".format(self.step_length, self.fh, self.title), self.conn,
                         index=False,
                         if_exists="replace")

    def get_records(self):
        '''
        获取数据记录,如果没有的话，重新计算
        :return:
        '''
        try:
            df = pd.read_sql("select * from step_length_{}_fh_{}_{}".format(self.step_length, self.fh, self.title),
                             self.conn)
            if len(df) < 2:
                self.get_length()
                df = pd.read_sql("select * from step_length_{}_fh_{}_{}".format(self.step_length, self.fh, self.title),
                                 self.conn)
        except pd.errors.DatabaseError:
            self.get_length()
            df = pd.read_sql("select * from step_length_{}_fh_{}_{}".format(self.step_length, self.fh, self.title),
                             self.conn)
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
        df = get_one_stock_data_from_sqlite(self.conn, ts_code, self.start_date, self.end_date, self.table)
        df_data = compute_data_y(df, self.df_index, self.fh, self.freq, self.clean)
        X_train, y_train, X_valid, y_valid = split_data(df_data, self.step_length)
        # 获取股票对应位置的数据
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


def new_fit(settings, load_model=False, batch_size=64, dataset=StockPytorchDatasetNew):
    '''
    根据配置信息进行训练
    :param settings: 配置信息
    :return:
    '''
    import keras
    from torch.utils.data import DataLoader
    from sktime.classification.deep_learning.inceptiontime import InceptionTimeClassifier
    from sktime.classification.deep_learning.lstmfcn import LSTMFCNClassifier
    db = settings["db"]
    table = settings["table"]
    conn = sqlite3.connect(db, check_same_thread=False)
    start_date = settings["start_date"]
    end_date = settings["end_date"]
    step_length = settings["step_length"]
    fh = settings["fh"]
    freq = settings["freq"]
    n_vars = settings["n_vars"]
    depth = settings["depth"]
    clean = settings["clean"]
    if clean:
        title = "clean"
    else:
        title = "all"
    if settings["model"] == "lstmfcn":
        network = LSTMFCNClassifier(verbose=True, n_epochs=100,batch_size=batch_size)
        model_save_path = r"D:\redhand\clean\data\state_dict\lstmfcn_{}_{}_{}_{}.keras".format(title, freq,
                                                                                                       step_length, fh)
    elif settings["model"] == "inceptiontime":
        # 从sktime创建inceptiontime模型
        network = InceptionTimeClassifier(verbose=True, depth=depth)
        model_save_path = r"D:\redhand\clean\data\state_dict\inceptiontime_{}_{}_{}_{}.keras".format(title, freq,
                                                                                                     step_length, fh)
    else:
        return None
    # 创建pytorch 数据集，y onehot为True

    train_data = dataset(conn, start_date, end_date, True, True, True, step_length, fh, freq, clean, table)
    train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=False)
    test_data = dataset(conn, start_date, end_date, True, False, True, step_length, fh, freq, clean, table)
    test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=False)


    # 模型初始化，
    model_ = network.build_model((step_length, n_vars), len(CATEGORIES))
    # 模型参数初始化
    print(model_save_path)
    # model_save_path = r"/clean/data/state_dict/inceptiontime_new_150_15_bak.keras"
    if load_model:
        model_ = keras.saving.load_model(model_save_path)
    # 开始训练
    csv_logger = keras.callbacks.CSVLogger(
        r"D:\redhand\clean\data\log\inceptiontime_log_{}_{}_{}_{}.csv".format(title, freq, step_length, fh),
        separator=",",
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
                                    reduce_lr, remot])
    print(history.history)
    # 保存参数
    # loaded_model = keras.saving.load_model(r"D:\redhand\project\data\state_dict\inceptiontime.keras")


if __name__ == '__main__':
    settings_month_12_1 = {
        "start_date": "20000101",
        "end_date": "20231231",
        "db": r"D:\stock\stock.db",
        "table": "stock_monthly",
        "step_length": 12,
        "fh": 1,
        "freq": "me",  # d代表day,m代表minute,me代表month
        "n_vars": 7,  # 96/9
        "depth": 6,  # 6/9
        "clean": True,
        "model":"lstmfcn",#lstmfcn,inceptiontime
    }
    settings_month_15_1 = {
        "start_date": "20000101",
        "end_date": "20231231",
        "db": r"D:\stock\stock.db",
        "table": "stock_monthly",
        "step_length": 15,
        "fh": 1,
        "freq": "me",  # d代表day,m代表minute,me代表month
        "n_vars": 7,  # 96/9
        "depth": 6,  # 6/9
        "clean": True,
        "model": "lstmfcn",  # lstmfcn,inceptiontime
    }
    settings_day_20_3 = {
        "start_date": "20000101",
        "end_date": "20231231",
        "db": r"D:\redhand\clean\data\tushare_db\tushare.db",
        "table": "stock_all",
        "step_length": 20,
        "fh": 3,
        "freq": "d",  # d代表day,m代表minute
        "n_vars": 9,  # 96/9
        "depth": 9,  # 6/9
        "clean": True,
        "model": "lstmfcn",  # lstmfcn,inceptiontime
    }
    settings_day_30_3 = {
        "start_date": "20000101",
        "end_date": "20231231",
        "db": r"D:\redhand\clean\data\tushare_db\tushare.db",
        "table": "stock_all",
        "step_length": 30,
        "fh": 3,
        "freq": "d",  # d代表day,m代表minute
        "n_vars": 9,  # 96/9
        "depth": 9,  # 6/9
        "clean": True,
        "model": "inceptiontime",  # lstmfcn,inceptiontime
    }
    settings_day_100_15 = {
        "start_date": "20000101",
        "end_date": "20231231",
        "db": r"D:\redhand\clean\data\tushare_db\tushare.db",
        "table": "stock_all",
        "step_length": 100,
        "fh": 15,
        "freq": "d",  # d代表day,m代表minute
        "n_vars": 9,  # 96/9
        "depth": 9,  # 6/9
        "clean": True,
        "model": "lstmfcn",  # lstmfcn,inceptiontime
    }
    settings_day_150_15 = {
        "start_date": "20000101",
        "end_date": "20231231",
        "db": r"D:\redhand\clean\data\tushare_db\tushare.db",
        "table": "stock_all",
        "step_length": 150,
        "fh": 15,
        "freq": "d",  # d代表day,m代表minute
        "n_vars": 9,  # 96/9
        "depth": 9,  # 6/9
        "clean": True,
        "model": "lstmfcn",  # lstmfcn,inceptiontime
    }
    settings_day_200_20 = {
        "start_date": "20000101",
        "end_date": "20231231",
        "db": r"D:\redhand\clean\data\tushare_db\tushare.db",
        "table": "stock_all",
        "step_length": 200,
        "fh": 20,
        "freq": "d",  # d代表day,m代表minute
        "n_vars": 9,  # 96/9
        "depth": 9,  # 6/9
        "clean": True,
        "model": "lstmfcn",  # lstmfcn,inceptiontime
    }
    settings_minute_320_80 = {
        "start_date": "20040101",
        "end_date": "20231231",
        "db": r"D:\redhand\clean\data\tushare_db\stock_minute.db",
        "table": "stock_all",
        "step_length": 320,  # 16*20
        "fh": 80,  # 16*5
        "freq": "m",  # d代表day,m代表minute
        "n_vars": 67,
        "depth": 9,  # 6/9
        "clean": True,
        "model": "lstmfcn",  # lstmfcn,inceptiontime
    }
    # new_fit(settings_day_20_3,True)
    new_fit(settings_month_12_1,True)
    # conn = sqlite3.connect(r"D:\redhand\clean\data\tushare_db\tushare.db", check_same_thread=False)
    # get_stock_all_distribution(conn)
    # # save_stock_to_db()
    # from torch.utils.data import DataLoader
    #
    # conn_sqlite = sqlite3.connect(r"D:\redhand\clean\data\tushare_db\tushare.db")
    # df_index = pd.read_sql("select * from df_index;", conn_sqlite)
    # delete_table(conn_sqlite, "step_length_320_fh_80")
    # conn, start_date, end_date, one_hot=False, train=True, keras_data=False, step_length=100, fh=15,
    #                  freq="d"
    # train_data = StockPytorchDatasetNew(conn_sqlite,"20040101","20231231",True, True, True, 100, 10,"d",True)
    # train_dataloader = DataLoader(train_data, batch_size=64, shuffle=False)
    # test_data = StockPytorchDatasetNew(conn_sqlite,"20040101","20231231",True, False, True, 320, 80,"m")
    # test_dataloader = DataLoader(test_data, batch_size=64, shuffle=False)
    # df = get_one_stock_data_from_sqlite(conn_sqlite,"000001.SZ","20040101","20231231")
    # df_data = compute_data_y(df,df_index, 80,"m")
    # X_train, y_train, X_valid, y_valid = split_data(df_data, 320)
    # X_train = X_train.transpose((0, 2, 1))
    #
    # for key, (x_train, y_train) in enumerate(train_dataloader):
    #     test = X_train[64*key:64*(key+1)]
    #     print(key)
    #     print(x_train.shape)
    #     print("------------------------------------")
    #     if x_train == test:
    #         print("good")
    #     else:
    #         print("bad")
    #     if key>40:
    #         break
    #
    # df = pd.read_sql("select * from step_length_100_fh_15",conn)
    # print(df)
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
    # loaded_model = keras.saving.load_model(r"D:\redhand\clean\data\state_dict\inceptiontime_new_150_15_bak.keras")
    # rng = check_random_state(None)
    # for x,y in train_dataloader:
    #     print(loaded_model.evaluate(x,y))
    #     probs = loaded_model.predict(x)
    #     probs = probs / probs.sum(axis=1, keepdims=1)
    #     kinds = [0, 1, 2, 3, 4, 5, 6, 7, 8]
    #     res = np.array(
    #         [
    #             kinds[int(rng.choice(np.flatnonzero(prob == prob.max())))]
    #             for prob in probs
    #         ]
    #     )
    #     print(res,y)
