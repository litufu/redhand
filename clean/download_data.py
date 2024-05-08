import os
import numpy as np
import pandas as pd
import tushare as ts
from tsai.all import *
from constants import pro, LIST_DAYS, END_DATE, WINDOW_LENGTH, TOTAL_TRAIN_DATA_LENGTH, TOTAL_TEST_DATA_LENGTH, \
    X_TRAIN_PATH, Y_TRAIN_PATH, X_VALID_PATH, Y_VALID_PATH,TRAIN_DATA_LENGTH,TEST_DATA_LENGTH

from MYTT.apply_mytt import indicatior
ts.set_token('f88e93f91c79cdb865f22f40cac23a2907da36b53fa9aa150228ed27')


def categorize(a,fh=5):
    '''
    将涨跌幅分类
    区间	类别
    <-5	0
    【-5,-2】	0
    【-2,-1】	2
    【-1,0】	3
    0	4
    【0,1】	5
    【1,2】	6
    【2,5】	7
    >5	8

    :param a: 浮点数，代表涨跌幅
    :return:
    '''
    scale = fh // 5
    num = 0
    if a < -5.0 * scale:
        num = 0
    elif -5.0 * scale <= a < -2.0 * scale:
        num = 0
    elif -2.0 * scale <= a < -1.0 * scale:
        num = 2
    elif -1.0 * scale <= a < 0.0 * scale:
        num = 3
    elif abs(a) < 1e-9:
        num = 4
    elif 0.0 < a <= 1.0 * scale:
        num = 5
    elif 1.0 * scale < a <= 2.0 * scale:
        num = 6
    elif 2.0 * scale < a <= 5.0 * scale:
        num = 7
    elif a > 5.0 * scale:
        num = 8

    return num

def get_df_index():
    df_sh = pro.index_daily(ts_code='000001.SH')
    df_sh["trade_date"] = df_sh["trade_date"].astype(int)
    df_sh = df_sh.sort_values(by="trade_date", ascending=True)
    df_sh["ma5"] = df_sh['close'].rolling(5).mean()
    df_sh["ma10"] = df_sh['close'].rolling(10).mean()
    df_sh["ma30"] = df_sh['close'].rolling(30).mean()
    df_sh["ma60"] = df_sh['close'].rolling(60).mean()
    df_sz = pro.index_daily(ts_code='399001.SZ')
    df_sz["trade_date"] = df_sz["trade_date"].astype(int)
    df_sz = df_sz.sort_values(by="trade_date", ascending=True)
    df_sz["ma5"] = df_sz['close'].rolling(5).mean()
    df_sz["ma10"] = df_sz['close'].rolling(10).mean()
    df_sz["ma30"] = df_sz['close'].rolling(30).mean()
    df_sz["ma60"] = df_sz['close'].rolling(60).mean()
    df_index = df_sh.merge(df_sz, how="left", suffixes=["_sh_index", "_sz_index"], on="trade_date")
    return df_index


def get_list_date(ts_code, df_stock_basic):
    '''
    获取股票的上市日期
    :param ts_code: 股票代码
    :param df_stock_basic: 股票基本信息
    :return:
    '''

    df_stock_basic = df_stock_basic[df_stock_basic["ts_code"] == ts_code]
    list_date = df_stock_basic["list_date"].values[0]
    return list_date


def get_stock_basic(end_date, list_days):
    '''
    获取截止日超过已上市天数的非北交所股票列表
    :param end_date: 截止日期
    :param list_days: 已上市天数
    :return:不包含北交所，上市截止日超过list_days的上市公司名单
    '''
    df_stock_basic = pro.stock_basic(exchange='', list_status='L',
                                     fields='ts_code,market,symbol,name,area,industry,list_date')
    df_stock_basic["list_duaration"] = end_date - df_stock_basic["list_date"].astype(int)
    # 上市超过LIST_DAYS
    df_stock_basic = df_stock_basic[df_stock_basic["list_duaration"] > list_days]
    # 不包含北交所，北交所有流动性问题
    df_stock_basic = df_stock_basic[df_stock_basic["market"] != "北交所"]
    return df_stock_basic


def get_one_stock_data(ts_code, df_index, start_date, end_date, is_merge_index=True):
    '''
    获取一只股票的数据，并且可以合并股票指数的数据，返还一个股票本身的数据或合并指数后的合并数据
    :param ts_code: 股票代码
    :param df_index: 交易数据
    :param start_date: 股票开始日期
    :param start_date: 股票结束日期
    :param end_date: 股票结束日期
    :param is_merge_index：是否合并指数
    :return:合并后的数据
    '''
    df_stock = ts.pro_bar(ts_code=ts_code, adj='qfq', start_date=str(start_date), end_date=str(end_date))
    df_stock.dropna(inplace=True)
    # df_stock = pro.daily(ts_code=ts_code, start_date=start_date, end_date=end_date)
    df_stock["trade_date"] = df_stock["trade_date"].astype(int)
    if is_merge_index:
        df_merge = df_stock.merge(df_index, how="left", suffixes=["_stock", "_index"], on="trade_date")
    else:
        df_merge = df_stock.copy(deep=True)
    return df_merge


def handle_stock_df(df, fh, is_merge_index=True,is_contain_y=True):
    '''
    1.按照交易日进行排序
    2.添加股票相关技术指标
    3.返回纯数据版本的股票交易数据，
    :param df:获取的股票和指数数据
    :param fh:未来几天的涨跌幅future_horizion
    :param is_merge_index:是否合并指数数据
    :return:按照交易日排序后的纯股票+指数交易数据
    '''
    # 股票按照交易日降序排列
    df = df.sort_values(by="trade_date", ascending=True,ignore_index=True)
    # 添加股票技术指标
    df = indicatior(df)
    # 股票按照交易日升序排列
    # df = df.sort_values(by="trade_date", ascending=False)
    # 删除无关的信息
    if is_merge_index:
        df = df.drop(labels=["ts_code", "trade_date", "ts_code_sh_index", "ts_code_sz_index"], axis=1)
    else:
        df = df.drop(labels=["ts_code", "trade_date"], axis=1)
    # 计算未来几天的股票变动
    if is_contain_y:
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
    else:
        df.dropna(inplace=True)
        df = (df - df.mean()) / df.std()
        return df


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
    if (y is None) or (len(y)<20):
        return None,None,None,None,None
    splits = get_splits(y, valid_size=.2, stratify=True, random_state=23, shuffle=False,show_plot=False)
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


def prepare_data(fh, ts_code, df_index, df_stock_basic, window_length, is_merge_index):
    '''

    :param fh: 要预测的未来天数
    :param ts_code: 股票代码
    :param df_index: 指数df
    :param df_stock_basic: 上市股票基本信息
    :param window_length: 过去几天的数据
    :param is_merge_index: 是否合并指数数据
    :return:
    '''
    # 获取上市公司的上市日期
    list_date = get_list_date(ts_code, df_stock_basic)
    print("正在处理{}".format(ts_code))
    # 获取股票和指数交易日数据，从上市日到截止日
    df_stock = get_one_stock_data(ts_code, df_index, list_date, END_DATE, is_merge_index)
    # 数据处理
    df_data = handle_stock_df(df_stock, fh, is_merge_index)
    columns = df_data.columns.tolist()
    # X：所有数据都是X,包括股票的信息和指数的信息
    get_x = columns[:-1]
    # y的值:columns的索引：使用股票的pct_chg
    get_y = "target"
    _, X_train, y_train, X_valid, y_valid = transform_data(df_data, window_length, get_x, get_y)
    print("处理完成{}".format(ts_code))
    return X_train, y_train, X_valid, y_valid


def get_files_by_suffix(dir, suffix):
    '''
    获取路径下面后缀的文件
    :param dir:
    :param suffix:
    :return:
    '''
    files_res = []
    file_paths_res = []
    for root, dirs, files in os.walk(dir):
        for file in files:
            if file.endswith(suffix):
                file_path = os.path.join(root, file)
                file_paths_res.append(file_path)
                files_res.append(file)

    return files_res, file_paths_res


def download_all_data(fh, dir, record_filename, is_merge_index=True):
    '''
    下载所有的数据并处理
    :param fh: 需要预测未来几天的行情
    :param dir: 存储的目录
    :param is_merge_index: 是否需要合并指数
    :return:
    '''
    # df_index = pd.read_csv(r"D:\redhand\clean\data\index.csv")
    df_index = get_df_index()
    df_stock_basic = get_stock_basic(END_DATE, LIST_DAYS)
    x_train_paths = []
    y_train_paths = []
    x_valid_paths = []
    y_valid_paths = []
    train_data_lengths = []
    test_data_lengths = []
    total_train_data_lengths = []
    total_test_data_lengths = []
    total_train_data_length = 0
    total_test_data_length = 0
    start = True
    for key, ts_code in enumerate(list(df_stock_basic["ts_code"])):
        # if key>3:
        #     break
        # if ts_code == "600221.SH":
        #     start = True
        # else:
        #     print("pass:{}".format(ts_code))
        if start:
            print("start_{}".format(ts_code))
            x_train, y_train, x_valid, y_valid = prepare_data(fh, ts_code, df_index, df_stock_basic, WINDOW_LENGTH,
                                                              is_merge_index)
            if x_train is None:
                continue
            train_data_length = len(y_train)
            total_train_data_length = total_train_data_length + train_data_length
            test_data_length = len(y_valid)
            total_test_data_length = total_test_data_length + test_data_length
            x_train_path = os.path.join(dir, 'x_train_{}.npy').format(ts_code)
            y_train_path = os.path.join(dir, 'y_train_{}.npy').format(ts_code)
            x_valid_path = os.path.join(dir, 'x_valid_{}.npy').format(ts_code)
            y_valid_path = os.path.join(dir, 'y_valid_{}.npy').format(ts_code)
            np.save(x_train_path, x_train)
            np.save(y_train_path, y_train)
            np.save(x_valid_path, x_valid)
            np.save(y_valid_path, y_valid)
            # x_train_paths.append(x_train_path)
            # y_train_paths.append(y_train_path)
            # x_valid_paths.append(x_valid_path)
            # y_valid_paths.append(y_valid_path)
            # train_data_lengths.append(train_data_length)
            # test_data_lengths.append(test_data_length)
            # total_train_data_lengths.append(total_train_data_length)
            # total_test_data_lengths.append(total_test_data_length)

    # dic = {
    #     X_TRAIN_PATH: x_train_paths,
    #     Y_TRAIN_PATH: y_train_paths,
    #     X_VALID_PATH: x_valid_paths,
    #     Y_VALID_PATH: y_valid_paths,
    #     TRAIN_DATA_LENGTH: train_data_lengths,
    #     TEST_DATA_LENGTH: test_data_lengths,
    #     TOTAL_TRAIN_DATA_LENGTH: total_train_data_lengths,
    #     TOTAL_TEST_DATA_LENGTH: total_test_data_lengths,
    # }
    # df = pd.DataFrame(dic)
    # df.to_csv(record_filename, index=False)


def generate_record_from_dir(path, record_filename):
    '''
    根据已经下载的文件夹中的npy文件，生成一张记录表，记录每个npy文件的长度:
    generate_record_from_dir("D:\redhand\project\data",r"D:\redhand\project\data\stock_record1.csv")
    :param path: 包含npy的文件
    :param record_filename: 生成的记录文件名，全路径
    :return:
    '''

    x_train_paths = []
    y_train_paths = []
    x_valid_paths = []
    y_valid_paths = []
    train_data_lengths = []
    test_data_lengths = []
    total_train_data_lengths = []
    total_test_data_lengths = []
    total_train_data_length = 0
    total_test_data_length = 0
    files, filepaths = get_files_by_suffix(path, ".npy")
    for key, filepath in enumerate(filepaths):

        if "x_train" in filepath:
            x_train_paths.append(filepath)
        elif "x_valid" in filepath:
            x_valid_paths.append(filepath)
        elif "y_train" in filepath:
            y_train_data = np.load(filepath)
            train_data_length = len(y_train_data)
            total_train_data_length += train_data_length
            train_data_lengths.append(train_data_length)
            total_train_data_lengths.append(total_train_data_length)
            y_train_paths.append(filepath)
        elif "y_valid" in filepath:
            y_valid_data = np.load(filepath)
            valid_data_length = len(y_valid_data)
            total_test_data_length += valid_data_length
            test_data_lengths.append(valid_data_length)
            total_test_data_lengths.append(total_test_data_length)
            y_valid_paths.append(filepath)
        else:
            pass

    dic = {
        X_TRAIN_PATH: x_train_paths,
        Y_TRAIN_PATH: y_train_paths,
        X_VALID_PATH: x_valid_paths,
        Y_VALID_PATH: y_valid_paths,
        TRAIN_DATA_LENGTH: train_data_lengths,
        TEST_DATA_LENGTH: test_data_lengths,
        TOTAL_TRAIN_DATA_LENGTH: total_train_data_lengths,
        TOTAL_TEST_DATA_LENGTH: total_test_data_lengths,
    }

    df = pd.DataFrame(dic)
    df.to_csv(record_filename, index=False)


if __name__ == '__main__':
    # download_all_data(15, r"D:\redhand\clean\data\stocks", r"D:\redhand\clean\data\stock_record.csv")
    #
    #
    # generate_record_from_dir(r"D:\redhand\clean\data\stocks",r"D:\redhand\clean\data\stock_record.csv")
    # d1 = np.load(r"D:\redhand\clean\data\stocks\x_train_000001.SZ.npy")
    # d2 = np.load(r"D:\redhand\clean\data\stocks\y_train_000001.SZ.npy")
    d3 = np.load(r"D:\redhand\clean\data\stocks\y_valid_000001.SZ.npy")
    # 使用 NumPy 的 unique 函数统计元素出现次数
    unique_elements, counts = np.unique(d3, return_counts=True)
    # 将结果组合成字典
    numpy_result = dict(zip(unique_elements, counts))
    print("本次统计数据：{}".format(numpy_result))

    # d4 = np.load(r"D:\redhand\clean\data\stocks\x_valid_000001.SZ.npy")
    # print(d1.shape)
    # print(d2.shape)
    # print(d3.shape)
    # print(d4.shape)