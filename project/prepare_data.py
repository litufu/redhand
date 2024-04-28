import os
import sys
sys.path.append(os.getcwd())

from tsai.all import *
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from MYTT.apply_mytt import indicatior



def categorize(a):
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
    num = 0
    if a < -5.0:
        num = 0
    elif a >= -5.0 and a < -2.0:
        num = 0
    elif a >= -2.0 and a < -1.0:
        num = 2
    elif a >= -1.0 and a < 0.0:
        num = 3
    elif abs(a) < 1e-9:
        num = 4
    elif a > 0.0 and a <= 1.0:
        num = 5
    elif a > 1.0 and a <= 2.0:
        num = 6
    elif a > 2.0 and a <= 5.0:
        num = 7
    elif a > 5.0:
        num = 8

    return num


def handle_stock_df(df,fh,is_merge_index=True):
    '''
    1.按照交易日进行排序
    2.添加股票相关技术指标
    3.返回纯数据版本的股票交易数据，
    :param df:获取的股票和指数数据
    :param fh:未来几天的涨跌幅future_horizion
    :return:按照交易日排序后的纯股票+指数交易数据
    '''
    df = df.sort_values(by="trade_date", ascending=True)
    df = indicatior(df)
    if is_merge_index:
        df = df.drop(labels=["ts_code", "trade_date", "ts_code_sh_index", "ts_code_sz_index"], axis=1)
    else:
        df = df.drop(labels=["ts_code", "trade_date"], axis=1)
    df["next_n_close"] = df["close"].shift(-fh)
    df.dropna(inplace=True)
    df["next_n_pct_chg"] = ((df["next_n_close"] - df["close"])/df["close"])*100

    df_data = df.copy()
    df["target"] = df["next_n_pct_chg"].apply(categorize)
    # 进行数据归一化
    df_data = (df_data - df_data.mean()) / df_data.std()
    df_data["target"] = df["target"]

    return df_data


def transform_data(df, window_length, get_x, get_y, horizon=1, stride=1, start=0, seq_first=True):
    '''

    :param df:要处理的数据，DataFrame格式
    :param window_length:window_length is usually selected based on prior domain knowledge or by trial and error
    :param get_x:Indicates which are the columns that contain the x data.
    :param get_y:In multivariate time series, you must indicate which is/are the y columns
    :param horizon:0 means y is taken from the last time stamp of the time sequence (default = 1)
    :param stride:None for non-overlapping (stride = window_length) (default = 1). This depends on how often you want to predict once the model
    :param start:use all data since the first time stamp (default = 0)
    :return:TSDataloader，X_train,y_train,X_valid,y_valid
    '''
    # 数据归一化


    X, y = SlidingWindow(window_length, stride=stride, start=start, get_x=get_x, get_y=get_y, horizon=horizon,
                         seq_first=seq_first)(df)
    splits = get_splits(y, valid_size=.2, stratify=True, random_state=23, shuffle=False)
    X_train = X[splits[0]]
    y_train = y[splits[0]]
    X_valid = X[splits[1]]
    y_valid = y[splits[1]]

    tfms = [None, [Categorize()]]
    dsets = TSDatasets(X, y, tfms=tfms, splits=splits)
    batch_tfms = TSStandardize()
    dls = get_ts_dls(X, y, splits=splits, tfms=tfms, batch_tfms=batch_tfms, bs=[64, 128])
    return dls,X_train,y_train,X_valid,y_valid


if __name__ == '__main__':
    # 读取股票数据
    # stock_path =r"D:\redhand\project\tushare\merge.csv"
    # df = pd.read_csv(stock_path)
    # print(df.shape)
    # print(df[:10])
    # df_data = handle_stock_df(df,1)
    # columns = df_data.columns.tolist()
    # print(df_data.shape)
    # print(df_data[:10])
    #
    # # 过去几期的数据
    # window_length = 100
    # # 预测未来几期的数据
    # horizon = 1
    # # X：所有数据都是X,包括股票的信息和指数的信息
    # get_x = columns[:-1]
    # # y的值:columns的索引：使用股票的pct_chg
    # get_y = "target"
    # dls,X_train,y_train,X_valid,y_valid = transform_data(df_data, window_length, get_x, get_y)
    # print(X_train.shape,y_train.shape,X_valid.shape,y_valid.shape,type(X_train))
    # print(X_train[:10])

    # X, y = SlidingWindow(window_length, horizon=horizon, get_x=get_x, get_y=get_y)(df_data)
    # print(X.shape)
    # print(y.shape)
    # print(X[:10])
    # print(y[:10])

    # ds_name = 'OliveOil'
    # X, y, _ = get_UCR_data(ds_name, return_split=False)
    # print(X.shape)
    # print(X[:10])
    # X = X[:, 0]
    # print(X[:10])
    # print(y)

    # window_length = 5
    # n_vars = 3
    #
    # t = (torch.stack(n_vars * [torch.arange(10)]).T * tensor([1, 10, 100]))
    # df = pd.DataFrame(t, columns=[f'var_{i}' for i in range(n_vars)])
    # print('input shape:', df.shape)
    # display(df)
    # X, y = SlidingWindow(window_length)(df)
    # test_eq(X.shape, ((5, 3, 5)))
    # test_eq(y.shape, ((5, 3)))
    # print(X)
    # print(y)


    amount = pd.Series([100, 90, 110, 150, 110, 130, 80, 90, 100, 150])
    print()
