from tsai.all import *
import numpy as np
import pandas as pd


def categorize(a):
    '''
    将涨跌幅分类
    区间	类别
    <-5	1
    【-5,-2】	2
    【-2,-1】	3
    【-1,0】	4
    0	5
    【0,1】	6
    【1,2】	7
    【2,5】	8
    >5	9

    :param a: 浮点数，代表涨跌幅
    :return:
    '''
    num = 0
    if a < -5.0:
        num = 1
    elif a >= -5.0 and a < -2.0:
        num = 2
    elif a >= -2.0 and a < -1.0:
        num = 3
    elif a >= -1.0 and a < 0.0:
        num = 4
    elif abs(a) < 1e-9:
        num = 5
    elif a > 0.0 and a <= 1.0:
        num = 6
    elif a > 1.0 and a <= 2.0:
        num = 7
    elif a > 2.0 and a <= 5.0:
        num = 8
    elif a > 5.0:
        num = 9

    return num


def get_df_data(stock_path):
    '''
    返回纯数据版本的股票交易数据，按照交易日进行排序
    :param stock_path:股票交易数据存放的地址
    :return:按照交易日排序后的纯股票+指数交易数据
    '''
    df = pd.read_csv(stock_path)
    df = df.sort_values(by="trade_date", ascending=True)
    df["target"] = df["pct_chg"].apply(categorize)
    df_data = df.drop(labels=["ts_code", "trade_date", "ts_code_sh_index", "ts_code_sz_index"], axis=1)
    return df_data


def handle_data(df, window_length, get_x, get_y, horizon=1, stride=1, start=0, seq_first=True):
    '''

    :param df:要处理的数据，DataFrame格式
    :param window_length:window_length is usually selected based on prior domain knowledge or by trial and error
    :param get_x:Indicates which are the columns that contain the x data.
    :param get_y:In multivariate time series, you must indicate which is/are the y columns
    :param horizon:0 means y is taken from the last time stamp of the time sequence (default = 1)
    :param stride:None for non-overlapping (stride = window_length) (default = 1). This depends on how often you want to predict once the model
    :param start:use all data since the first time stamp (default = 0)
    :return:TSDataloader
    '''

    X, y = SlidingWindow(window_length, stride=stride, start=start, get_x=get_x, get_y=get_y, horizon=horizon,
                         seq_first=seq_first)(df)
    splits = get_splits(y, valid_size=.2, stratify=True, random_state=23, shuffle=False)
    tfms = [None, [Categorize()]]
    dsets = TSDatasets(X, y, tfms=tfms, splits=splits)
    batch_tfms = TSStandardize()
    dls = get_ts_dls(X, y, splits=splits, tfms=tfms, batch_tfms=batch_tfms, bs=[64, 128])
    return dls


if __name__ == '__main__':
    # 读取股票数据
    stock_path =r"D:\redhand\project\tushare\merge.csv"
    df_data = get_df_data(stock_path)
    columns = df_data.columns.tolist()
    print(df_data.shape)
    print(df_data)

    # 过去几期的数据
    window_length = 100
    # 预测未来几期的数据
    horizon = 1
    # X：所有数据都是X,包括股票的信息和指数的信息
    get_x = columns[:-1]
    # y的值:columns的索引：使用股票的pct_chg
    get_y = "target"
    dls = handle_data(df_data, window_length, get_x, get_y)
    print(dls.dataset)


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
