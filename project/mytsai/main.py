import numpy as np
from datetime import datetime
import sqlite3

from tsai.imports import *
from tsai.utils import *
from tsai.data.validation import *
from tsai.data.core import *
from project.MYTT.apply_mytt import indicatior

def transform_date(origin, contain_split=True):
    '''
    将%Y-%m-%d转换成%Y%m%d
    '''
    if contain_split:
        return (datetime.strptime(str(origin), "%Y-%m-%d")).strftime('%Y%m%d')
    else:
        return (datetime.strptime(str(origin), "%Y%m%d")).strftime('%Y-%m-%d')


def get_one_stock_data_from_sqlite(ts_code, start_date, end_date, is_merge_index):
    start_date = transform_date(start_date, False)
    end_date = transform_date(end_date, False)
    df_stock = pd.read_sql(
        '''SELECT * FROM  stock_all where ts_code='{}' AND trade_date>='{}' AND trade_date<='{}';'''.format(ts_code,
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

if __name__ == '__main__':
    conn = sqlite3.connect(r"D:\redhand\clean\data\tushare_db\tushare.db")
    cursor = conn.cursor()
    is_real = False
    df_stock_basic = pd.read_sql("select * from stock_basic;",conn)
    df_index = pd.read_sql("select * from df_index;", conn)
    start_date = "20200101"
    end_date = "20231231"
    is_merge_index = False

    for key, ts_code in enumerate(list(df_stock_basic["ts_code"])):
        # print("开始处理{}:{}".format(key,ts_code))
        # 测试使用，实际使用需要注释掉
        # TODO:区分测试环境和正式环境
        if not is_real:
            if key > 5:
                break
        # print("正在处理{}".format(ts_code))
        # 获取股票和指数交易日数据，截止日前一年的数据
        # df_stock = get_one_stock_data(ts_code, df_index, start_date, end_date, is_merge_index)
        df_stock = get_one_stock_data_from_sqlite(ts_code, start_date, end_date, is_merge_index)
        print(df_stock)


# dsets = []
# for i in range(3):
#
#     TSUnwindowedDataset(x,y, window_size=100, stride=1, seq_first=True)
#     dsets.append(dset)
#
#
#
# metadataset = TSMetaDataset(dsets)
# splits = TimeSplitter(show_plot=False)(metadataset)
# metadatasets = TSMetaDatasets(metadataset, splits=splits)
# dls = TSDataLoaders.from_dsets(metadatasets.train, metadatasets.valid)