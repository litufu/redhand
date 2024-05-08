import os
import sys
sys.path.append(os.getcwd())

import numpy as np
from datetime import datetime
import sqlite3

# from tsai.imports import *
# from tsai.utils import *
# from tsai.data.validation import *
# from tsai.data.core import *
# from tsai.data.unwindowed import *
# from tsai.data.metadatasets import *
from tsai.all import *
from MYTT.apply_mytt import indicatior
from stock_dataset import StockPytorchDataset

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

if __name__ == '__main__':
    # conn = sqlite3.connect(r"D:\redhand\clean\data\tushare_db\tushare.db")
    # cursor = conn.cursor()
    # is_real = False
    # df_stock_basic = pd.read_sql("select * from stock_basic;",conn)
    # df_index = pd.read_sql("select * from df_index;", conn)
    # start_date = "20200101"
    # end_date = "20231231"
    # is_merge_index = False
    # fh = 15
    # dsets = []
    #
    # for key, ts_code in enumerate(list(df_stock_basic["ts_code"])):
    #     print("开始处理{}:{}".format(key,ts_code))
    #     # 测试使用，实际使用需要注释掉
    #     # TODO:区分测试环境和正式环境
    #     if not is_real:
    #         if key > 3:
    #             break
    #     # print("正在处理{}".format(ts_code))
    #     # 获取股票和指数交易日数据，截止日前一年的数据
    #     # df_stock = get_one_stock_data(ts_code, df_index, start_date, end_date, is_merge_index)
    #     df_stock = get_one_stock_data_from_sqlite(ts_code, start_date, end_date, is_merge_index)
    #     # print(df_stock)
    #     df_data = handle_stock_df(df_stock, fh, is_merge_index,)
    #     # print(df_data)
    #     # print(df_data.shape)
    #     x = df_data[df_data.columns[:-1]].to_numpy()
    #     # print(x)
    #     print(x.shape)
    #     y = df_data[df_data.columns[-1]].to_numpy()
    #     # oh_encoder = OneHot(9)
    #     # y_cat = ToNumpyCategory()(y)
    #     # y = oh_encoder(y_cat)
    #     print(y.shape)
    #     dset = TSUnwindowedDataset(x, y, window_size=100, stride=1, seq_first=True)
    #     print(dset[:])
    #     dataset = dset[:][0]
    #     print(dataset,dataset.data.shape)
    #     dsets.append(dset)
    #
    # metadataset = TSMetaDataset(dsets)
    # splits = TimeSplitter(show_plot=False)(metadataset)
    # metadatasets = TSMetaDatasets(metadataset, splits=splits)
    # print(metadatasets.train,metadatasets.train[:][0].shape,metadatasets.train[:][1].shape)
    # dls = TSDataLoaders.from_dsets(metadatasets.train, metadatasets.valid)
    # print(dls.vars)
    # print(dls.c)
    # print(dls.)
    record_file = r"D:\redhand\clean\data\stock_record.csv"
    train_data = StockPytorchDataset(record_file, True, True, True)
    train_dataloader = TSDataLoader(train_data, bs=64, shuffle=False, num_workers=0)
    test_data = StockPytorchDataset(record_file, True, False, True)
    test_dataloader = TSDataLoader(test_data, bs=64, shuffle=False, num_workers=0)
    model = InceptionTime(96, 9)
    learn = Learner(train_dataloader, model, metrics=accuracy)
    learn.save('stage0')
    # learn.lr_find()
    learn.fit_one_cycle(25, lr_max=1e-3)
    # learn.save('stage1')
