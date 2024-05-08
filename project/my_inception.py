import os
import sys
import pandas as pd
import time
import numpy as np
from pathlib import Path
sys.path.append(os.getcwd())

from sktime.classification.deep_learning.inceptiontime import InceptionTimeClassifier
from sktime.classification.deep_learning.lstmfcn import LSTMFCNClassifier
from sktime.classification.deep_learning.cnn import CNNClassifier
from sktime.classification.kernel_based import RocketClassifier
import tushare as ts
from prepare_data import categorize,transform_data,handle_stock_df
from MYTT.apply_mytt import indicatior
from get_data import get_one_stock_data,get_stock_basic,get_list_date

# 最少的上市天数，大于窗口
LIST_DAYS = 200
# 截止日期
END_DATE = 20231231

ts.set_token('f88e93f91c79cdb865f22f40cac23a2907da36b53fa9aa150228ed27')
pro = ts.pro_api()


# def handle_all_stock(clf,fh=1):
#     df_index = pd.read_csv("data/index.csv")
#     df_stock_basic = get_stock_basic(END_DATE, LIST_DAYS)
#     for key, ts_code in enumerate(list(df_stock_basic["ts_code"])):
#         print("开始处理第{}个，股票代码：{}".format(key, ts_code))
#         # print("开始时stat_dict:{}".format(clf.state_dict()))
#         list_date = get_list_date(ts_code, df_stock_basic)
#         handle_one_stock(clf, fh, ts_code, df_index, list_date, END_DATE, key)
#         print("处理完毕第{}个，股票代码：{}".format(key, ts_code))
#         # print("结束时stat_dict:{}".format(clf.state_dict()))


def handle_one_stock(clf,clf_name,fh,ts_codes,df_index,df_stock_basic,window_length,iter):
    # 第一步:准备数据
    prepare_start = time.time()
    X_train, y_train, X_valid, y_valid = merge_data(fh, ts_codes, df_index, df_stock_basic, window_length)
    print("训练数据X：{}，训练数据y:{},验证数据X:{},验证数据y:{}".format(X_train.shape, y_train.shape, X_valid.shape,
                                                                       y_valid.shape))
    print("准备数据完成,花费时间：{}".format(time.time() - prepare_start))
    # 第二步:训练模型
    train_start = time.time()
    print("第二步：开始训练模型")
    clf.fit(X_train, y_train)
    train_acc = clf.score(X_train,y_train)
    print("训练数据完成，花费时间:{}，训练精度：{}".format(time.time() - train_start,train_acc))
    clf.save("{}_{}".format(clf_name,iter))
    # 第三步：测试数据
    print("第三步：开始测试模型")
    test_start = time.time()
    test_acc = clf.score(X_valid,y_valid)
    print("第{}次训练精度为{}，测试精度为：{}".format(iter,train_acc,test_acc))
    print("模型测试完成，花费时间：{}".format(time.time() - test_start))


def prepare_data(fh,ts_code,df_index,df_stock_basic,window_length,is_merge_index):
    list_date = get_list_date(ts_code, df_stock_basic)
    print("正在处理{}".format(ts_code))
    # 获取股票和指数交易日数据，从上市日到截止日
    df_stock = get_one_stock_data(ts_code, df_index, list_date, END_DATE,is_merge_index)
    # 数据处理
    df_data = handle_stock_df(df_stock, fh,is_merge_index)
    columns = df_data.columns.tolist()
    # # 过去几期的数据
    # window_length = 100
    # X：所有数据都是X,包括股票的信息和指数的信息
    get_x = columns[:-1]
    # y的值:columns的索引：使用股票的pct_chg
    get_y = "target"
    _, X_train, y_train, X_valid, y_valid = transform_data(df_data, window_length, get_x, get_y)
    print("处理完成{}".format(ts_code))
    return X_train, y_train, X_valid, y_valid


def merge_data(fh,ts_codes,df_index,df_stock_basic,window_length):
    print("开始处理数据：{}".format(ts_codes))
    X_train, y_train, X_valid, y_valid = prepare_data(fh,ts_codes[0],df_index,df_stock_basic,window_length)
    for ts_code in ts_codes[1:]:
        X_train_tmp, y_train_tmp, X_valid_tmp, y_valid_tmp = prepare_data(fh, ts_code, df_index, df_stock_basic, window_length)
        X_train = np.concatenate((X_train,X_train_tmp),axis=0)
        y_train = np.concatenate((y_train,y_train_tmp),axis=0)
        X_valid = np.concatenate((X_valid,X_valid_tmp),axis=0)
        y_valid = np.concatenate((y_valid,y_valid_tmp),axis=0)
    print("汇总数据完成")
    return X_train, y_train, X_valid, y_valid


def handle_data(clf,clf_name,fh):
    df_index = pd.read_csv("data/index.csv")
    df_stock_basic = get_stock_basic(END_DATE, LIST_DAYS)
    ts_codes = []
    for key, ts_code in enumerate(list(df_stock_basic["ts_code"])):
        if key > 1000:
            ts_codes.append(ts_code)
        if (key % 20 == 0 and key != 0 and len(ts_codes)>0) :
            print("开始时stat_dict:{}".format(clf.get_params()))
            handle_one_stock(clf,clf_name,fh,ts_codes,df_index,df_stock_basic,100,key)
            print("结束时stat_dict:{}".format(clf.get_params()))
            ts_codes = []



if __name__ == '__main__':
    # X_train, y_train, X_valid, y_valid = get_merge_df(ts_codes)
    # print(X_train.shape)
    # print(y_train.shape)
    # print(X_valid.shape)
    # print(y_valid.shape)
    # first:0.1669873028087726  0.9960645037435208
    # first:0.1642028655901751  0.8824964539007092    #
    # network = LSTMFCNClassifier(n_epochs=65, verbose=0)
    # network = CNNClassifier(n_epochs=50, verbose=True)
    # network = RocketClassifier(num_kernels=10000,rocket_transform="multirocket",use_multivariate="yes")
    x_train = np.load(r"D:\redhand\clean\data\stocks\x_train_000001.SZ.npy")
    y_train = np.load(r"D:\redhand\clean\data\stocks\y_train_000001.SZ.npy")
    x_test = np.load(r"D:\redhand\clean\data\stocks\x_valid_000001.SZ.npy")
    y_test = np.load(r"D:\redhand\clean\data\stocks\y_valid_000001.SZ.npy")

    network = InceptionTimeClassifier(n_epochs=75, verbose=False)
    network.fit(x_train, y_train)
    network.score(x_test, y_test)
    network.save("{}_{}".format(network, 1))

    # network = network.load_from_path(Path(r"D:\redhand\project\inception_time_0.zip"))
    # handle_data(network,"inception_time", 15)






    # network.fit(X_train, y_train)
    # print(network.score(X_valid, y_valid))
    # print(network.score(X_train, y_train))
