import time
import torch
import pandas as pd
import numpy as np
from get_data import get_one_stock_data,get_stock_basic,get_list_date
from prepare_data import handle_stock_df,transform_data
from sklearn.metrics import accuracy_score
from multiRocket.my_multirocket_multivariate import MultiRocket
# 最少的上市天数，大于窗口
LIST_DAYS = 200
# 截止日期
END_DATE = 20231231

def handle_all_stock(clf,fh=1):
    '''
    预测未来几天的模型
    :param fh: 未来几天
    :return:
    '''

    df_index = pd.read_csv("data/index.csv")
    df_stock_basic = get_stock_basic(END_DATE, LIST_DAYS)
    for key,ts_code in enumerate(list(df_stock_basic["ts_code"])):
        print("开始处理第{}个，股票代码：{}".format(key,ts_code))
        print("开始时stat_dict:{}".format(clf.state_dict()))
        list_date = get_list_date(ts_code, df_stock_basic)
        handle_one_stock(clf,fh, ts_code, df_index, list_date, END_DATE,key)
        print("处理完毕第{}个，股票代码：{}".format(key, ts_code))
        print("结束时stat_dict:{}".format(clf.state_dict()))


def handle_one_stock(clf,fh,ts_code,df_index,start_date,end_date,iter):
    # 第一步:准备数据
    prepare_start = time.time()
    print("第一步：开始准备数据")
    # 获取股票和指数交易日数据
    df_stock = get_one_stock_data(ts_code,df_index,start_date,end_date)
    # 数据处理
    df_data = handle_stock_df(df_stock,fh)
    columns = df_data.columns.tolist()
    # 过去几期的数据
    window_length = 100
    # 预测未来几期的数据
    horizon = 1
    # X：所有数据都是X,包括股票的信息和指数的信息
    get_x = columns[:-1]
    # y的值:columns的索引：使用股票的pct_chg
    get_y = "target"
    _, X_train, y_train, X_valid, y_valid = transform_data(df_data, window_length, get_x, get_y)
    print("训练数据X：{}，训练数据y:{},验证数据X:{},验证数据y:{}".format(X_train.shape, y_train.shape, X_valid.shape, y_valid.shape))
    print("准备数据完成,花费时间：{}".format(time.time() - prepare_start))
    # 第二步:训练模型
    train_start = time.time()
    train_acc = 0.0
    print("第二步：开始训练模型")
    yhat_train = clf.fit(X_train, y_train,predict_on_train=True)
    train_acc = accuracy_score(y_train, yhat_train)
    print("训练数据完成，花费时间:{}".format(time.time() - train_start))
    clf.save("model_{}.pth".format(iter))
    # 第三步：测试数据
    print("第三步：开始测试模型")
    test_start = time.time()
    yhat_test = clf.predict(X_valid)
    test_acc = accuracy_score(y_valid, yhat_test)
    print("第{}次训练精度为{}，测试精度为：{}".format(iter,train_acc,test_acc))
    print("模型测试完成，花费时间：{}".format(time.time() - test_start))

if __name__ == '__main__':
    clf = MultiRocket(
        num_features=49728,  # 必须是2*4*84的整数倍
        classes=np.array([0, 1, 2, 3, 4, 5, 6, 7, 8]),
        classifier="logistic",
        max_epochs=5,
    )
    clf = clf.load("model_0.pth")
    handle_all_stock(clf,10)
    # state_dict = torch.load("model_4.pth")
    # print(state_dict)
    # model = torch.nn.Sequential(torch.nn.Linear(49728, 9)).to(torch.device("cpu"))
    # model.load_state_dict(state_dict)
    # print(model)
    # print(type(model))
    # print(model.state_dict())

