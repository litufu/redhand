import time
import numpy as np
import pandas as pd
import requests
import json
from sklearn.metrics import accuracy_score
from multiRocket.my_multirocket_multivariate import MultiRocket
from download_data import get_stock_basic,get_list_date,get_one_stock_data,handle_stock_df,transform_data
from constants import END_DATE,LIST_DAYS

def send_to_server(send):
    requests.post("http://121.41.21.130:9000/publish/epoch/end/",{"data": json.dumps(send)},)


def handle_all_stock(clf,log_file,fh=1):
    '''
    处理所有的股票，预测未来几天的模型
    :param fh: 未来几天
    :return:
    '''

    df_index = pd.read_csv("data/index.csv")
    df_stock_basic = get_stock_basic(END_DATE, LIST_DAYS)
    for key,ts_code in enumerate(list(df_stock_basic["ts_code"])):
        print("开始处理第{}个，股票代码：{}".format(key,ts_code))
        print("开始时stat_dict:{}".format(clf.state_dict()))
        list_date = get_list_date(ts_code, df_stock_basic)
        train_acc,test_acc = handle_one_stock(clf,fh, ts_code, df_index, list_date, END_DATE,key)
        status = {"epoch":key,"ts_code":ts_code,"train_acc":train_acc,"test_acc":test_acc}
        send_to_server(json.dumps(status))
        df = pd.DataFrame(status,index=[key])
        df.to_csv(log_file,index=False,mode="a")
        print("处理完毕第{}个，股票代码：{}".format(key, ts_code))
        print("结束时stat_dict:{}".format(clf.state_dict()))


def handle_one_stock(clf,fh,ts_code,df_index,start_date,end_date,key):

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
    # X：所有数据都是X,包括股票的信息和指数的信息
    get_x = columns[:-1]
    # y的值:columns的索引：使用股票的pct_chg
    get_y = "target"
    _, X_train, y_train, X_valid, y_valid = transform_data(df_data, window_length, get_x, get_y)
    print("训练数据X：{}，训练数据y:{},验证数据X:{},验证数据y:{}".format(X_train.shape, y_train.shape, X_valid.shape, y_valid.shape))
    print("准备数据完成,花费时间：{}".format(time.time() - prepare_start))
    # 第二步:训练模型
    train_start = time.time()
    print("第二步：开始训练模型")
    yhat_train = clf.fit(X_train, y_train,predict_on_train=True)
    train_acc = accuracy_score(y_train, yhat_train)
    print("训练数据完成，花费时间:{}".format(time.time() - train_start))
    clf.save(r"D:\redhand\clean\data\model\multirocket_{}.pth".format(key))
    # 第三步：测试数据
    print("第三步：开始测试模型")
    test_start = time.time()
    yhat_test = clf.predict(X_valid)
    test_acc = accuracy_score(y_valid, yhat_test)
    print("第{}次训练精度为{}，测试精度为：{}".format(key,train_acc,test_acc))
    print("模型测试完成，花费时间：{}".format(time.time() - test_start))
    return train_acc,test_acc


if __name__ == '__main__':
    clf = MultiRocket(
        num_features=49728,  # 必须是2*4*84的整数倍
        classes=np.array([0, 1, 2, 3, 4, 5, 6, 7, 8]),
        classifier="logistic",
        max_epochs=75,
    )
    clf = clf.load(r"D:\redhand\clean\data\model\multirocket_1.pth")
    log_file = r"D:\redhand\clean\data\log\multirocket_log.csv"
    handle_all_stock(clf,log_file, 15)