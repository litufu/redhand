import keras
import numpy as np
import datetime
import time
import math
import logging
from copy import deepcopy
import sqlite3
import pandas as pd
from sktime.classification.deep_learning import InceptionTimeClassifier
from pathlib import Path
from sklearn.utils import check_random_state
from constants import pro, LIST_DAYS
from download_data import get_df_index, get_stock_basic, get_one_stock_data, handle_stock_df


logging.basicConfig(filename=r"D:\redhand\clean\data\log\test.log", filemode="w", format="%(asctime)s %(name)s:%(levelname)s:%(message)s",
                    datefmt="%d-%m-%Y %H:%M:%S", level=logging.DEBUG)
conn = sqlite3.connect(r"D:\redhand\clean\data\tushare_db\tushare.db")
# 获取交易日期
sql_query = '''SELECT * FROM trade_cal where cal_date>='{}' and cal_date<='{}' '''.format("20100101", "20241231")
df_trade_date = pd.read_sql(sql_query, conn)
df_trade_date["is_open"] = df_trade_date["is_open"].astype(int)
df_trade_date["cal_date"] = df_trade_date["cal_date"].astype(str)
df_trade_date["exchange"] = df_trade_date["exchange"].astype(str)
df_trade_date.sort_values(by="cal_date",ascending=True,inplace=True)


def get_trade_cal():
    '''
    获取交易所日历
    :return:
    '''
    df = pro.trade_cal(exchange='')
    df.to_sql("trade_cal", conn, if_exists="replace", index=False)


def get_stock_basic_to_sqlite(end_date,list_days):
    df_stock_basic = pro.stock_basic(exchange='', list_status='L',
                                     fields='ts_code,market,symbol,name,area,industry,list_date')
    df_stock_basic["list_duaration"] = end_date - df_stock_basic["list_date"].astype(int)
    # 上市超过LIST_DAYS
    df_stock_basic = df_stock_basic[df_stock_basic["list_duaration"] > list_days]
    # 不包含北交所，北交所有流动性问题
    df_stock_basic = df_stock_basic[df_stock_basic["market"] != "北交所"]
    df_stock_basic.to_sql("stock_basic", conn, if_exists="replace", index=False)


def get_index_to_sqlite():
    df_index = get_df_index()
    df_index.to_sql("df_index",conn,if_exists='replace',index=False)

def get_index_from_sqlite():
    df_index = pd.read_sql("select * from df_index;", conn)
    # df_index = pd.read_sql("select * from index;",conn)
    return df_index


df_index = get_index_from_sqlite()


def predict(kinds, model, x):
    '''
    使用模型批量预测x对应的y值
    :param kinds: 所要预测的y的所有类别
    :param model: 训练好的模型
    :param x: 要预测的值
    :return: 预测的y
    '''
    probs = predict_proba(model, x)
    rng = check_random_state(None)
    return np.array(
        [
            kinds[int(rng.choice(np.flatnonzero(prob == prob.max())))]
            for prob in probs
        ]
    )


def predict_proba(model, X, batch_size=32):
    '''
    使用训练好的模型批量预测x所对应的概率
    :param model:
    :param X:
    :param batch_size:
    :return:
    '''
    # X = X.transpose((0, 2, 1))
    probs = model.predict(X, batch_size)
    probs = probs / probs.sum(axis=1, keepdims=1)
    return probs


def is_trade_date(date):
    '''
    判断一个日期是否为交易日
    :param date:
    :return:
    '''
    is_open = df_trade_date[df_trade_date["cal_date"] == "SSE" & df_trade_date["cal_date"] == date]["is_open"].values[0]
    if is_open == 0:
        return False
    else:
        return True


def get_date_before_days(current_date_str, before_days):
    '''
    获取当期日期前或后的交易日期
    :param current_date_str: 当前日期字符串 "%Y%m%d"
    :param before_days: 前几天用正数，后几天用负数
    :return: 对应日期字符串
    '''

    current_date_str = str(current_date_str) if type(current_date_str) != str else current_date_str
    # current_date = datetime.datetime.strptime(current_date_str, "%Y%m%d")
    trade_dates = df_trade_date[(df_trade_date["is_open"] == 1) & (df_trade_date["exchange"] == "SSE")]["cal_date"].values
    trade_dates = trade_dates.tolist()
    index = trade_dates.index(current_date_str)
    find_index = index - before_days
    if find_index < 0:
        raise Exception("无法找到之前的日期")
    elif find_index > len(trade_dates):
        raise Exception("超过 最大日期")
    return trade_dates[find_index]


def holds(active, now_holds):
    '''
    根据传入的买入、卖出、入金、出金生成最新的持仓情况。
    :param active:
    {"method":"buy","ts_code":stock_code,",num:"num",price:"price}
    {"method":"sell","ts_code":stock_code,",num:"num",price:"price}
    {"method":"in","amount":amount,}
    {"method":"out","amount":amount}
    :return: holds = {"stocks":[{"ts_code":stockcode,"num":num},{"ts_code":stockcode,"num":num}],money:amout,date:"date"}
    '''
    # 交易佣金费率
    rate1 = 0.0005
    # 印花税:减半征收
    rate2 = 0.001 / 2
    # 过户费
    rate3 = 0.00001

    # 现在持股和资金状态
    if now_holds is None:
        now_holds = {"stocks": [], "money": 0.00,"date":""}
    else:
        logging.info("active:{} date:{}".format(active,now_holds["date"]))
    if active["method"] == "buy":
        buy_amount = active["num"] * active["price"]
        if buy_amount > now_holds["money"]:
            print("超过总资金，无法购买")
        else:
            stock = {"ts_code": active["ts_code"], "num": active["num"]}
            is_hold = False
            for hold_stock in now_holds["stocks"]:
                if active["ts_code"] == hold_stock["ts_code"]:
                    hold_stock["num"] += active["num"]
                    is_hold = True
            if not is_hold:
                now_holds["stocks"].append(stock)
            now_holds["money"] = now_holds["money"] - buy_amount - buy_amount * rate1 - buy_amount * rate3
        return now_holds
    elif active["method"] == "sell":
        sell_amout = active["num"] * active["price"]
        is_hold = False
        stocks = now_holds["stocks"]
        for key,hold_stock in enumerate(stocks):
            if (active["ts_code"] == hold_stock["ts_code"]) and (active["num"] == hold_stock["num"]):
                del stocks[key]
                now_holds["stocks"] = stocks
                is_hold = True
        if not is_hold:
            raise Exception("没有找到要卖的股票或者卖的股票数量超过持有数量:now_holds{},active:{}".format(now_holds,active))
        else:
            now_holds["money"] = now_holds["money"] + sell_amout - sell_amout * (rate2 + rate3 + rate1)
        return now_holds
    elif active["method"] == "in":
        now_holds["money"] += active["amount"]
        return now_holds
    elif active["method"] == "out":
        if now_holds["money"] > active["amount"]:
            now_holds["money"] -= active["amount"]
        else:
            print("没有足够的资金")
        return now_holds
    else:
        return now_holds


def get_price(df_stock, stock_code, kind):
    '''
    获取一只股票某天的开盘价/收盘价等行情
    :param df_stock: 某天所有股票行情
    :param stock_code: 要查询的股票行情
    :param kind: open,close,high,low
    :return: float
    '''
    prices = df_stock[df_stock["ts_code"] == stock_code][kind].values
    if len(prices) == 0:
        raise Exception("无法找到价格，{}，{}，{}".format(df_stock,stock_code,kind))
    return prices[0]


def coumpute_buy_and_sell_active(now_holds, df_to_buy, next_trade_date):
    '''
    根据模型生成的买入操作，生成操作
    :param now_holds: {"stocks":[{"ts_code":stockcode,"num":num},{"ts_code":stockcode,"num":num}],money:amout}
    :param df_to_buy: ["ts_code","prob"]
    :param next_trade_date: 下一个购买日 %Y%m%d
    :return: now_holds
    '''

    # df_stock = pro.daily(trade_date=next_trade_date)
    # 要进行交易的交易日
    # trade_date = transform_date(next_trade_date, False)
    trade_date = next_trade_date
    df_stock = pd.read_sql('''SELECT * FROM  stock_all where trade_date='{}';'''.format(trade_date), con=conn)
    # df_stock["trade_date"] = df_stock["trade_date"].apply(
    #     lambda x: (datetime.datetime.strptime(x, "%Y-%m-%d")).strftime('%Y%m%d'))
    # 获取要卖和要买股票的开盘价
    df_to_buy = pd.merge(df_to_buy, df_stock, on="ts_code")
    buy_stocks = df_to_buy["ts_code"].values
    if len(buy_stocks) == 0:
        raise Exception("未找到股票对应的价格{}".format(df_to_buy))

    sold_actives = []
    buy_actives = []

    now_holds["date"] = next_trade_date
    length = len(df_to_buy)
    # 如果没有要买的股票，则卖出所有的股票
    if length == 0:
        for hold_stock in now_holds["stocks"]:
            price = get_price(df_stock, hold_stock["ts_code"], "open")
            sold_actives.append({"method": "sell", "ts_code": hold_stock["ts_code"], "num": hold_stock["num"], "price": price})
    else:
        for hold_stock in now_holds["stocks"]:
            stock_code = hold_stock["ts_code"]
            # 检查是否有继续持股的股票，继续持有的股票不需要再买
            if stock_code in buy_stocks:
                df_to_buy = df_to_buy[df_to_buy["ts_code"] != stock_code]
            else:
                price = get_price(df_stock, stock_code, "open")
                sold_actives.append({"method": "sell", "ts_code": stock_code, "num": hold_stock["num"], "price": price})

    print("sold_actives:{},now_holds:{}".format(sold_actives, now_holds))
    #     卖出股票
    for sold_active in sold_actives:
        now_holds = holds(sold_active, now_holds)
    print("after sold_actives:{},now_holds:{}".format(sold_actives, now_holds))

    # 买入股票
    df_to_buy["buy_ratio"] = df_to_buy["prob"] / df_to_buy["prob"].sum()
    df_to_buy["buy_amount"] = df_to_buy["buy_ratio"] * now_holds["money"]
    df_to_buy["hand_num"] = df_to_buy["buy_amount"] / df_to_buy["open"] / 100
    df_to_buy["num"] = df_to_buy["hand_num"].apply(lambda x: math.floor(x) * 100)
    for _, row in df_to_buy.iterrows():
        buy_actives.append({"method": "buy", "ts_code": row["ts_code"], "num": row["num"], "price": row["open"]})

    print("buy_actives:{},now_holds:{}".format(buy_actives, now_holds))
    for buy_active in buy_actives:
        now_holds = holds(buy_active, now_holds)
    print("after buy_actives:{},now_holds:{}".format(sold_actives, now_holds))

    return now_holds


def transform_date(origin, contain_split=True):
    '''
    将%Y-%m-%d转换成%Y%m%d
    '''
    if contain_split:
        return (datetime.datetime.strptime(str(origin), "%Y-%m-%d")).strftime('%Y%m%d')
    else:
        return (datetime.datetime.strptime(str(origin), "%Y%m%d")).strftime('%Y-%m-%d')


def get_one_stock_data_from_sqlite(ts_code, start_date, end_date, is_merge_index,table="stock_all"):
    # start_date = transform_date(start_date, False)
    # end_date = transform_date(end_date, False)
    df_stock = pd.read_sql(
        '''SELECT * FROM  {} where ts_code='{}' AND trade_date>='{}' AND trade_date<='{}';'''.format(table,ts_code,
                                                                                                            start_date,
                                                                                                            end_date),
        con=conn)  # 把数据库的入口传给它 # 简单明了的 sql 语句
    # df_stock["trade_date"] = df_stock["trade_date"].apply(
    #     lambda x: (datetime.datetime.strptime(x, "%Y-%m-%d")).strftime('%Y%m%d'))
    df_stock.dropna(inplace=True)
    df_stock["trade_date"] = df_stock["trade_date"].astype(int)
    if is_merge_index:
        df_merge = df_stock.merge(df_index, how="left", suffixes=["_stock", "_index"], on="trade_date")
    else:
        df_merge = df_stock.copy(deep=True)
    return df_merge


def get_stock_basic_from_sqlite():
    '''
    获取截止日超过已上市天数的非北交所股票列表
    :param end_date: 截止日期
    :param list_days: 已上市天数
    :return:不包含北交所，上市截止日超过list_days的上市公司名单
    '''
    df_stock_basic = pd.read_sql("select * from stock_basic;",conn)
    return df_stock_basic



def down_load_data(fh, end_date, is_merge_index,is_real,step_length=100):
    # TODO:转换成从sqlite3数据库导入
    df_stock_basic = get_stock_basic_from_sqlite()
    #
    start_date = get_date_before_days(end_date, int(2.5*step_length))
    data = None
    stocks = []
    for key, ts_code in enumerate(list(df_stock_basic["ts_code"])):
        # print("开始处理{}:{}".format(key,ts_code))
        # 测试使用，实际使用需要注释掉
        # TODO:区分测试环境和正式环境
        if not is_real:
            if key > 100:
                break
        # print("正在处理{}".format(ts_code))
        # 获取股票和指数交易日数据，截止日前一年的数据
        # df_stock = get_one_stock_data(ts_code, df_index, start_date, end_date, is_merge_index)
        df_stock = get_one_stock_data_from_sqlite(ts_code, start_date, end_date, is_merge_index)
        if len(df_stock)<240:
            continue
        # 数据处理
        df_data = handle_stock_df(df_stock, fh, is_merge_index, is_contain_y=False)
        # 求后100个数据
        df_step_length = df_data.tail(step_length)
        if len(df_step_length) < step_length:
            continue
        df_step_length_numpy = df_step_length.to_numpy()
        df_step_length_numpy = df_step_length_numpy.reshape(1, step_length, 96)
        if data is None:
            data = deepcopy(df_step_length_numpy)
        else:
            data = np.concatenate((data, df_step_length_numpy), axis=0)
        stocks.append(ts_code)
    return data, stocks


def compute_to_buy(model, kinds, data, stocks, end_date, top=5, low_prob=0.85,top_prob=0.98):
    '''
    计算要买的股票
    :param model: 用于预测的模型
    :param kinds: 分类类型
    :param data: 要分类的数据
    :param stocks: 分类数据对应的股票代码
    :param end_date: 截止日期
    :param top: 要买几只股票
    :param low_prob: 被选择的最低概率
    :param top_prob:超过该概率必被选中
    :return: 要买的股票及其概率
    '''
    # data,stocks = down_load_data(fh,end_date,is_merge_index)

    pred_prob = predict_proba(model, data, 32)
    pred_res = predict(kinds, model, data)
    # 使用 NumPy 的 unique 函数统计元素出现次数
    unique_elements, counts = np.unique(pred_res, return_counts=True)
    # 将结果组合成字典
    numpy_result = dict(zip(unique_elements, counts))
    logging.info("本次统计数据：{}".format(numpy_result))
    print("本次统计数据：{}".format(numpy_result))
    # 找到数字最大的数，然后找到数字最大的数对应的概率
    # TODO:是否直接设置成8
    indices = np.flatnonzero(pred_res == pred_res.max())
    # 找到涨幅最大的股票
    stocks = np.array(stocks)
    choice_stocks = stocks[indices]
    # 计算涨幅最大股票对应的概率
    # 概率必须大于70%
    choice_stocks_prob = np.array([prob.max() for prob in pred_prob[indices]])
    if len(choice_stocks) > 0:
        df = pd.DataFrame({"ts_code": choice_stocks, "prob": choice_stocks_prob})
        # 选择概率最大的前5只股票，第二天开盘买入，同时卖出前一天的五只股票，如果前一天的股票仍然在涨幅最大的股票列报中，则不处置
        df = df.sort_values(by="prob", ascending=False, ignore_index=True)
        print("概率大于{}的股票个".format(str(top_prob)))
        logging.info("概率大于{}的股票有个".format(str(top_prob)))
        # 获取概率值大于98的股票
        df_to_buy = df.head(top)
        # df_to_buy = df_to_buy[df_to_buy["prob"] > low_prob]
        # df_to_buy = df[df["prob"]>=top_prob]
        # if len(df_to_buy) > 5:
        #     df_to_buy = df.head(top)
        # else:
        #     # 选择概率超过70%的股票
        #     df_to_buy = df_to_buy[df_to_buy["prob"] > low_prob]
        #     df_to_buy = df.head(top)
        #
    next_trade_date = get_date_before_days(end_date, -1)
    return df_to_buy, next_trade_date


def compute_now_holds_value(now_holds):
    '''
    计算现在持仓总价值
    :param now_holds:
    :param trade_date:
    :return:
    '''
    # df_stock = pro.daily(trade_date=now_holds["date"])
    # date = transform_date(now_holds["date"],False)
    date = now_holds["date"]
    df_stock = pd.read_sql('''SELECT * FROM  stock_all where trade_date='{}';'''.format(date), con=conn)
    df_hold = pd.DataFrame(now_holds["stocks"])
    df_hold = pd.merge(df_hold, df_stock, on="ts_code")
    df_hold["amount"] = df_hold["num"] * df_hold["close"]
    value = df_hold["amount"].sum() + now_holds["money"]
    return value


def run(model_save_path,trade_date, days=50, initial_amount=100000,is_real=True,fh = 15,step_length=100):
    '''

    :param trade_date: 开始计算日期
    :return:
    '''

    model = keras.saving.load_model(model_save_path)
    print(model.summary())

    is_merge_index = True
    kinds = [0, 1, 2, 3, 4, 5, 6, 7, 8]
    now_holds = holds({"method": "in", "amount": initial_amount}, None)

    for i in range(days):
        print("处理第{}天".format(i))
        start_time = time.time()
        # 获取后几天的日期
        # "%Y%m%d"
        end_date = get_date_before_days(trade_date, -i)
        # int date
        END_DATE = int(end_date)
        data, stocks = down_load_data(fh, END_DATE, is_merge_index,is_real,step_length)
        # next_date :"%Y%m%d"
        df_to_buy, next_date = compute_to_buy(model, kinds, data, stocks, end_date, top=5, low_prob=0.7)
        logging.info("{}:{}".format(end_date,df_to_buy))
        now_holds = coumpute_buy_and_sell_active(now_holds, df_to_buy, next_date)
        logging.info("{}:{}".format(end_date,now_holds))
        now_holds_value = compute_now_holds_value(now_holds)
        receive_rate = (now_holds_value - initial_amount) / initial_amount
        logging.info("第{}天，现在市值：{},累计收益率：{}".format(i, now_holds_value, receive_rate))
        print(now_holds["date"],now_holds_value,receive_rate)
        logging.info("总共花费了{}".format(time.time()-start_time))
        print("总共花费了{}".format(time.time()-start_time))


if __name__ == '__main__':
    # while True:
        model_save_path_1 = r"D:\redhand\clean\data\state_dict\inceptiontime_new.keras"
        model_save_path_2 = r"D:\redhand\clean\data\state_dict\inceptiontime_new_100_15.keras"
        model_save_path_3 = r"D:\redhand\clean\data\state_dict\inceptiontime_new_150_15.keras"
        model_save_path_4 = r"D:\redhand\clean\data\state_dict\inceptiontime_new_200_20.keras"
        model_save_path_5 = r"C:\Users\ASUS\Downloads\inceptiontime new 150 15.keras"
    #     model_save_path_4 = r"D:\redhand\clean\data\state_dict\inceptiontime_new_0502.keras"
    #     model_save_path_5 = r"D:\redhand\clean\data\state_dict\inceptiontime_new_0501.keras"
    #     model_save_path_sktime = r"D:\redhand\project\InceptionTimeClassifier(n_epochs=75)_1.zip"
        run(model_save_path_3,"20240102",is_real=False,fh=15,step_length=150)
    # print(get_date_before_days("20240102", 1))
    # 下载数据
    # get_trade_cal()
    # get_index_to_sqlite()
    # get_stock_basic_to_sqlite(20240430, LIST_DAYS)


    # sold_actives = [{'method': 'sell', 'ts_code': '000025.SZ', 'num': 1200, 'price': 15.9}]
    # now_holds = {'stocks': [{'ts_code': '000002.SZ', 'num': 1900}, {'ts_code': '000001.SZ', 'num': 2100}, {'ts_code': '000016.SZ', 'num': 4900}, {'ts_code': '000012.SZ', 'num': 3500}, {'ts_code': '000025.SZ', 'num': 1200}], 'money': 2560.33109999999, 'date': '20240116'}
    # excute_actives(sold_actives,[],now_holds)


