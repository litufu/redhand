import keras
import sqlite3
import pandas as pd
import numpy as np
import time
import datetime
from inceptiontime_predict import predict_proba, predict
from copy import deepcopy
from constants import CATEGORIES, pro
import logging

logging.basicConfig(filename=r"D:\redhand\clean\data\log\future_inceptiontime.log", filemode="w",
                    format="%(asctime)s %(name)s:%(levelname)s:%(message)s",
                    datefmt="%d-%m-%Y %H:%M:%S", level=logging.DEBUG)


# 已测试
def drop_table(conn, table_name):
    cursor = conn.cursor()
    cursor.execute("DROP TABLE IF EXISTS {}".format(table_name))
    conn.commit()
    cursor.close()
    conn.close()


def get_future_basic(conn):
    # exchanges = ["DCE","SHFE","CZCE","CFFEX","INE","GFEX"]
    # for exchange in exchanges:
    #     df = pro.fut_basic(exchange=exchange)
    #     df.to_sql("basic", conn, if_exists='append', index=False)
    df = pd.read_csv(r"D:\redhand\clean\data\future_basic.csv",encoding="gbk")
    df.to_sql("basic", conn, if_exists='replace', index=False)


def get_trade_cal(conn):
    exchanges = ["DCE", "SHFE", "CZCE", "CFFEX", "INE", "GFEX"]
    for exchange in exchanges:
        df = pro.trade_cal(exchange=exchange)
        df.to_sql("trade_cal", conn, if_exists='append', index=False)


def get_future_settle(conn):
    '''
    获取期货结算信息
    :param conn:
    :return:
    '''
    df = pd.read_sql("SELECT distinct cal_date FROM trade_cal WHERE is_open=1", conn)
    for trade_date in df["cal_date"].values.tolist():
        print(trade_date)
        if trade_date > "20240630":
            continue
        df_settle = pro.fut_settle(trade_date=trade_date)
        print(df_settle)
        df_settle.to_sql("settle", conn, if_exists='append', index=False)
        time.sleep(0.8)


# 已测试
def transform_future_symbol(symbol,df_basic):
    '''
    将期货合约转换为标准格式
    :param symbol:
    :param df_basic:期货基本信息表
    :return:
    '''

    code = symbol.split(".")[1]

    df = df_basic.loc[df_basic["symbol"] == code.upper()]
    if len(df) == 0:
        raise ValueError("symbol not found in basic table {}".format(symbol))
    else:
        ts_code = df["ts_code"].values[0]
        return ts_code

# 已测试
def get_multiplier(symbol, conn,df_basic):
    '''
    获取合约乘数
    :param conn:
    :param symbol:
    :return:
    '''
    symbol = transform_future_symbol(symbol,df_basic)
    df = pd.read_sql("SELECT * FROM basic WHERE ts_code='{}'".format(symbol), conn)
    return df["multiplier"].values[0]


# 已测试
def create_hold_table(conn, table_name):
    '''
    创建持仓表
    :param conn:
    :param table_name:
    :return:
    '''
    sql = f"CREATE TABLE IF NOT EXISTS {table_name} (symbol TEXT,active TEXT,active_time TEXT,active_price REAL,active_num REAL,active_amount REAL,active_bond REAL,active_fee REAL,deactive TEXT,deactive_time TEXT,deactive_price REAL,deactive_num REAL,deactive_amount REAL,deactive_bond REAL,deactive_fee REAL)"
    cur = conn.cursor()
    cur.execute(sql)
    conn.commit()
    cur.close()


def store(conn, table, data):
    '''
    table columns:[symbol,active,active_time,active_price,active_num,active_amount,active_fee,deactive,deactive_time,deactive_num,deactive_price,deactive_amount,deactive_fee]
    :param conn:
    :param table:hold table name
    :param data:
    :return:
    '''
    sql = f"INSERT INTO {table} (symbol,active,active_time,active_price,active_num,active_amount,active_fee,deactive,deactive_time,deactive_num,deactive_price,deactive_amount,deactive_fee) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)"
    cur = conn.cursor()
    for item in data:
        cur.execute(sql, item)
    conn.commit()
    cur.close()


# 已测试
def get_now_holds(conn, hold_table):
    '''
    获取当前持仓
    :param conn:
    :param hold_table:持有仓位表名
    :return:df_hold,当前持仓
    '''

    df_hold = pd.read_sql(f"SELECT * FROM {hold_table} where deactive is null", conn)
    print("now_holds:{}".format(df_hold))
    return df_hold


def holds_future(active, conn, hold_table, date_money):
    '''
    :param active:
    {"method":"buy","symbol":symbol,",num:"num",price:"price,"datetime":"2021-09-01"}
    {"method":"flat_buy","symbol":symbol,",num:"num",price:"price}
    {"method":"sell","symbol":symbol,",num:"num",price:"price}
    {"method":"flat_sell","symbol":symbol,",num:"num",price:"price}
    {"method":"in","bond":amount,}
    {"method":"out","bond":amount}
    :param conn:
    :param hold_table:持有仓位表名
    :param date_money:
    :return:
    '''
    print("date_money:{}".format(date_money))
    # 开仓
    if active["method"] == "buy" or active["method"] == "sell":
        bond = active["bond"]
        if bond > date_money["money"]:
            print("超过总资金，无法购买")
        else:
            insert_sql = f"INSERT INTO {hold_table} (symbol,active,active_time,active_price,active_num,active_amount,active_bond,active_fee) VALUES (?,?,?,?,?,?,?,?)"
            cur = conn.cursor()
            cur.execute(insert_sql, (
            active["symbol"], active["method"], active["datetime"], active["price"], active["num"], active["amount"],
            active["bond"], 0))
            conn.commit()
            date_money["money"] = date_money["money"] - bond
    # 平仓
    elif active["method"] == "flat_buy" or active["method"] == "flat_sell":
        update_sql = f"UPDATE {hold_table} SET deactive='{active['method']}',deactive_time='{active['datetime']}',deactive_price='{active['price']}',deactive_num='{active['num']}',deactive_amount='{active['amount']}',deactive_bond='{active['bond']}',deactive_fee=0 WHERE symbol='{active['symbol']}' AND deactive is null"
        cur = conn.cursor()
        cur.execute(update_sql)
        conn.commit()
        date_money["money"] = date_money["money"] + active["bond"]
    # 资金入金
    elif active["method"] == "in":
        date_money["money"] += active["bond"]
    # 资金出金
    elif active["method"] == "out":
        if date_money["money"] > active["bond"]:
            date_money["money"] -= active["bond"]
        else:
            print("没有足够的资金")
    return date_money


# 已测试
def get_trade_time(conn, future_table, freq):
    '''
    获取期货所有的交易时间
    :param conn:
    :param future_table:期货数据表名
    :param freq:
    :return:df_trade_time,期货所有的交易时间
   已测试           datetime
0  2005-01-05 09:00:00
1  2005-01-05 09:05:00
2  2005-01-05 09:10:00
3  2005-01-05 09:15:00
4  2005-01-05 09:20:00
5  2005-01-05 09:25:00
    '''

    if freq == "d":
        df_trade_time = pd.read_sql(f"SELECT distinct FROM {future_table} ", conn)
        df_trade_time = df_trade_time.sort_values(by="trade_date", ascending=True, ignore_index=True)
    elif freq == "m":
        df_trade_time = pd.read_sql(f"SELECT distinct datetime FROM {future_table} ", conn)
        df_trade_time = df_trade_time.sort_values(by="datetime", ascending=True, ignore_index=True)
    return df_trade_time


def get_future_time(conn, future_table, symbol,current_date_str,before_num):

    '''
    获取期货某个合约的交易时间
    :param conn:
    :param future_table:期货数据表名
    :param symbol:期货合约
    :param current_date_str:当前日期字符串
    :param before_num:前几天或前几分钟，后几天或后几分钟使用负数
    :return:df_trade_time,期货某个合约的交易时间
    '''
    current_date_str = str(current_date_str) if type(current_date_str) != str else current_date_str
    # 1、获取期货所有交易时间
    df_future_time = pd.read_sql(f"SELECT * FROM {future_table} where symbol='{symbol}'", conn)
    trade_dates = df_future_time["datetime"].values.tolist()
    index = trade_dates.index(current_date_str)
    find_index = index - before_num
    if find_index < 0:
        raise Exception("无法找到之前的日期")
    elif find_index > len(trade_dates):
        raise Exception("超过 最大日期")
    return trade_dates[find_index]


# 已测试
def get_before_time(current_date_str, df_trade_date, before_num, freq):
    '''
    获取当期交易时间前或后的交易时间,需要根据期货品种来确定
    :param current_date_str:当前日期字符串
    :param before_num:前几天或前几分钟，后几天或后几分钟使用负数
    :param df_trade_date:期货交易时间表
    :param freq:交易频率，d代表日，m代表分钟
    :return:交易日期字符串或时间字符串
    '''
    current_date_str = str(current_date_str) if type(current_date_str) != str else current_date_str
    if freq == "d":
        trade_dates = df_trade_date["trade_date"].values.tolist()
    elif freq == "m":
        trade_dates = df_trade_date["datetime"].values.tolist()
    index = trade_dates.index(current_date_str)
    find_index = index - before_num
    if find_index < 0:
        raise Exception("无法找到之前的日期")
    elif find_index > len(trade_dates):
        raise Exception("超过 最大日期")
    return trade_dates[find_index]


# 已测试
def get_one_future_data(conn,future_table, symbol, step_length,end_date):
    '''
    获取一只期货的交易数据
    :param conn: 交易时段所有的期货交易信息
    :param future_table: 期货数据表名
    :param symbol: 期货代码
    :param step_length: 往后回顾的长度
    :param end_date: 截止日期
    :return:
    '''
    # 处理数据
    df_futures = pd.read_sql(f"SELECT * FROM {future_table} where symbol='{symbol}' and datetime<='{end_date}' order by datetime desc limit {step_length}", conn)
    df_futures = df_futures.sort_values(by="datetime", ascending=True, ignore_index=True)
    df_futures = df_futures.dropna()
    if len(df_futures) == step_length:
        return df_futures
    else:
        return None


# 已测试
def handle_future_data(df):
    '''
    处理期货数据
    :param df: 期货数据
    '''
    df = df.sort_values(by="datetime", ascending=True, ignore_index=True)
    df = df.drop(labels=["symbol", "trade_date", "datetime"], axis=1)
    df = (df - df.mean()) / df.std()
    return df


# 已测试
def download_future_data(conn, end_date, is_real, df_trade_date, step_length, freq, n_vars, future_table):
    '''
    获取期货数据信息
    注意：不同的期货有不同的交易时间，因此需要根据交易时间来获取数据
    :param conn: 数据库连接
    :param end_date: 截止日期
    :param is_real: 是否真实测试
    :param df_trade_date: 交易日历
    :param step_length: 往后回顾的长度
    :param freq: 日线还是分钟线
    :param n_vars: 字段个数
    :param future_table: 期货数据表名
    :return:
    '''
    #1、获取当前时间的所有期货数据
    now_future = pd.read_sql(f"SELECT * FROM {future_table} where datetime='{end_date}'", conn)
    data = None
    futures = []
    # 2、根据当前时间的期货数据获取期货代码
    for key, symbol in enumerate(np.unique(now_future["symbol"])):
        # 测试使用，实际使用需要注释掉
        # TODO:区分测试环境和正式环境
        if not is_real:
            if key > 100:
                break
        print("正在处理{}".format(symbol))
        # 3、获取所有期货代码往前step_length时刻的数据
        df_future = get_one_future_data(conn,future_table, symbol, step_length,end_date)
        # 如果在开始时间和截止时间内，该股票的数据小于历史回顾和未来预测期数据，那么该股票将没有可测试的数据，因此不测试该股票
        if df_future is None:
            continue
        # 数据处理
        df_data = handle_future_data(df_future)
        df_step_length_numpy = df_data.to_numpy()
        df_step_length_numpy = df_step_length_numpy.reshape(1, step_length, n_vars)
        # 汇总数据
        if data is None:
            data = deepcopy(df_step_length_numpy)
        else:
            data = np.concatenate((data, df_step_length_numpy), axis=0)
        futures.append(symbol)
    return data, futures


def compute_to_active(model, kinds, data, symbols, end_date, df_trade_date, top, freq):
    '''
    计算要买的期货或要卖的期货
    :param model: 用于预测的模型
    :param kinds: 分类类型
    :param data: 要分类的数据
    :param symbols: 分类数据对应的股票代码
    :param end_date: 截止日期
    :param df_trade_date: 期货交易时间表
    :param top: 选择概率最高的前几只期货
    :param freq: 数据频率
    :return: 要买的股票及其概率
    '''
    # 使用模型对数据进行预测
    pred_prob = predict_proba(model, data, 32)
    pred_res = predict(kinds, model, data)
    # 使用 NumPy 的 unique 函数统计元素出现次数
    unique_elements, counts = np.unique(pred_res, return_counts=True)
    numpy_result = dict(zip(unique_elements, counts))
    print("本次统计数据：{}".format(numpy_result))
    # 找到数字最大的数和最小的数，然后找到数字最大的数和最小的数对应的概率
    indices_max = np.flatnonzero(pred_res == kinds[0])
    indices_min = np.flatnonzero(pred_res == kinds[-1])
    # 找到涨幅最大的股票
    symbols = np.array(symbols)
    # 要买的期货
    choice_futures_to_buy = symbols[indices_max]
    # 要卖的期货
    choice_futures_to_sell = symbols[indices_min]
    # 计算涨幅最大期货或跌幅最大期货对应的概率
    choice_futures_prob_buy = np.array([prob.max() for prob in pred_prob[indices_max]])
    choice_futures_prob_sell = np.array([prob.max() for prob in pred_prob[indices_min]])
    df_to_buy = None
    if len(choice_futures_to_buy) > 0:
        df = pd.DataFrame({"symbol": choice_futures_to_buy, "prob": choice_futures_prob_buy})
        # 选择概率最大的前几只期货，下一次时间开盘点买入，同时平仓前一天的持仓期货，如果前一天的期货仍然在涨幅最大的期货列报中，则不处置
        df = df.sort_values(by="prob", ascending=False, ignore_index=True)
        # 获取概率最大的几个期货
        df_to_buy = df.head(top)
    df_to_sell = None
    if len(choice_futures_to_sell) > 0:
        df = pd.DataFrame({"symbol": choice_futures_to_sell, "prob": choice_futures_prob_sell})
        # 选择概率最大的前几只期货，下一次时间开盘点卖出，同时平仓前一天的持仓期货，如果前一天的期货仍然在跌幅最大的期货列报中，则不处置
        df = df.sort_values(by="prob", ascending=False, ignore_index=True)
        # 获取概率最大的几个期货
        df_to_sell = df.head(top)
    next_trade_time = get_before_time(end_date, df_trade_date, -1, freq)
    return df_to_buy, df_to_sell, next_trade_time


# 已测试
def get_future_price(df_future, symbol, kind):
    '''
    获取一只期货某天的开盘价/收盘价等行情
    :param df_future: 某时点所有期货行情
    :param symbol: 要查询的期货代码
    :param kind: open,close,high,low
    :return: float
    '''
    prices = df_future[df_future["symbol"] == symbol][kind].values
    if len(prices) == 0:
        raise Exception("无法找到价格，{}，{}，{}".format(df_future, symbol, kind))
    return prices[0]


def get_future_num(df_hold, symbol):
    '''
    获取持仓期货的数量
    :param df_hold: 当前持仓
    :param symbol: 期货代码
    :return: int
    '''
    num = df_hold[df_hold["symbol"] == symbol]["active_num"].values[0]
    if num is None:
        return 0
    else:
        return num


# 已测试
def get_margin_rate(symbol, conn, trade_date, active,df_basic):
    '''
    获取期货的保证金率
    :param conn: 数据库连接
    :param symbol: 期货代码
    :param trade_date: 交易日期
    '''
    active_contrast = {
        "buy": "long_margin_rate",
        "sell": "short_margin_rate",
        "flat_buy": "short_margin_rate",
        "flat_sell": "long_margin_rate",
    }
    symbol = transform_future_symbol(symbol,df_basic)
    rate = active_contrast[active]
    sql = f"SELECT {rate} FROM settle WHERE ts_code='{symbol}' and trade_date='{trade_date}'"
    df = pd.read_sql(sql, conn)
    if len(df) == 0:
        # 求最相近的期货品种
        sql = f"SELECT {rate} FROM settle WHERE ts_code LIKE '{symbol[:2]}%'"
        df = pd.read_sql(sql, conn)
        if len(df) == 0:
            return 0.15
            print("无法找到保证金率{}".format(symbol))
        else:
            return df[rate].values.tolist()[0]
    else:
        return df[rate].values.tolist()[0]


# 已测试
def time_to_date(time_str):
    '''
    时间字符串转日期字符串
    :param time_str: 时间字符串
    :return: 日期字符串
    '''
    return datetime.datetime.strptime(time_str, '%Y-%m-%d %H:%M:%S').strftime('%Y%m%d')


# 已测试
def compute_flat_active(conn, symbol, df_future, now_holds, trade_time, active,df_basic):
    '''
    计算平仓操作
    :param conn:
    :param symbol:
    :param df_future:
    :param now_holds:
    :param trade_time:
    :param active:
    :param df_basic:期货基础数据
    :return:
    '''
    df_future = deepcopy(df_future)
    price = get_future_price(df_future, symbol, "open")
    num = get_future_num(now_holds, symbol)
    multiplier = get_multiplier(symbol, conn,df_basic)
    amount = price * num * multiplier
    trade_date = time_to_date(trade_time)
    margin_rate = get_margin_rate(symbol, conn, trade_date, active,df_basic)
    bond = amount * margin_rate
    active = {"method": active, "symbol": symbol, "num": num, "price": price, "datetime": trade_time,
              "amount": amount, "bond": bond}
    return active


# 已测试
def compute_open_active(conn, symbol, df_future, now_holds, trade_time, active,df_basic):
    '''
    计算开仓操作
    :param conn:
    :param symbol:
    :param df_future:
    :param now_holds:
    :param trade_time:
    :param active:
    :param df_basic:期货基础数据
    :return:
    '''
    df_future = deepcopy(df_future)
    price = get_future_price(df_future, symbol, "open")
    num = 1
    multiplier = get_multiplier(symbol, conn,df_basic)
    amount = price * num * multiplier
    trade_date = time_to_date(trade_time)
    margin_rate = get_margin_rate(symbol, conn, trade_date, active,df_basic)
    bond = amount * margin_rate
    active = {"method": active, "symbol": symbol, "num": num, "price": price, "datetime": trade_time,
              "amount": amount, "bond": bond}
    return active


# 已测试
def coumpute_buy_and_sell_active(conn, date_money, df_to_buy, df_to_sell, next_trade_time, future_table, hold_table,df_basic):
    '''
    根据模型生成的买入操作，生成操作
    :param conn: 数据库连接
    :param now_holds: {"trade_date":trade_date,money:amount}
    :param df_to_buy: ["symbol","prob"]
    :param df_to_sell: ["symbol","prob"]
    :param next_trade_date: 下一个购买日 %Y%m%d %H%M%S
    :param future_table:期货数据表名
    :return: date_money
    '''
    # 交易日期
    trade_time = next_trade_time
    date_money["datetime"] = next_trade_time

    # 获取股票该下一个交易时段的所有期货数据
    df_future = pd.read_sql('''SELECT * FROM  {} where datetime='{}';'''.format(future_table, trade_time), con=conn)
    # 获取要卖和要买期货的开盘价
    if df_to_buy is not None:
        buy_futures = df_to_buy["symbol"].values.tolist()
    else:
        buy_futures = []

    if df_to_sell is not None:
        sell_futures = df_to_sell["symbol"].values.tolist()
    else:
        sell_futures = []

    # 如果要持仓中存在要买的期货，则继续持有
    now_holds = get_now_holds(conn, hold_table)
    now_holds_buy = now_holds[now_holds["active"] == "buy"]
    hold_buy_futures = now_holds_buy["symbol"].values.tolist()
    now_holds_sell = now_holds[now_holds["active"] == "sell"]
    hold_sell_futures = now_holds_sell["symbol"].values.tolist()

    # 计算原持有差集并平仓处理
    flat_buy = set(hold_buy_futures) - set(buy_futures)
    for symbol in flat_buy:
        if len(df_future[df_future["symbol"] == symbol]) == 0:
            continue
        active = compute_flat_active(conn, symbol, df_future, now_holds, trade_time, "flat_buy",df_basic)
        date_money = holds_future(active, conn, hold_table, date_money)
    flat_sell = set(hold_sell_futures) - set(sell_futures)
    for symbol in flat_sell:
        if len(df_future[df_future["symbol"] == symbol]) == 0:
            continue
        active = compute_flat_active(conn, symbol, df_future, now_holds, trade_time, "flat_sell",df_basic)
        date_money = holds_future(active, conn, hold_table, date_money)

    # 计算新购买差集并开仓处理
    continues_buy = set(buy_futures) - set(hold_buy_futures)
    for symbol in continues_buy:
        if len(df_future[df_future["symbol"] == symbol]) == 0:
            continue
        active = compute_open_active(conn, symbol, df_future, now_holds, trade_time, "buy",df_basic)
        date_money = holds_future(active, conn, hold_table, date_money)
    continues_sell = set(sell_futures) - set(hold_sell_futures)
    for symbol in continues_sell:
        if len(df_future[df_future["symbol"] == symbol]) == 0:
            continue
        active = compute_open_active(conn, symbol, df_future, now_holds, trade_time, "sell",df_basic)
        date_money = holds_future(active, conn, hold_table, date_money)

    return date_money

def get_recent_trade_price(symbol,conn, future_table, now_time,price_type):
    '''
    获取最近交易日的交易时间
    :param symbol: 期货代码
    :param conn: 数据库连接
    :param future_table: 期货数据表名
    :param now_time: 当前时间
    :param df_trade_date: 期货交易时间表
    :param freq: 数据频率
    :param price_type: 价格类型close open high low
    :return: 最近交易日的交易时间
    '''

    # 获取该交易日的交易时间
    df_future = pd.read_sql('''SELECT * FROM  {} where datetime<'{}' and symbol='{}' order by datetime desc limit 1;'''.format(future_table,now_time, symbol), con=conn)
    if len(df_future) == 0:
        return None
    else:
        return df_future[price_type].values[0]

def compute_now_value(conn, date_money, hold_table, future_table,df_basic):
    '''
    计算当前持仓价值
    :param conn: 数据库连接
    :param date_money: {"datetime":trade_date,"money":amount}
    :param hold_table: 持仓表名
    :param future_table: 期货表名
    :return: float
    '''
    # 根据买入还是卖出计算持仓价值
    now_holds = get_now_holds(conn, hold_table)
    df_old_hold = deepcopy(now_holds)
    now = date_money["datetime"]
    df_future = pd.read_sql('''SELECT * FROM  {} where datetime='{}';'''.format(future_table, now), con=conn)
    df_hold = pd.merge(df_old_hold, df_future, on="symbol")

    if len(df_hold) != len(df_old_hold):
        df_old_hold["multiplier"] = df_old_hold["symbol"].apply(lambda x: get_multiplier(x, conn, df_basic))
        # 查找最近的交易日期所对应的期货价格
        df_old_hold["close"] = df_old_hold["symbol"].apply(lambda x: get_recent_trade_price(x, conn, future_table, now, "close"))
        df_old_hold["amount"] = df_old_hold["close"] * df_old_hold["active_num"] * df_old_hold["multiplier"]
        # 计算买入持仓价值
        df_old_hold_buy = df_old_hold.loc[df_old_hold["active"] == "buy"]
        df_old_hold_buy = deepcopy(df_old_hold_buy)
        df_old_hold_buy["bond"] = df_old_hold_buy["active_bond"] + (df_old_hold_buy["amount"] - df_old_hold_buy["active_amount"])
        buy_value = df_old_hold_buy["bond"].sum()
        # 计算卖出持仓价值
        df_old_hold_sell = df_old_hold.loc[df_old_hold["active"] == "sell"]
        df_old_hold_sell = deepcopy(df_old_hold_sell)
        df_old_hold_sell["bond"] = df_old_hold_sell["active_bond"] + (df_old_hold_sell["active_amount"] - df_old_hold_sell["amount"])
        sell_value = df_old_hold_sell["bond"].sum()

        value = buy_value + sell_value + date_money["money"]
        return value
    else:
        df_hold["multiplier"] = df_hold["symbol"].apply(lambda x: get_multiplier(x, conn, df_basic))
        df_hold["amount"] = df_hold["close"] * df_hold["active_num"] * df_hold["multiplier"]
        # 计算买入持仓价值
        df_hold_buy = df_hold.loc[df_hold["active"] == "buy"]
        df_hold_buy = deepcopy(df_hold_buy)
        df_hold_buy["bond"] = df_hold_buy["active_bond"] + (df_hold_buy["amount"] - df_hold_buy["active_amount"])
        buy_value = df_hold_buy["bond"].sum()
        # 计算卖出持仓价值
        df_hold_sell = df_hold.loc[df_hold["active"] == "sell"]
        df_hold_sell = deepcopy(df_hold_sell)
        df_hold_sell["bond"] = df_hold_sell["active_bond"] + (df_hold_sell["active_amount"] - df_hold_sell["amount"])
        sell_value = df_hold_sell["bond"].sum()

        value = buy_value + sell_value + date_money["money"]
        return value


def run_future_model(settings):
    # 获取模型
    model_save_path = settings["model_save_path"]
    model = keras.saving.load_model(model_save_path)
    print(model.summary())
    # 获取数据库地址
    db_path = settings["db"]
    conn = sqlite3.connect(db_path)
    # 获取交易开始日%y%m%d %H%M%S
    trade_time = settings["trade_time"]
    # 获取期货数据表名
    future_table = settings["future_table"]
    # 获取期货数据表名
    hold_table = settings["hold_table"]
    # 获取分类类型
    kinds = CATEGORIES
    # 获取选择概率最高的前几只期货
    top = settings["top"]
    # 字段个数
    n_vars = settings["n_vars"]
    # 是否真实交易
    is_real = settings["is_real"]
    # 获取回顾期间
    step_length = settings["step_length"]
    # 获取交易期数
    periods = settings["periods"]
    # 获取交易频率
    freq = settings["freq"]
    # 获取初始资金
    initial_amount = settings["initial_amount"]
    # 获取所有的交易时间
    df_trade_time = get_trade_time(conn, future_table, freq)
    # 初始化现有资金
    now_holds = {"datetime": trade_time, "money": initial_amount}
    # 获取期货基础数据
    df_basic = pd.read_sql('''SELECT * FROM  basic;''', con=conn)

    # 开始逐期测试
    for i in range(periods):
        # 从第0天开始
        print("处理第{}期".format(i))
        start_time = time.time()
        # 获取后几期的时间
        # "%Y%m%d"
        end_date = get_before_time(trade_time, df_trade_time, -i, freq)
        print("处理期间：{}-{}".format(end_date, trade_time))
        # 获取要测试的期货及其代码数据
        data, futures = download_future_data(conn, end_date, is_real, df_trade_time, step_length, freq, n_vars,
                                             future_table)
        print(data.shape)
        # next_date :"%Y%m%d"
        # 计算要买卖的期货和下一个交易时间,下一个交易时间可能有的期货没有数据
        df_to_buy, df_to_sell, next_date = compute_to_active(model, kinds, data, futures, end_date, df_trade_time, top,
                                                             freq)
        logging.info("{}:{},{}".format(end_date, df_to_buy, df_to_sell))
        # 计算买卖操作
        date_money = coumpute_buy_and_sell_active(conn, now_holds, df_to_buy, df_to_sell, next_date, future_table,
                                                  hold_table,df_basic)
        print("{}:{}".format(end_date, date_money))
        # 计算现在持有期货的价值
        now_holds_value = compute_now_value(conn, date_money, hold_table, future_table,df_basic)
        # 计算收益率
        receive_rate = (now_holds_value - initial_amount) / initial_amount
        logging.info("第{}期，现在市值：{},累计收益率：{}".format(i, now_holds_value, receive_rate))
        print(date_money["datetime"], now_holds_value, receive_rate)
        logging.info("总共花费了{}".format(time.time() - start_time))
        print("总共花费了{}".format(time.time() - start_time))


def delete_data_from_settle(conn, conditions):
    '''
    20241231 DCE
    20241230 DCE
    20241227 DCE
    20241226 DCE
    :param conn:
    :param conditions:
    :return:
    '''
    for condition in conditions:
        exchange = condition["exchange"]
        trade_date = condition["trade_date"]
        sql = f"DELETE FROM settle WHERE exchange='{exchange}' AND trade_date='{trade_date}'"
        cursor = conn.cursor()
        cursor.execute(sql)
        conn.commit()


if __name__ == '__main__':
    settings_minute_24_2 = {
        "model_save_path": r"D:\redhand\clean\data\state_dict\future\inceptiontime_future_m_24_2.keras",
        "trade_time": "2020-01-08 09:00:00",
        "is_real": True,
        "periods": 2500,
        "top": 2,
        "db": r"D:\futures\futures.db",
        "future_table": "futures",
        "hold_table": "hold_minute_24_2",
        "step_length": 24,  # 16*20
        "fh": 2,  # 16*5
        "freq": "m",  # d代表day,m代表minute
        "n_vars": 6,
        "depth": 6,
        "initial_amount": 100000,
        "model": "inceptiontime",  # lstmfcn,inceptiontime
    }
    settings_minute_24_3 = {
        "model_save_path": r"D:\redhand\clean\data\state_dict\future\inceptiontime_future_m_24_3.keras",
        "trade_time": "2020-01-08 09:00:00",
        "is_real": True,
        "periods": 2500,
        "top": 2,
        "db": r"D:\futures\futures.db",
        "future_table": "futures",
        "hold_table": "hold_minute_24_3",
        "step_length": 24,  # 16*20
        "fh": 3,  # 16*5
        "freq": "m",  # d代表day,m代表minute
        "n_vars": 6,
        "depth": 6,
        "initial_amount": 100000,
        "model": "inceptiontime",  # lstmfcn,inceptiontime
    }
    settings_minute_24_4 = {
        "model_save_path": r"D:\redhand\clean\data\state_dict\future\inceptiontime_future_m_24_4.keras",
        "trade_time": "2020-01-08 09:00:00",
        "is_real": True,
        "periods": 2500,
        "top": 2,
        "db": r"D:\futures\futures.db",
        "future_table": "futures",
        "hold_table": "hold_minute_24_4",
        "step_length": 24,  # 16*20
        "fh": 4,  # 16*5
        "freq": "m",  # d代表day,m代表minute
        "n_vars": 6,
        "depth": 6,
        "initial_amount": 100000,
        "model": "inceptiontime",  # lstmfcn,inceptiontime
    }
    settings_minute_10_8 = {
        "model_save_path": r"D:\redhand\clean\data\state_dict\future\inceptiontime_new_future_m_10_8.keras",
        "trade_time": "2020-01-08 09:00:00",
        "is_real": True,
        "periods": 2500,
        "top": 2,
        "db": r"D:\futures\futures.db",
        "future_table": "futures",
        "hold_table": "hold_minute_10_8",
        "step_length": 10,  # 16*20
        "fh": 8,  # 16*5
        "freq": "m",  # d代表day,m代表minute
        "n_vars": 6,
        "depth": 6,
        "initial_amount": 100000,
        "model": "inceptiontime",  # lstmfcn,inceptiontime
    }
    settings_minute_10_10 = {
        "model_save_path": r"D:\redhand\clean\data\state_dict\future\inceptiontime_new_future_m_10_10.keras",
        "trade_time": "2020-01-08 09:00:00",
        "is_real": True,
        "periods": 2500,
        "top": 2,
        "db": r"D:\futures\futures.db",
        "future_table": "futures",
        "hold_table": "hold_minute_10_10",
        "step_length": 10,  # 16*20
        "fh": 10,  # 16*5
        "freq": "m",  # d代表day,m代表minute
        "n_vars": 6,
        "depth": 6,
        "initial_amount": 100000,
        "model": "inceptiontime",  # lstmfcn,inceptiontime
    }
    settings_minute_20_10 = {
        "model_save_path": r"D:\redhand\clean\data\state_dict\future\inceptiontime_new_future_m_20_10.keras",
        "trade_time": "2020-01-08 09:00:00",
        "is_real": True,
        "periods": 2500,
        "top": 2,
        "db": r"D:\futures\futures.db",
        "future_table": "futures",
        "hold_table": "hold_minute_20_10",
        "step_length": 20,  # 16*20
        "fh": 10,  # 16*5
        "freq": "m",  # d代表day,m代表minute
        "n_vars": 6,
        "depth": 6,
        "initial_amount": 100000,
        "model": "inceptiontime",  # lstmfcn,inceptiontime
    }
    conn = sqlite3.connect(settings_minute_20_10["db"])
    # drop_table(conn, settings_minute_20_10["hold_table"])
    create_hold_table(conn, settings_minute_20_10["hold_table"])
    # df_hold = pd.read_sql('''SELECT * FROM  {}  where deactive is null; '''.format(settings_minute_24_2["hold_table"]), con=conn)
    # df = pro.fut_settle(ts_code='JD2005.DCE')
    # print(df)
    # df_settle = pd.read_sql('''SELECT * FROM  settle where ts_code like 'JD%';''', con=conn)
    # print(df_settle)
    # df_hold.to_csv("df_hold.csv", index=False, encoding="gbk")
    # df_future = pd.read_sql('''SELECT * FROM  {} where datetime='{}';'''.format(settings_minute_24_2["future_table"], "2020-01-08 10:15:00"), con=conn)
    # df_future.to_csv("df_future.csv", index=False, encoding="gbk")
    # date_money = {"datetime": "2020-01-08 09:10:00", "money": 83750.5}
    # print(df_future)
    # now_holds_value = compute_now_value(conn, date_money,"hold", "futures")
    # print(now_holds_value)
    # drop_table(conn, "hold")
    # run_future_model(settings_minute_20_10)
    # df_hold = pd.read_sql('''SELECT * FROM  {} ; '''.format(settings_minute_10_8["hold_table"]), con=conn)
    # df_hold.to_csv("df_hold_10_8.csv", index=False, encoding="gbk")
