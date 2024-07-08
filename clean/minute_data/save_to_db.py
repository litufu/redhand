import os
import sqlite3

import numpy as np
import tushare as ts
import pandas as pd

ts.set_token('f88e93f91c79cdb865f22f40cac23a2907da36b53fa9aa150228ed27')
pro = ts.pro_api()

conn = sqlite3.connect(r"D:\redhand\clean\data\tushare_db\stock_minute.db")


def save_minute_to_db(folder_path):
    for root, dirs, files in os.walk(folder_path):
        for key, file in enumerate(files):
            print(file)
            if key <= 5:
                continue

            file_path = os.path.join(root, file)
            df = pd.read_csv(file_path, index_col=0)
            df.rename(
                columns={'日期': 'trade_date', '时间': 'trade_time', "代码": "ts_code", "开盘": "open", "最高": "high",
                         "最低": "low", "收盘": "close", "成交量(股)": "vol", "成交金额(元)": "amount",
                         "复权状态": "adj"}, inplace=True)
            df["trade_date"] = df["trade_date"].apply(change_date)
            df["ts_code"] = df["ts_code"].apply(change_ts_code)
            df.to_sql("stock_all",conn,index=False,if_exists="append")


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

def get_index_to_sqlite():
    df_index = get_df_index()
    df_index.to_sql("df_index",conn,if_exists='replace',index=False)


def get_stock_basic_to_sqlite(end_date,list_days):
    df_stock_basic = pro.stock_basic(exchange='', list_status='L',
                                     fields='ts_code,market,symbol,name,area,industry,list_date')
    df_stock_basic["list_duaration"] = end_date - df_stock_basic["list_date"].astype(int)
    # 上市超过LIST_DAYS
    df_stock_basic = df_stock_basic[df_stock_basic["list_duaration"] > list_days]
    # 不包含北交所，北交所有流动性问题
    df_stock_basic = df_stock_basic[df_stock_basic["market"] != "北交所"]
    df_stock_basic.to_sql("stock_basic", conn, if_exists="replace", index=False)


def change_date(trade_date):
    new_date = trade_date.replace("-","")
    return new_date

def change_ts_code(ts_code):
    temps = ts_code.split(".")
    if len(temps)!=2:
        raise Exception("数据拆分错误：{}".format(ts_code))
    res = ".".join([temps[1],temps[0].upper()])
    return res


def update_minute_format():
    '''
    更新分钟数据库数据，使得日期格式和代码格式保持一致
    :return:
    '''
    conn = sqlite3.connect(r"D:\redhand\clean\data\tushare_db\stock_minute.db")
    cursor = conn.cursor()
    df = pd.read_sql("select trade_date,ts_code from stock_all",conn)
    trade_dates = np.unique(df["trade_date"])
    for key,trade_date in enumerate(trade_dates):
        print(trade_date)
        if "-" in trade_date:
            new_date = change_date(trade_date)
            sql_trade_date = "UPDATE stock_all SET trade_date = REPLACE(trade_date, '{}', '{}');".format(trade_date,new_date)
            cursor.execute(sql_trade_date)
            if key%10 == 0:
                conn.commit()
    ts_codes = np.unique(df["ts_code"])
    for key,ts_code in enumerate(ts_codes):
        print(ts_code)
        if ts_code.startswith("s"):
            new_ts_code = change_ts_code(ts_code)
            sql_ts_code = "UPDATE stock_all SET ts_code = REPLACE(ts_code, '{}', '{}');".format(ts_code,new_ts_code)
            cursor.execute(sql_ts_code)
            if key%10 == 0:
                conn.commit()
    conn.commit()
    cursor.close()
    conn.close()

def delete_table(conn, tablename):
    '''
    删除数据表
    :param conn:
    :param tablename:
    :return:
    '''
    cursor = conn.cursor()
    try:
        cursor.execute("DROP TABLE {}".format(tablename))
    except Exception as e:
        print(e)

if __name__ == '__main__':

    conn = sqlite3.connect(r"D:\redhand\clean\data\tushare_db\stock_minute.db")
    # delete_table(conn,"stock_all")


    # save_minute_to_db(r"D:\stockdata\minute")
    df = pd.read_sql("select * from stock_all order by ts_code asc limit 3", conn)
    print(df)
    # update_minute_format()
    # get_index_to_sqlite()
    # get_stock_basic_to_sqlite(20231231, 200)