import time, datetime, sqlite3, logging
from tqdm import tqdm
import pandas as pd
import numpy as np
from constants import ts,pro


def transforme_data(freq="ME",table_name="stock_monthly"):
    conn = sqlite3.connect(r"D:\stock\stock.db")
    df_index = pd.read_sql("select * from share_index ", conn)
    for ts_code in df_index['ts_code']:
        print(ts_code)
        data = pd.read_sql("select * from stock_all where ts_code='{}'".format(ts_code), conn)
        if len(data) < 2:
            continue
        data['trade_date'] = pd.to_datetime(data['trade_date'])
        # monthly_data = data.resample('M', on="trade_date").last()
        logic = {'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'vol': 'sum', 'amount': 'sum','turnover_rate':'sum'}
        monthly_data = data.resample(rule=freq, closed='left', on="trade_date").apply(logic)
        monthly_data.dropna(inplace=True)
        if len(monthly_data) < 2:
            continue
        monthly_data.reset_index(inplace=True)
        monthly_data = monthly_data.iloc[:-1]
        monthly_data['ts_code'] = ts_code
        monthly_data['trade_date'] = monthly_data['trade_date'].apply(lambda x: x.strftime('%Y%m%d'))
        print(monthly_data.head())
        monthly_data.to_sql(table_name, conn, if_exists='append', index=False)


def dropTable(table_name):
    conn = sqlite3.connect(r"D:\stock\stock.db")
    cursor = conn.cursor()
    cursor.execute("DROP TABLE IF EXISTS {}".format(table_name))
    # cursor.execute("DROP TABLE IF EXISTS stock_all")
    conn.commit()
    cursor.close()
    conn.close()


def downLoadData():
    # 创建sqlite3的股票列表数据表，如果不存在则创建
    conn = sqlite3.connect(r"D:\stock\stock.db")
    # 查询当前所有正常上市交易的股票列表
    data = pro.stock_basic(exchange='', list_status='L', fields='ts_code,symbol,name,area,industry,list_date')
    # 将股票列表数据存入sqlite3的股票列表数据表，如果存在则替换
    data.to_sql('share_index', conn, if_exists='replace', index=False)
    print(data)
    stock_pool = data['ts_code']
    # 循环获取单个股票的日线行情
    is_get = True
    for i in tqdm(range(len(stock_pool))):
        if is_get:
            print("正在更新日线：", i)
            print(stock_pool[i])
            try:
                df = ts.pro_bar(ts_code=stock_pool[i],adj='qfq', freq='D', asset='E',factors=['tor', ])
                print(df.head())
                df.to_sql('stock_all', conn, if_exists='append', index=False)
                time.sleep(0.121)
            except Exception as aa:
                print(aa)
                print('No DATA Code: ' + str(i))
                continue
        else:
            print("pass:{}".format(stock_pool[i]))
        # if stock_pool[i] == "000925.SZ":
        #     is_get = True
    print('All Finished!')


if __name__ == '__main__':
    # downLoadData()
    transforme_data(freq="ME",table_name="stock_monthly")
    # transforme_data(freq="W",table_name="weekly_stock_monthly")
    # dropTable("stock_monthly")
    # conn = sqlite3.connect(r"D:\stock\stock.db")
    # df = pd.read_sql("select * from stock_monthly limit 10", conn)
    # print(df)
    # cursor = conn.cursor()
    # df = pd.read_sql("select * from stock_all where ts_code='{}'".format('000925.SZ'), conn)
    # print(df)
    # print(np.unique(df["ts_code"]))
    # for trade_date in np.unique(df["trade_date"]):
    #     print(trade_date)
    #     if "-" in trade_date:
    #         print(trade_date)
    #         new_trade_date = datetime.datetime.strptime(trade_date, "%Y-%m-%d").strftime('%Y%m%d')
    #         sql = '''UPDATE stock_all SET trade_date = '{}' WHERE trade_date = '{}';'''.format(new_trade_date,trade_date)
    #         cursor.execute(sql)
    # df_new = pd.read_sql("select trade_date from stock_all", conn)
    # print(np.unique(df_new["trade_date"]))
    # cursor.close()
    # conn.execute("VACUUM")
    # conn.close()
