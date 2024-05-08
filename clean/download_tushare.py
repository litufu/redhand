import time, datetime, sqlite3, logging
from tqdm import tqdm
import pandas as pd
import numpy as np
from constants import pro


def downLoadData(pro, start_dt='20100101'):
    # 查询当前所有正常上市交易的股票列表
    data = pro.stock_basic(exchange='', list_status='L', fields='ts_code,symbol,name,area,industry,list_date')
    # 创建sqlite3的股票列表数据表，如果不存在则创建
    conn = sqlite3.connect(r"D:\redhand\clean\data\tushare_db\tushare.db")
    cursor = conn.cursor()
    try:
        cursor.execute("DROP TABLE share_index")
    except:
        pass
    sql = "CREATE TABLE if not exists share_index(ts_code,symbol,name,area,industry,list_date)"
    cursor.execute(sql)
    # 将股票列表数据存入sqlite3的股票列表数据表，如果存在则替换
    data.to_sql('share_index', conn, if_exists='replace', index=False)
    conn.commit()
    print(data)
    # 创建sqlite3的股票日线行情数据表，如果不存在则创建
    # amount:成交额，千元； vol：成交手数，百股
    sql = "CREATE TABLE if not exists stock_all(ts_code text,trade_date text,open real,high real,low real,close real,pre_close real,\
        change real,pct_chg real,vol real,amount real,UNIQUE(ts_code, trade_date))"
    cursor.execute(sql)
    # 设定获取日线行情的初始日期和终止日期，其中终止日期设定为昨天。

    time_temp = datetime.datetime.now() - datetime.timedelta(days=1)
    end_dt = time_temp.strftime('%Y%m%d')
    stock_pool = data['ts_code']
    total = len(stock_pool)
    # 循环获取单个股票的日线行情
    is_get = False
    for i in tqdm(range(len(stock_pool))):
        if is_get:
            print("正在更新日线：", i)
            try:
                df = pro.daily(ts_code=stock_pool[i], start_date=start_dt, end_date=end_dt)
                time.sleep(0.121)
                print(stock_pool[i])
                # print(df.head())
                # 打印进度
                info = 'Seq: ' + str(i + 1) + ' of ' + str(total) + '   Code: ' + str(stock_pool[i])
                c_len = df.shape[0]
                print("c_len", c_len)
            except Exception as aa:
                print(aa)
                print('No DATA Code: ' + str(i))
                continue

            for j in range(c_len):
                resu0 = list(df.iloc[c_len - 1 - j])
                print("resu0", resu0)
                resu = []
                for k in range(len(resu0)):
                    if str(resu0[k]) == 'nan':
                        resu.append(-1)
                    else:
                        resu.append(resu0[k])
                        # print("resu0[k]", resu0[k])

                # state_dt = (datetime.datetime.strptime(resu[1], "%Y%m%d")).strftime('%Y-%m-%d')
                # print('state_dt', state_dt)
                sql = "insert or ignore into stock_all(trade_date,ts_code,open,close,high,low,vol,amount,pre_close,change,pct_chg) VALUES ('%s', '%s', '%.2f', '%.2f','%.2f','%.2f','%i','%.2f','%.2f','%.2f','%.2f')" % (
                    str(resu[1]), str(resu[0]), float(resu[2]), float(resu[5]), float(resu[3]), float(resu[4]),
                    float(resu[9]),
                    float(resu[10]), float(resu[6]), float(resu[7]), float(resu[8]))

                try:
                    cursor.execute(sql)
                    conn.commit()
                    logging.info(info)
                except Exception as err:
                    print(err)
        else:
            print("pass:{}".format(stock_pool[i]))
        if stock_pool[i] == "000925.SZ":
            is_get = True
    cursor.close()
    conn.execute("VACUUM")
    conn.close()
    print('All Finished!')


if __name__ == '__main__':
    start_dt = '20100101'
    downLoadData(pro,start_dt)
    # conn = sqlite3.connect(r"D:\redhand\clean\data\tushare_db\tushare.db")
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
