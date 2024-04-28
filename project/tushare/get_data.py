import pandas as pd
import tushare as ts

ts.set_token('f88e93f91c79cdb865f22f40cac23a2907da36b53fa9aa150228ed27')
pro = ts.pro_api()

data_basic = pro.stock_basic(exchange='', list_status='L', fields='ts_code,symbol,name,area,industry,list_date')
print(data_basic)
data_basic.to_csv("data_basic.csv",index=False)
#
# # 1.获取股票数据
#
# df_stock = pro.daily(ts_code='000001.SZ', start_date='20180701', end_date='20231231')
# df_stock["trade_date"] = df_stock["trade_date"].astype(int)
# df_stock["ma5"] = df_stock['close'].rolling(5).mean()
# df_stock["ma10"] = df_stock['close'].rolling(10).mean()
# df_stock["ma30"] = df_stock['close'].rolling(30).mean()
# df_stock["ma60"] = df_stock['close'].rolling(60).mean()
# df_stock.to_csv("df_stock.csv")
#
# # 2.获取指数数据
# # 获取上证指数
# df_sh = pro.index_daily(ts_code='000001.SH')
# df_sh["trade_date"] = df_sh["trade_date"].astype(int)
# df_sh["ma5"] = df_sh['close'].rolling(5).mean()
# df_sh["ma10"] = df_sh['close'].rolling(10).mean()
# df_sh["ma30"] = df_sh['close'].rolling(30).mean()
# df_sh["ma60"] = df_sh['close'].rolling(60).mean()
# # df_sh.to_csv("df_sh.csv")
# # print(df_sh)
# # 获取深证成指
# df_sz = pro.index_daily(ts_code='399001.SZ')
# df_sz["trade_date"] = df_sz["trade_date"].astype(int)
# # df_sz["ma5"] = df_sz['close'].rolling(5).mean()
# # df_sz["ma10"] = df_sz['close'].rolling(10).mean()
# # df_sz["ma30"] = df_sz['close'].rolling(30).mean()
# # df_sz["ma60"] = df_sz['close'].rolling(60).mean()
# # print(df_sz)
#
# # 3.获取行业数据
# # df = pro.ths_index(exchange="A",type="N")
# # df.to_csv("同花顺行业.csv")
# # print(df)
# # 获取大盘指数
# # df = pro.index_basic()
# # df.to_csv("index.csv")
#
# # 4.数据拼接
# # 合并深证和上证指数
# df_index = df_sh.merge(df_sz,how="left", suffixes=["_sh_index", "_sz_index"],on="trade_date")
# df_index.to_csv("index.csv",index=False)
# # 合并指数和股票
# df_merge = df_stock.merge(df_index,how="left", suffixes=["_stock", "_index"],on="trade_date")
# df_merge.to_csv("merge.csv",index=False)
