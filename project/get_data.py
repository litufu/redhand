import tushare as ts
import pandas as pd

ts.set_token('f88e93f91c79cdb865f22f40cac23a2907da36b53fa9aa150228ed27')
pro = ts.pro_api()




def get_stock_basic(end_date,list_days):
    '''
    获取截止日超过已上市天数的非北交所股票列表
    :param end_date: 截止日期
    :param list_days: 已上市天数
    :return:
    '''
    df_stock_basic = pro.stock_basic(exchange='', list_status='L', fields='ts_code,market,symbol,name,area,industry,list_date')
    df_stock_basic["list_duaration"] = end_date - df_stock_basic["list_date"].astype(int)
    # 上市超过LIST_DAYS
    df_stock_basic = df_stock_basic[df_stock_basic["list_duaration"]>list_days]
    # 不包含北交所，北交所有流动性问题
    df_stock_basic = df_stock_basic[df_stock_basic["market"]!="北交所"]
    return df_stock_basic


def get_list_date(ts_code,df_stock_basic):
    '''
    获取股票的上市日期
    :param ts_code: 股票代码
    :param df_stock_basic: 股票基本信息
    :return:
    '''

    df_stock_basic = df_stock_basic[df_stock_basic["ts_code"]==ts_code]
    list_date = df_stock_basic["list_date"].values[0]
    return list_date


def get_one_stock_data(ts_code,df_index,start_date,end_date,is_merge_index=True):
    '''

    :param ts_code: 股票代码
    :param df_index: 交易数据
    :param start_date: 股票开始日期
    :param start_date: 股票结束日期
    :param end_date: 股票结束日期
    :param is_merge_index：是否合并指数
    :return:合并后的数据
    '''
    df_stock = pro.daily(ts_code=ts_code, start_date=start_date, end_date=end_date)
    df_stock["trade_date"] = df_stock["trade_date"].astype(int)
    if is_merge_index:
        df_merge = df_stock.merge(df_index, how="left", suffixes=["_stock", "_index"], on="trade_date")
    else:
        df_merge = df_stock.copy(deep=True)
    return df_merge






