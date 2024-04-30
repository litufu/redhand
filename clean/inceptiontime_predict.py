import keras
import numpy as np
import datetime
from sklearn.utils import check_random_state
from constants import pro,LIST_DAYS
from download_data import get_df_index,get_stock_basic,get_one_stock_data,handle_stock_df

# model_save_path = r"D:\redhand\clean\data\state_dict\inceptiontime.keras"
# model_ = keras.saving.load_model(model_save_path)


def predict(kinds, x):
    probs = predict_proba(x)
    rng = check_random_state(None)
    return np.array(
        [
            kinds[int(rng.choice(np.flatnonzero(prob == prob.max())))]
            for prob in probs
        ]
    )


def predict_proba(model,X,batch_size):
    X = X.transpose((0, 2, 1))
    probs = model.predict(X, batch_size)
    probs = probs / probs.sum(axis=1, keepdims=1)
    return probs


def get_date_before_days(current_date_str,before_days):
    current_date = datetime.datetime.strptime(current_date_str, "%Y%m%d")
    prev_date = current_date - datetime.timedelta(days=before_days)
    date_str = prev_date.strftime("%Y%m%d")
    return int(date_str)

# 预测步骤
#
# 一、获取当日数据
trade_date = "20240102"
END_DATE = int(trade_date)
#start_date:一年前
start_date = get_date_before_days(trade_date,365)
print(start_date)
is_merge_index = True
fh = 15


# 当日股票数据
# df_stock = pro.daily(trade_date=END_DATE)
# print(df_stock)
# 计算数据指标
#
#
# 对今天数据进行预测
df_index = get_df_index()
df_stock_basic = get_stock_basic(END_DATE, LIST_DAYS)
for key, ts_code in enumerate(list(df_stock_basic["ts_code"])):
    print("正在处理{}".format(ts_code))
    # 获取股票和指数交易日数据，截止日前一年的数据
    df_stock = get_one_stock_data(ts_code, df_index, start_date, END_DATE, is_merge_index)
    # 数据处理
    df_data = handle_stock_df(df_stock, fh, is_merge_index)
    df_data.to_csv("{}.csv".format(ts_code))
    break