import os
import sys
sys.path.append(os.getcwd())

from MYTT import MyTT
import pandas as pd

def indicatior(df):
    CLOSE = df["close"].values
    OPEN = df["open"].values
    HIGH = df["high"].values
    LOW = df["low"].values
    VOL = df["vol"].values
    # 1.MACD
    DIF, DEA, MACD = MyTT.MACD(CLOSE)

    # 2.KDJ
    K, D, J = MyTT.KDJ(CLOSE, HIGH, LOW)

    # 3.RSI
    RSI = MyTT.RSI(CLOSE)

    # 4.WR
    WR,WR1 = MyTT.WR(CLOSE, HIGH, LOW)

    # 5.BIAS
    BIAS1, BIAS2, BIAS3 = MyTT.BIAS(CLOSE)

    # 6.BOLL
    BOLL_UPPER, BOLL_MID, BOLL_LOWER = MyTT.BOLL(CLOSE)

    # 7.PSY
    PSY, PSYMA = MyTT.PSY(CLOSE)

    # 8.CCI
    CCI = MyTT.CCI(CLOSE, HIGH, LOW)

    # 9.ATR
    ATR = MyTT.ATR(CLOSE, HIGH, LOW)

    # 10.BBI
    BBI = MyTT.BBI(CLOSE)

    # 11.DMI
    PDI, MDI, ADX, ADXR = MyTT.DMI(CLOSE, HIGH, LOW)

    # 12.TAQ
    TAQ_UPPER, TAQ_MID, TAQ_LOWER = MyTT.TAQ(HIGH, LOW, 20)

    # 13.KTN
    KTN_UPPER, KTN_MID, KTN_LOWER = MyTT.KTN(CLOSE, HIGH, LOW)

    # 14.TRIX
    TRIX, TRMA = MyTT.TRIX(CLOSE)

    # 15.VR
    VR = MyTT.VR(CLOSE, VOL)

    # 16.CR
    CR = MyTT.CR(CLOSE, HIGH, LOW)

    # 17.EMV
    EMV, MAEMV = MyTT.EMV(HIGH, LOW, VOL)

    # 18.DPO
    DPO, MADPO = MyTT.DPO(CLOSE)

    # 19.BRAR
    AR, BR = MyTT.BRAR(OPEN, CLOSE, HIGH, LOW)

    # 20.DFMA
    _, DFMA = MyTT.DFMA(CLOSE)

    # 21.MTM
    MTM, MTMMA = MyTT.MTM(CLOSE)

    # 22.MASS
    MASS, MA_MASS = MyTT.MASS(HIGH, LOW)

    # 23.ROC
    ROC, MAROC = MyTT.ROC(CLOSE)

    # 24.EXPMA
    EXPMA_SHORT, EXPMA_LONG = MyTT.EXPMA(CLOSE)

    # 25.OBV
    OBV = MyTT.OBV(CLOSE, VOL)

    # 26.MFI
    MFI = MyTT.MFI(CLOSE, HIGH, LOW, VOL)

    # 27.ASI
    ASI, ASIT = MyTT.ASI(OPEN, CLOSE, HIGH, LOW)

    # 28.XSII
    TD1, TD2, TD3, TD4 = MyTT.XSII(CLOSE, HIGH, LOW)

    res = {
        "DIF":DIF,
        "DEA":DEA,
        "MACD":MACD,
        "K":K,
        "D":D,
        "J":J,
        "RSI":RSI,
        "WR":WR,
        "WR1":WR1,
        "BIAS1":BIAS1,
        "BIAS2":BIAS2,
        "BIAS3":BIAS3,
        "BOLL_UPPER": BOLL_UPPER,
        "BOLL_MID": BOLL_MID,
        "BOLL_LOWER": BOLL_LOWER,
        "PSY": PSY,
        "PSYMA": PSYMA,
        "CCI":CCI,
        "ATR": ATR,
        "BBI": BBI,
        "PDI": PDI,
        "MDI": MDI,
        "ADX": ADX,
        "ADXR": ADXR,
        "TAQ_UPPER": TAQ_UPPER,
        "TAQ_MID": TAQ_MID,
        "TAQ_LOWER": TAQ_LOWER,
        "KTN_UPPER": KTN_UPPER,
        "KTN_MID": KTN_MID,
        "KTN_LOWER": KTN_LOWER,
        "TRIX": TRIX,
        "TRMA": TRMA,
        "VR": VR,
        "CR": CR,
        "EMV": EMV,
        "MAEMV": MAEMV,
        "DPO": DPO,
        "MADPO": MADPO,
        "AR": AR,
        "BR": BR,
        "DFMA": DFMA,
        "MTM": MTM,
        "MTMMA": MTMMA,
        "MASS": MASS,
        "MA_MASS": MA_MASS,
        "ROC": ROC,
        "MAROC": MAROC,
        "EXPMA_SHORT": EXPMA_SHORT,
        "EXPMA_LONG": EXPMA_LONG,
        "OBV": OBV,
        "MFI": MFI,
        "ASI": ASI,
        "ASIT": ASIT,
        "TD1": TD1,
        "TD2": TD2,
        "TD3": TD3,
        "TD4": TD4,
    }
    df_indicatior = pd.DataFrame(res)
    df = pd.concat([df,df_indicatior],axis=1).copy()

    # 29.均线
    df["ma5"] = df['close'].rolling(5).mean()
    df["ma10"] = df['close'].rolling(10).mean()
    df["ma30"] = df['close'].rolling(30).mean()
    df["ma60"] = df['close'].rolling(60).mean()

    return df





if __name__ == '__main__':
    df = pd.read_csv(r"D:\redhand\project\tushare\df_stock.csv")
    df = indicatior(df)
    print(df)
    for column in df.columns:
        print("\"{}\":{},".format(column,column))
    # print(df.columns)














