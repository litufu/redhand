import time

import numpy as np
import pandas as pd
import tushare as ts
import sqlite3
from tsai.all import *
from torch.utils.data import Dataset
from sklearn.preprocessing import OneHotEncoder
from stock_dataset_new import categorize,get_one_stock_data_from_sqlite,split_data,find_loc,new_fit
from constants import TOTAL_TRAIN_DATA_LENGTH, TOTAL_TEST_DATA_LENGTH, X_TRAIN_PATH, Y_TRAIN_PATH, X_VALID_PATH, \
    Y_VALID_PATH

ts.set_token('f88e93f91c79cdb865f22f40cac23a2907da36b53fa9aa150228ed27')
pro = ts.pro_api()

# 资产负债表
# 名称	类型	默认显示	描述
# ts_code	str	Y	TS股票代码
# ann_date	str	Y	公告日期
# f_ann_date	str	Y	实际公告日期
# end_date	str	Y	报告期
# report_type	str	Y	报表类型
# comp_type	str	Y	公司类型(1一般工商业2银行3保险4证券)
# end_type	str	Y	报告期类型
# total_share	float	Y	期末总股本
# cap_rese	float	Y	资本公积金
# undistr_porfit	float	Y	未分配利润
# surplus_rese	float	Y	盈余公积金
# special_rese	float	Y	专项储备
# money_cap	float	Y	货币资金
# trad_asset	float	Y	交易性金融资产
# notes_receiv	float	Y	应收票据
# accounts_receiv	float	Y	应收账款
# oth_receiv	float	Y	其他应收款
# prepayment	float	Y	预付款项
# div_receiv	float	Y	应收股利
# int_receiv	float	Y	应收利息
# inventories	float	Y	存货
# amor_exp	float	Y	待摊费用
# nca_within_1y	float	Y	一年内到期的非流动资产
# sett_rsrv	float	Y	结算备付金
# loanto_oth_bank_fi	float	Y	拆出资金
# premium_receiv	float	Y	应收保费
# reinsur_receiv	float	Y	应收分保账款
# reinsur_res_receiv	float	Y	应收分保合同准备金
# pur_resale_fa	float	Y	买入返售金融资产
# oth_cur_assets	float	Y	其他流动资产
# total_cur_assets	float	Y	流动资产合计
# fa_avail_for_sale	float	Y	可供出售金融资产
# htm_invest	float	Y	持有至到期投资
# lt_eqt_invest	float	Y	长期股权投资
# invest_real_estate	float	Y	投资性房地产
# time_deposits	float	Y	定期存款
# oth_assets	float	Y	其他资产
# lt_rec	float	Y	长期应收款
# fix_assets	float	Y	固定资产
# cip	float	Y	在建工程
# const_materials	float	Y	工程物资
# fixed_assets_disp	float	Y	固定资产清理
# produc_bio_assets	float	Y	生产性生物资产
# oil_and_gas_assets	float	Y	油气资产
# intan_assets	float	Y	无形资产
# r_and_d	float	Y	研发支出
# goodwill	float	Y	商誉
# lt_amor_exp	float	Y	长期待摊费用
# defer_tax_assets	float	Y	递延所得税资产
# decr_in_disbur	float	Y	发放贷款及垫款
# oth_nca	float	Y	其他非流动资产
# total_nca	float	Y	非流动资产合计
# cash_reser_cb	float	Y	现金及存放中央银行款项
# depos_in_oth_bfi	float	Y	存放同业和其它金融机构款项
# prec_metals	float	Y	贵金属
# deriv_assets	float	Y	衍生金融资产
# rr_reins_une_prem	float	Y	应收分保未到期责任准备金
# rr_reins_outstd_cla	float	Y	应收分保未决赔款准备金
# rr_reins_lins_liab	float	Y	应收分保寿险责任准备金
# rr_reins_lthins_liab	float	Y	应收分保长期健康险责任准备金
# refund_depos	float	Y	存出保证金
# ph_pledge_loans	float	Y	保户质押贷款
# refund_cap_depos	float	Y	存出资本保证金
# indep_acct_assets	float	Y	独立账户资产
# client_depos	float	Y	其中：客户资金存款
# client_prov	float	Y	其中：客户备付金
# transac_seat_fee	float	Y	其中:交易席位费
# invest_as_receiv	float	Y	应收款项类投资
# total_assets	float	Y	资产总计
# lt_borr	float	Y	长期借款
# st_borr	float	Y	短期借款
# cb_borr	float	Y	向中央银行借款
# depos_ib_deposits	float	Y	吸收存款及同业存放
# loan_oth_bank	float	Y	拆入资金
# trading_fl	float	Y	交易性金融负债
# notes_payable	float	Y	应付票据
# acct_payable	float	Y	应付账款
# adv_receipts	float	Y	预收款项
# sold_for_repur_fa	float	Y	卖出回购金融资产款
# comm_payable	float	Y	应付手续费及佣金
# payroll_payable	float	Y	应付职工薪酬
# taxes_payable	float	Y	应交税费
# int_payable	float	Y	应付利息
# div_payable	float	Y	应付股利
# oth_payable	float	Y	其他应付款
# acc_exp	float	Y	预提费用
# deferred_inc	float	Y	递延收益
# st_bonds_payable	float	Y	应付短期债券
# payable_to_reinsurer	float	Y	应付分保账款
# rsrv_insur_cont	float	Y	保险合同准备金
# acting_trading_sec	float	Y	代理买卖证券款
# acting_uw_sec	float	Y	代理承销证券款
# non_cur_liab_due_1y	float	Y	一年内到期的非流动负债
# oth_cur_liab	float	Y	其他流动负债
# total_cur_liab	float	Y	流动负债合计
# bond_payable	float	Y	应付债券
# lt_payable	float	Y	长期应付款
# specific_payables	float	Y	专项应付款
# estimated_liab	float	Y	预计负债
# defer_tax_liab	float	Y	递延所得税负债
# defer_inc_non_cur_liab	float	Y	递延收益-非流动负债
# oth_ncl	float	Y	其他非流动负债
# total_ncl	float	Y	非流动负债合计
# depos_oth_bfi	float	Y	同业和其它金融机构存放款项
# deriv_liab	float	Y	衍生金融负债
# depos	float	Y	吸收存款
# agency_bus_liab	float	Y	代理业务负债
# oth_liab	float	Y	其他负债
# prem_receiv_adva	float	Y	预收保费
# depos_received	float	Y	存入保证金
# ph_invest	float	Y	保户储金及投资款
# reser_une_prem	float	Y	未到期责任准备金
# reser_outstd_claims	float	Y	未决赔款准备金
# reser_lins_liab	float	Y	寿险责任准备金
# reser_lthins_liab	float	Y	长期健康险责任准备金
# indept_acc_liab	float	Y	独立账户负债
# pledge_borr	float	Y	其中:质押借款
# indem_payable	float	Y	应付赔付款
# policy_div_payable	float	Y	应付保单红利
# total_liab	float	Y	负债合计
# treasury_share	float	Y	减:库存股
# ordin_risk_reser	float	Y	一般风险准备
# forex_differ	float	Y	外币报表折算差额
# invest_loss_unconf	float	Y	未确认的投资损失
# minority_int	float	Y	少数股东权益
# total_hldr_eqy_exc_min_int	float	Y	股东权益合计(不含少数股东权益)
# total_hldr_eqy_inc_min_int	float	Y	股东权益合计(含少数股东权益)
# total_liab_hldr_eqy	float	Y	负债及股东权益总计
# lt_payroll_payable	float	Y	长期应付职工薪酬
# oth_comp_income	float	Y	其他综合收益
# oth_eqt_tools	float	Y	其他权益工具
# oth_eqt_tools_p_shr	float	Y	其他权益工具(优先股)
# lending_funds	float	Y	融出资金
# acc_receivable	float	Y	应收款项
# st_fin_payable	float	Y	应付短期融资款
# payables	float	Y	应付款项
# hfs_assets	float	Y	持有待售的资产
# hfs_sales	float	Y	持有待售的负债
# cost_fin_assets	float	Y	以摊余成本计量的金融资产
# fair_value_fin_assets	float	Y	以公允价值计量且其变动计入其他综合收益的金融资产
# cip_total	float	Y	在建工程(合计)(元)
# oth_pay_total	float	Y	其他应付款(合计)(元)
# long_pay_total	float	Y	长期应付款(合计)(元)
# debt_invest	float	Y	债权投资(元)
# oth_debt_invest	float	Y	其他债权投资(元)
# oth_eq_invest	float	N	其他权益工具投资(元)
# oth_illiq_fin_assets	float	N	其他非流动金融资产(元)
# oth_eq_ppbond	float	N	其他权益工具:永续债(元)
# receiv_financing	float	N	应收款项融资
# use_right_assets	float	N	使用权资产
# lease_liab	float	N	租赁负债
# contract_assets	float	Y	合同资产
# contract_liab	float	Y	合同负债
# accounts_receiv_bill	float	Y	应收票据及应收账款
# accounts_pay	float	Y	应付票据及应付账款
# oth_rcv_total	float	Y	其他应收款(合计)（元）
# fix_assets_total	float	Y	固定资产(合计)(元)
# update_flag	str	Y	更新标识
# 现金流量表
no_cash = ["ts_code","ann_date","f_ann_date","comp_type","report_type","end_type","net_profit", "finan_exp", "free_cashflow", "uncon_invest_loss", "prov_depr_assets", "depr_fa_coga_dpba",
           "amort_intang_assets",
           "lt_amort_deferred_exp", "decr_deferred_exp", "incr_acc_exp", "loss_disp_fiolta", "loss_scr_fa",
           "loss_fv_chg",
           "invest_loss", "decr_def_inc_tax_assets", "incr_def_inc_tax_liab", "decr_inventories", "decr_oper_payable",
           "incr_oper_payable",
           "others", "im_net_cashflow_oper_act", "conv_debt_into_cap", "conv_copbonds_due_within_1y", "fa_fnc_leases",
           "im_n_incr_cash_equ", "credit_impa_loss", "use_right_asset_dep", "oth_loss_asset", "end_bal_cash",
           "beg_bal_cash",
           "end_bal_cash_equ", "beg_bal_cash_equ"]
# 利润表
no_profit = ["ts_code","ann_date","f_ann_date","comp_type","report_type","end_type","ebit", "ebitda", "undist_profit", "distable_profit", "transfer_surplus_rese", "transfer_housing_imprest",
             "transfer_oth", "adj_lossgain", "withdra_legal_surplus", "withdra_legal_pubfund", "withdra_biz_devfund",
             "withdra_rese_fund", "withdra_oth_ersu", "workers_welfare", "distr_profit_shrhder", "prfshare_payable_dvd",
             "comshare_payable_dvd", "capit_comstock_div"]
# 指标
no_indicator = ["ts_code","ann_date","extra_item", "profit_dedt", "gross_margin", "op_income",
                "ebit", "ebitda", "fcff", "fcfe", "current_exint", "noncurrent_exint", "interestdebt", "netdebt",
                "tangible_asset",
                "working_capital", "networking_capital", "invest_capital", "retained_earnings", "diluted2_eps",
                "netprofit_margin",
                "cogs_of_sales", "expense_of_sales", "profit_to_gr", "saleexp_to_gr", "adminexp_of_gr", "finaexp_of_gr",
                "impai_ttm",
                "gc_of_gr", "op_of_gr", "ebit_of_gr", "debt_to_assets", "ca_to_assets", "nca_to_assets", "turn_days",
                "fixed_assets","ocfps","retainedps","cfps","ebit_ps","fcff_ps","fcfe_ps",
                "profit_to_op", "total_revenue_ps","revenue_ps","capital_rese_ps","surplus_rese_ps","undist_profit_ps",


                ]

cash = ['c_fr_sale_sg', 'recp_tax_rends', 'n_depos_incr_fi', 'n_incr_loans_cb', 'n_inc_borr_oth_fi',
        'prem_fr_orig_contr', 'n_incr_insured_dep', 'n_reinsur_prem', 'n_incr_disp_tfa', 'ifc_cash_incr',
        'n_incr_disp_faas', 'n_incr_loans_oth_bank', 'n_cap_incr_repur', 'c_fr_oth_operate_a', 'c_inf_fr_operate_a',
        'c_paid_goods_s', 'c_paid_to_for_empl', 'c_paid_for_taxes', 'n_incr_clt_loan_adv', 'n_incr_dep_cbob',
        'c_pay_claims_orig_inco', 'pay_handling_chrg', 'pay_comm_insur_plcy', 'oth_cash_pay_oper_act',
        'st_cash_out_act', 'n_cashflow_act', 'oth_recp_ral_inv_act', 'c_disp_withdrwl_invest', 'c_recp_return_invest',
        'n_recp_disp_fiolta', 'n_recp_disp_sobu', 'stot_inflows_inv_act', 'c_pay_acq_const_fiolta', 'c_paid_invest',
        'n_disp_subs_oth_biz', 'oth_pay_ral_inv_act', 'n_incr_pledge_loan', 'stot_out_inv_act', 'n_cashflow_inv_act',
        'c_recp_borrow', 'proc_issue_bonds', 'oth_cash_recp_ral_fnc_act', 'stot_cash_in_fnc_act', 'c_prepay_amt_borr',
        'c_pay_dist_dpcp_int_exp', 'incl_dvd_profit_paid_sc_ms', 'oth_cashpay_ral_fnc_act', 'stot_cashout_fnc_act',
        'n_cash_flows_fnc_act', 'eff_fx_flu_cash', 'n_incr_cash_cash_equ', 'c_cash_equ_beg_period',
        'c_cash_equ_end_period', 'c_recp_cap_contrib', 'incl_cash_rec_saims', 'net_dism_capital_add',
        'net_cash_rece_sec']
profit = ['total_revenue', 'revenue', 'int_income', 'prem_earned', 'comm_income', 'n_commis_income', 'n_oth_income',
          'n_oth_b_income', 'prem_income', 'out_prem', 'une_prem_reser', 'reins_income', 'n_sec_tb_income',
          'n_sec_uw_income', 'n_asset_mg_income', 'oth_b_income', 'fv_value_chg_gain', 'invest_income',
          'ass_invest_income', 'forex_gain', 'total_cogs', 'oper_cost', 'int_exp', 'comm_exp', 'biz_tax_surchg',
          'sell_exp', 'admin_exp', 'fin_exp', 'assets_impair_loss', 'prem_refund', 'compens_payout', 'reser_insur_liab',
          'div_payt', 'reins_exp', 'oper_exp', 'compens_payout_refu', 'insur_reser_refu', 'reins_cost_refund',
          'other_bus_cost', 'operate_profit', 'non_oper_income', 'non_oper_exp', 'nca_disploss', 'total_profit',
          'income_tax', 'n_income', 'n_income_attr_p', 'minority_gain', 'oth_compr_income', 't_compr_income',
          'compr_inc_attr_p', 'compr_inc_attr_m_s', 'insurance_exp', 'rd_exp', 'fin_exp_int_exp', 'fin_exp_int_inc',
          'continued_net_profit']
balance = ['total_share', 'cap_rese', 'undistr_porfit', 'surplus_rese', 'special_rese', 'money_cap', 'trad_asset',
           'notes_receiv', 'accounts_receiv', 'oth_receiv', 'prepayment', 'div_receiv', 'int_receiv', 'inventories',
           'amor_exp', 'nca_within_1y', 'sett_rsrv', 'loanto_oth_bank_fi', 'premium_receiv', 'reinsur_receiv',
           'reinsur_res_receiv', 'pur_resale_fa', 'oth_cur_assets', 'total_cur_assets', 'fa_avail_for_sale',
           'htm_invest', 'lt_eqt_invest', 'invest_real_estate', 'time_deposits', 'oth_assets', 'lt_rec', 'fix_assets',
           'cip', 'const_materials', 'fixed_assets_disp', 'produc_bio_assets', 'oil_and_gas_assets', 'intan_assets',
           'r_and_d', 'goodwill', 'lt_amor_exp', 'defer_tax_assets', 'decr_in_disbur', 'oth_nca', 'total_nca',
           'cash_reser_cb', 'depos_in_oth_bfi', 'prec_metals', 'deriv_assets', 'rr_reins_une_prem',
           'rr_reins_outstd_cla', 'rr_reins_lins_liab', 'rr_reins_lthins_liab', 'refund_depos', 'ph_pledge_loans',
           'refund_cap_depos', 'indep_acct_assets', 'client_depos', 'client_prov', 'transac_seat_fee',
           'invest_as_receiv', 'total_assets', 'lt_borr', 'st_borr', 'cb_borr', 'depos_ib_deposits', 'loan_oth_bank',
           'trading_fl', 'notes_payable', 'acct_payable', 'adv_receipts', 'sold_for_repur_fa', 'comm_payable',
           'payroll_payable', 'taxes_payable', 'int_payable', 'div_payable', 'oth_payable', 'acc_exp', 'deferred_inc',
           'st_bonds_payable', 'payable_to_reinsurer', 'rsrv_insur_cont', 'acting_trading_sec', 'acting_uw_sec',
           'non_cur_liab_due_1y', 'oth_cur_liab', 'total_cur_liab', 'bond_payable', 'lt_payable', 'specific_payables',
           'estimated_liab', 'defer_tax_liab', 'defer_inc_non_cur_liab', 'oth_ncl', 'total_ncl', 'depos_oth_bfi',
           'deriv_liab', 'depos', 'agency_bus_liab', 'oth_liab', 'prem_receiv_adva', 'depos_received', 'ph_invest',
           'reser_une_prem', 'reser_outstd_claims', 'reser_lins_liab', 'reser_lthins_liab', 'indept_acc_liab',
           'pledge_borr', 'indem_payable', 'policy_div_payable', 'total_liab', 'treasury_share', 'ordin_risk_reser',
           'forex_differ', 'invest_loss_unconf', 'minority_int', 'total_hldr_eqy_exc_min_int',
           'total_hldr_eqy_inc_min_int', 'total_liab_hldr_eqy', 'lt_payroll_payable', 'oth_comp_income',
           'oth_eqt_tools', 'oth_eqt_tools_p_shr', 'lending_funds', 'acc_receivable', 'st_fin_payable', 'payables',
           'hfs_assets', 'hfs_sales', 'cost_fin_assets', 'fair_value_fin_assets', 'contract_assets', 'contract_liab',
           'accounts_receiv_bill', 'accounts_pay', 'oth_rcv_total', 'fix_assets_total', 'cip_total', 'oth_pay_total',
           'long_pay_total', 'debt_invest', 'oth_debt_invest']
indicator = ['current_ratio', 'quick_ratio', 'cash_ratio', 'ar_turn', 'ca_turn', 'fa_turn',
             'assets_turn', 'grossprofit_margin',
             'roe', 'roe_waa', 'roe_dt', 'roa', 'npta', 'roic', 'roe_yearly', 'roa2_yearly', 'assets_to_eqt',
             'dp_assets_to_eqt', 'tbassets_to_totalassets', 'int_to_talcap', 'eqt_to_talcapital', 'currentdebt_to_debt',
             'longdeb_to_debt', 'ocf_to_shortdebt', 'debt_to_eqt', 'eqt_to_debt', 'eqt_to_interestdebt',
             'tangibleasset_to_debt', 'tangasset_to_intdebt', 'tangibleasset_to_netdebt', 'ocf_to_debt', 'roa_yearly',
             'roa_dp', 'q_saleexp_to_gr', 'q_gc_to_gr', 'q_roe', 'q_dt_roe', 'q_npta', 'q_ocf_to_sales',
             'basic_eps_yoy', 'dt_eps_yoy', 'cfps_yoy', 'op_yoy', 'ebt_yoy', 'netprofit_yoy', 'dt_netprofit_yoy',
             'ocf_yoy', 'roe_yoy', 'bps_yoy', 'assets_yoy', 'eqt_yoy', 'tr_yoy', 'or_yoy', 'q_sales_yoy', 'q_op_qoq',
             'equity_yoy']

old_x_fields = [*balance,*profit,*cash,"assets_turn"]

new_bs = ["total_share","total_cur_assets","total_nca","total_assets","total_cur_liab","total_ncl","total_liab",
          "total_hldr_eqy_exc_min_int","total_hldr_eqy_inc_min_int"]
new_pr = ["total_revenue","total_cogs","operate_profit","total_profit","n_income","n_income_attr_p","minority_gain",
          "oth_compr_income","compr_inc_attr_p","compr_inc_attr_m_s"]
new_cs = ["c_inf_fr_operate_a","st_cash_out_act","stot_inflows_inv_act","stot_out_inv_act","stot_cash_in_fnc_act",
          "stot_cashout_fnc_act","n_incr_cash_cash_equ","c_cash_equ_beg_period","c_cash_equ_end_period",
          ]

x_fields = [*new_bs,*new_pr,*new_cs,"assets_turn"]

def next_date(end_date):
    year = int(end_date[0:4]) + 1
    new_date = "{}{}".format(year,end_date[4:])
    return new_date

def split_and_combine_finance(df):
    '''
    将所有行数据拆分成单季度数据，然后分别合并季度数据
    :param df:
    :return:
    '''
    pass

def handle_finance_df(df,no_fields,in_fields,total_col):
    '''

    :param df: 原始数据
    :param no_fields: 删除的列
    :param in_fields: 保留的数据列
    :return: 新的处理后的df
    '''
    # 填充空白数据
    df.fillna(0,inplace=True)
    # 去除重复的报告日数据
    df.drop_duplicates(subset=['end_date', ], keep='first', inplace=True, ignore_index=True)
    # 删除不需要的列
    df.drop(no_fields, axis=1, inplace=True)
    # 拷贝一份新的df
    df_new = df.copy(deep=True)
    # 将end_date设置为索引
    df_new.set_index("end_date",inplace=True)
    # 获取索引名称
    index_names = df_new.index.to_list()
    # 筛选出12月31日的日期
    select_index_names = []
    for name in index_names:
        str_name = str(name)
        if str_name.endswith("1231"):
            select_index_names.append(name)

    # 将所有12月31日的数据清零
    for name in select_index_names:
        for col in in_fields:
            df_new.loc[name, col] = 0
    # df_new将日期恢复
    df_new.reset_index(inplace=True)
    # 将原数据向下走一下
    df = df.shift(periods=1,)
    # 获取两表数据
    df_data = df[in_fields]
    df_new_data = df_new[in_fields]
    # 两表数据进行相减
    diff = df_data.subtract(df_new_data,axis=1,fill_value=0)
    # 回复相差数据对应的截止日
    diff["end_date"] = df["end_date"]
    # 删除diff的第一行
    diff = diff.drop(0)
    # 滚动求每年的和
    for col in in_fields:
        diff[col] = diff[col].rolling(4).sum()
    diff["end_date"] = diff["end_date"].shift(periods=3,)
    diff = diff.drop([1,2,3])
    diff_percent = diff.copy(deep=True)
    # 求结构百分比
    diff_percent["total"] = diff_percent[total_col]
    for col in in_fields:
        diff_percent[col] = diff_percent[col] / (diff_percent["total"]+0.01)

    return diff,diff_percent


def modify_trade_date(trade_date):

    trade_date_str = str(trade_date)
    three = "0331"
    six = "0630"
    nine = "0930"
    tw = "1231"
    year = trade_date_str[0:4]
    month = trade_date_str[4:6]
    if month == "03":
        new_date = "{}{}".format(year,three)
    elif month == "06":
        new_date = "{}{}".format(year, six)
    elif month == "09":
        new_date = "{}{}".format(year, nine)
    elif month == "12":
        new_date = "{}{}".format(year, tw)
    else:
        new_date = trade_date_str

    return new_date


def get_finance(env):
    conn = sqlite3.connect(r"D:\redhand\clean\data\tushare_db\tushare.db")
    df_stock_basic = pd.read_sql("select * from stock_basic;", conn)
    is_start = False
    for key,ts_code in enumerate(df_stock_basic["ts_code"]):
        print(ts_code)
        if ts_code == "301502.SZ":
            is_start = True

        if is_start:
            print("starting_{}".format(ts_code))
            if not env:
                if key > 4:
                    break
            # 现金流量表
            df_cash = pro.cashflow(ts_code=ts_code)
            try:
                df_new_cash,df_new_cash_per = handle_finance_df(df_cash,no_cash,cash,"c_cash_equ_end_period")
            except Exception as e:
                continue

            # 利润表
            df_income = pro.income(ts_code=ts_code)
            df_new_income,df_new_income_per = handle_finance_df(df_income, no_profit, profit,"total_revenue")

            # 资产负债表
            df_balancesheet = pro.balancesheet(ts_code=ts_code)
            # 填充空白数据
            df_balancesheet.fillna(0, inplace=True)
            df_balancesheet.drop_duplicates(subset=['end_date', ], keep='first', inplace=True, ignore_index=True)
            df_balancesheet_per = df_balancesheet.copy(deep=True)
            # 计算资产负债表结构百分比
            df_balancesheet_per["total_zc"] = df_balancesheet_per["total_assets"]
            for col in balance:
                df_balancesheet_per[col] = df_balancesheet_per[col] / df_balancesheet_per["total_zc"]

            #  报表合并
            # （1）合并百分比数据
            cash_income_per = df_new_cash_per.merge(df_new_income_per, on="end_date")
            df_combine_per = cash_income_per.merge(df_balancesheet_per,on="end_date")
            df_combine_per["assets_turn"] = df_combine_per["total_y"] / (df_combine_per["total_zc"] + 0.01)
            # df_combine_per.to_csv("combine_per.csv")
            # (2)合并财务报表数据
            cash_income = df_new_cash.merge(df_new_income, on="end_date")
            df_combine = cash_income.merge(df_balancesheet,on="end_date")
            df_combine.to_sql("finance_data", conn, if_exists="append", index=False)
            # df_combine.to_csv("combine.csv")
            #TODO: 根据合并财务报表数据计算财务指标
            # 合并相同或相似的报表项目，如应收账款级应收票据等。减少项目数量。

            df_combine_per["next_date"] = df_combine_per["end_date"].apply(next_date)
            # 获取月度数据
            df_month = ts.pro_bar(ts_code=ts_code,adj='qfq',freq="m")
            df_month = df_month[["trade_date", "close"]]
            df_month["trade_date"] = df_month["trade_date"].apply(modify_trade_date)
            df_month.dropna(inplace=True)
            # 财务数据与月度数据合并
            df_f_1 = df_combine_per.merge(df_month, left_on="end_date", right_on="trade_date")
            df_f_2 = df_f_1.merge(df_month, left_on="next_date", right_on="trade_date")
            df_f_2["close_pct"] = ((df_f_2["close_y"] - df_f_2["close_x"]) / df_f_2["close_x"]) * 100
            df_f_2["target"] = df_f_2["close_pct"].apply(categorize, args=(1, "y"))
            df_f_2.rename(columns={'trade_date_x': 'trade_date'}, inplace=True)
            # df_f_2.to_csv("res.csv")
            # # 报表保存
            df_f_2.to_sql("finance", conn, if_exists="append", index=False)
            time.sleep(1)
            print("end{}".format(ts_code))


class StockFinanceDataset(Dataset):

    def __init__(self, conn, start_date, end_date, one_hot=False, train=True, keras_data=False, step_length=3, fh=1,freq="y"):

        '''

        :param conn: 数据库连接
        :param one_hot: 是否将y转化为one_hot形式
        :param train: 是否取训练集
        :param keras_data: 是否转化为keras形式的数据，即n_samples,n_steps,n_vars
        :param step_length: 过去多少时间
        :param fh: 需要预测的未来多少时间
        :param freq: 频率，d:代表是日线数据，m:代表是分钟数据
        '''
        self.conn = conn
        self.start_date = start_date
        self.end_date = end_date
        self.one_hot = one_hot
        self.train = train
        self.keras_data = keras_data
        self.step_length = step_length
        self.fh = fh
        self.freq=freq
        # 获取指数数据
        self.df_index = pd.read_sql("select * from df_index;", conn)
        # 获取股票基本信息
        self.df_stock_basic = pd.read_sql("select * from stock_basic;", conn)

    def get_length(self):
        '''
        获取要训练和测试的数据集长度
        ts_code,train_data_length,test_data_length,total_train_data_length,total_test_data_length
        '''
        # 股票代码
        ts_codes = []
        # 训练数据总长度
        total_train_data_lengths = []
        # 测试数据总长度
        total_test_data_lengths = []
        # 训练数据长度
        train_data_lengths = []
        # 测试数据长度
        test_data_lengths = []
        # 总的训练数据长度
        total_train_data_length = 0
        # 总的测试数据长度
        total_test_data_length = 0

        df_ts_codes = pd.read_sql("select ts_code from finance", self.conn)

        for key, ts_code in enumerate(np.unique(df_ts_codes["ts_code"])):
            # 遍历所有的股票代码
            print(ts_code)
            # 获取股票日线或者分钟线数据
            df_data = get_one_stock_data_from_sqlite(self.conn, ts_code, self.start_date, self.end_date,"finance")
            # 如果在开始时间和截止时间内，该股票的数据小于历史回顾和未来预测期数据，那么该股票将没有可测试的数据，因此不测试该股票
            if len(df_data) < self.step_length + self.fh + 1:
                continue
            # 生成训练和测试数据集
            X_train, y_train, X_valid, y_valid = split_data(df_data, self.step_length,get_x=x_fields)
            # 如果生成的数据为None
            if y_train is None:
                continue
            ts_codes.append(ts_code)
            train_data_lengths.append(len(y_train))
            test_data_lengths.append(len(y_valid))
            total_train_data_length += len(y_train)
            total_train_data_lengths.append(total_train_data_length)
            total_test_data_length += len(y_valid)
            total_test_data_lengths.append(total_test_data_length)
            print(len(y_train), len(y_valid))
        df_record = pd.DataFrame({"ts_code": ts_codes,
                                  "train_data_length": train_data_lengths,
                                  "test_data_length": test_data_lengths,
                                  "total_train_data_length": total_train_data_lengths,
                                  "total_test_data_length": total_test_data_lengths,
                                  })
        print(df_record)
        df_record.to_sql("finance_step_length_{}_fh_{}".format(self.step_length, self.fh), self.conn, index=False,
                         if_exists="replace")

    def get_records(self):
        '''
        获取数据记录,如果没有的话，重新计算
        :return:
        '''
        try:
            df = pd.read_sql("select * from finance_step_length_{}_fh_{}".format(self.step_length, self.fh), self.conn)
            if len(df) < 2:
                self.get_length()
                df = pd.read_sql("select * from finance_step_length_{}_fh_{}".format(self.step_length, self.fh), self.conn)
        except pd.errors.DatabaseError:
            self.get_length()
            df = pd.read_sql("select * from finance_step_length_{}_fh_{}".format(self.step_length, self.fh), self.conn)
        return df

    def __len__(self):
        '''
        获取数据长度
        :return:
        '''

        self.df_records = self.get_records()
        if self.train:
            return self.df_records[TOTAL_TRAIN_DATA_LENGTH].max()
        else:
            return self.df_records[TOTAL_TEST_DATA_LENGTH].max()

    def get_data(self, idx, total_data_length_cl):
        '''
        根据idx从数据集中获取数据
        :param idx:
        :param total_data_length_cl:
        :param train:
        :return:
        '''
        # 获取idx在那只股票的什么位置
        self.df_records = self.get_records()
        search_len = idx + 1
        pos_start, pos_end = find_loc(search_len, self.df_records[total_data_length_cl])
        if pos_start == 0:
            data_pos = idx
        else:
            data_pos = idx - self.df_records[total_data_length_cl].iloc[pos_start - 1]
        # 获取股票信息
        ts_code = self.df_records["ts_code"].iloc[pos_start]
        df_data = get_one_stock_data_from_sqlite(self.conn, ts_code, self.start_date, self.end_date,"finance")
        X_train, y_train, X_valid, y_valid = split_data(df_data, self.step_length,get_x=x_fields)
        # 获取股票对应位置的数据
        if self.train:
            x = X_train[data_pos]
            y = y_train[data_pos]
        else:
            x = X_valid[data_pos]
            y = y_valid[data_pos]
        # 将获取的位置数据进行转换
        if self.keras_data:
            x = x.transpose((1, 0))
        if self.one_hot:
            y = y.reshape(-1, 1)
            onehot_encoder = OneHotEncoder(categories=[[0, 1, 2, 3, 4, 5, 6, 7, 8]], sparse_output=False)
            y = onehot_encoder.fit_transform(y)
            y = np.squeeze(y)
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.long)

    def __getitem__(self, idx):
        if self.train:
            x, y = self.get_data(idx, TOTAL_TRAIN_DATA_LENGTH)
            return x, y
        else:
            x, y = self.get_data(idx, TOTAL_TEST_DATA_LENGTH)
            return x, y


if __name__ == '__main__':
    # get_finance(env=True)
    settings_year_3_1 = {
        "start_date": "20000101",
        "end_date": "20231231",
        "db": r"D:\redhand\clean\data\tushare_db\stock_year.db",
        "table": "finance",
        "step_length": 3,  # 16*20
        "fh": 1,  # 16*5
        "freq": "y",  # d代表day,m代表minute
        "n_vars": 29,
    }
    new_fit(settings_year_3_1,load_model=False,batch_size=32,dataset=StockFinanceDataset)

    # from torch.utils.data import DataLoader

    # conn = sqlite3.connect(r"D:\redhand\clean\data\tushare_db\stock_year.db")
    # train_data = StockFinanceDataset(conn, "20000101", "20231231", True, True, True, 3, 1)
    # train_dataloader = DataLoader(train_data, batch_size=32, shuffle=False)
    # for key, (x_train, y_train) in enumerate(train_dataloader):
    #     print(x_train.shape)
    #     print(x_train)
    #     print(key)

    # df_f = pd.read_sql("select * from finance_step_length_3_fh_1",conn)
    # print(df_f)
    # print(len(df_f))
    # print(np.unique(df_f["ts_code"]))
    # df_f.to_csv("res.csv")


    # cursor = conn.cursor()
    # try:
    #     cursor.execute("DROP TABLE {}".format("finance_step_length_3_fh_1"))
    # except Exception as e:
    #     print(e)
