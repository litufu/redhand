#! /usr/bin/env python3
import os
import time
import schedule
import yagmail


def send_mail():
    # 携带的附件名称
    filename = r'D:\redhand\clean\data\state_dict\inceptiontime_new.keras'
    file_info = os.stat(filename)
    time_obj = time.gmtime(file_info.st_mtime)
    formatted_time = time.strftime("%Y-%m-%d %H:%M:%S", time_obj)

    # 创建客户端
    yag = yagmail.SMTP(
        user='litufu2023@126.com',
        password='YFWCZEKRUPDPCCVO',
        host='smtp.126.com',  # 邮局的 smtp 地址
        port='465',  # 邮局的 smtp 端口
        smtp_ssl=True)
    # 发送邮件
    yag.send(to='18258274554@163.com',
             subject='{}'.format("文件最新修改时间".format(formatted_time)),
             contents='这是一封正常的邮件，不要过滤',
             attachments=filename)
    # 关闭 yagmail 客户端
    yag.close()


if __name__ == '__main__':
    schedule.every(60).minutes.do(send_mail)
    while True:
        schedule.run_pending()
        time.sleep(1)