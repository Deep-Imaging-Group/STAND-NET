import os

import smtplib
from email.mime.text import MIMEText
from email.header import Header

from smtplib import SMTP_SSL
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.header import Header
import argparse
from loguru import logger

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", type=str)
    parser.add_argument("--info_file", type=str)
    args = parser.parse_args()

    logger.info(args.subject)
    logger.info(args.info_file)

    path = args.info_file
    mail_content = []
    logger.info(os.path.exists(path))
    if os.path.exists(path):
        logger.info("-----------------------------")
        with open(path, "r") as f:
            mail_content = f.readlines()
            logger.info(mail_content)
        
        os.remove(path)

    if not mail_content:
        return
    mail_content.append("窗前明月光, 疑似地上霜")

    mail_content = "".join(mail_content)



    host_server = os.environ.get('SMTP_HOST', 'smtp.163.com')  #qq邮箱smtp服务器
    sender_qq = os.environ.get('SMTP_USER', '') #发件人邮箱
    pwd = os.environ.get('SMTP_PASSWORD', '')
    receiver = [x.strip() for x in os.environ.get('SMTP_TO', '').split(',') if x.strip()]#收件人邮箱
    mail_title = args.subject #邮件标题
    # mail_content = "窗前明月光, 疑似地上霜" #邮件正文内容
    # 初始化一个邮件主体
    msg = MIMEMultipart()
    msg["Subject"] = Header(mail_title,'utf-8')
    msg["From"] = sender_qq
    # msg["To"] = Header("测试邮箱",'utf-8')
    msg['To'] = ";".join(receiver)
    # 邮件正文内容
    msg.attach(MIMEText(mail_content,'plain','utf-8'))


    if not (sender_qq and pwd and receiver):
        logger.info('Email skipped: configure SMTP_USER, SMTP_PASSWORD and SMTP_TO')
        return

    smtp = SMTP_SSL(host_server, int(os.environ.get('SMTP_PORT', '465'))) # ssl登录

    smtp.login(sender_qq,pwd)

    smtp.sendmail(sender_qq,receiver,msg.as_string())

    # quit():用于结束SMTP会话。
    smtp.quit()

    logger.info("Experiment end email sent successfully...")


if __name__ == "__main__":
    main()
