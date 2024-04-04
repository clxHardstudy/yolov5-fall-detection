#!/usr/bin/python3
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.image import MIMEImage
from email.header import Header
import requests


def send_email(image_url):
    sender = 'clx20020905@foxmail.com'
    receivers = ['2050669795@qq.com']  # 接收邮箱
    auth_code = "vujkrnexykzvfcfd"  # 授权码

    # 创建一个多部分消息对象
    message = MIMEMultipart()
    message['From'] = Header("Sender<%s>" % sender)  # 发送者
    message['To'] = Header("Receiver<%s>" % receivers[0])  # 接收者
    subject = 'Fall-Detection'
    message['Subject'] = Header(subject, 'utf-8')

    # 添加文本消息
    text = MIMEText('检测到人员跌倒', 'plain', 'utf-8')
    message.attach(text)

    # 从URL加载图片
    response = requests.get(image_url)
    if response.status_code == 200:
        # 添加图片消息
        image_data = response.content
        image = MIMEImage(image_data)
        image.add_header('Content-Disposition', 'attachment', filename="image.jpg")
        message.attach(image)
    else:
        print("Failed to load image from URL")

    try:
        server = smtplib.SMTP_SSL('smtp.qq.com', 465)
        server.login(sender, auth_code)
        server.sendmail(sender, receivers, message.as_string())
        print("邮件发送成功")
        server.close()
    except smtplib.SMTPException:
        print("Error: 无法发送邮件")
