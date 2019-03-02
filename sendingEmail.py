# -*- coding: utf-8 -*-
"""
Created on Thu Jan 31 20:22:49 2019

@author: Hai
"""

# python script for sending message update 

# Python code to illustrate Sending mail from 
# your Gmail account 
import smtplib 

# creates SMTP session 
s = smtplib.SMTP('smtp.gmail.com', 587) 

# start TLS for security 
s.starttls() 

# Authentication 
s.login("zhanghai1024@gmail.com", "huwan881024") 

# message to be sent 
message =  """\
Subject: Hi there

This message is sent from Python."""


# sending the mail 
s.sendmail("zhanghai1024@gmail.com", "chenxia801@gmail.com", message) 

# terminating the session 
s.quit() 
