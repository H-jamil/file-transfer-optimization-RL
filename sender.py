import os
import time
import socket
import datetime
import numpy as np
import logging as log
import multiprocessing as mp
import pandas as pd
import re
from config import configurations
from transferClass import *


log_FORMAT = '%(created)f -- %(levelname)s: %(message)s'
log_file = "logs/" + datetime.datetime.now().strftime("%m_%d_%Y_%H_%M_%S") + ".log"

if configurations["loglevel"] == "debug":
  log.basicConfig(
      format=log_FORMAT,
      datefmt='%m/%d/%Y %I:%M:%S %p',
      level=log.DEBUG,
      handlers=[
          log.FileHandler(log_file),
          log.StreamHandler()
      ]
    )
  mp.log_to_stderr(log.info)

else:
  log.basicConfig(
      format=log_FORMAT,
      datefmt='%m/%d/%Y %I:%M:%S %p',
      level=log.INFO,
      handlers=[
          log.FileHandler(log_file),
          log.StreamHandler()
      ]
  )

configurations["thread_limit"] = configurations["max_cc"]
configurations["cpu_count"] = mp.cpu_count()

if __name__=="__main__":
  transfer=TransferClass(configurations,log)
  workers,reporting_process=transfer.run()
  while(transfer.file_incomplete.value != 0):
    if np.sum(transfer.process_status) == 0:
      print("Changing concurrency to 8 ******")
      transfer.change_concurrency([8])
      time.sleep(5)
      print("Changing concurrency to 6 ******")
      transfer.change_concurrency([6])
      time.sleep(5)
      print("Changing concurrency to 4 ******")
      transfer.change_concurrency([4])
      time.sleep(5)
      print("Changing concurrency to 8 ******")
      transfer.change_concurrency([8])
      time.sleep(5)

  for p in workers:
    if p.is_alive():
      p.terminate()
      p.join(timeout=0.1)

  if reporting_process.is_alive():
    reporting_process.terminate()
    reporting_process.join(timeout=0.1)


  list_main=[]
  for i in range(len(transfer.throughput_logs)):
    list_main.append(transfer.throughput_logs[i])

  df = pd.DataFrame(list_main, columns = ['curr_thrpt','goodput','cc_level','cwnd','rtt','packet_loss_rate','score','date_time'])
  mod_df=df.dropna(axis=0, how='any')
  mod_df.to_csv('record.csv', sep='\t', encoding='utf-8')
