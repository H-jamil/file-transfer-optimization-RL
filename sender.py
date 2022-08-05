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
from transferEnv import *
from optimizer_gd import *

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

# if __name__=="__main__":
#   transfer=TransferClass(configurations,log)
#   workers,reporting_process=transfer.run()
#   start_time=time.time()
#   while(transfer.file_incomplete.value != 0):
#     if np.sum(transfer.process_status) == 0:
#       print("Starting transfer *********")
#       print("Changing concurrency to 8 ******")
#       transfer.change_concurrency([8])
#       time.sleep(5)
#       print("Changing concurrency to 6 ******")
#       transfer.change_concurrency([6])
#       time.sleep(5)
#       print("Changing concurrency to 4 ******")
#       transfer.change_concurrency([4])
#       time.sleep(5)
#       print("Changing concurrency to 8 ******")
#       transfer.change_concurrency([8])
#       time.sleep(5)
#   end_time=time.time()
#   total_bytes = np.sum(transfer.file_sizes)
#   print(f"total_bytes:{total_bytes} start_time:{start_time}, end_time:{end_time} ")
#   transfer_throughput=int((total_bytes*8)/(np.round(end_time-start_time,1)*1000*1000))

#   print(f"transfer_throughput {transfer_throughput} Mbps#############")

#   for p in workers:
#     if p.is_alive():
#       p.terminate()
#       p.join(timeout=0.1)

#   if reporting_process.is_alive():
#     reporting_process.terminate()
#     reporting_process.join(timeout=0.1)

#   list_main=[]
#   for i in range(len(transfer.throughput_logs)):
#     list_main.append(transfer.throughput_logs[i])

#   df = pd.DataFrame(list_main, columns = ['curr_thrpt','goodput','cc_level','cwnd','rtt','packet_loss_rate','score','date_time'])
#   mod_df=df.dropna(axis=0, how='any')
#   mod_df.to_csv('record.csv', sep='\t', encoding='utf-8')

if __name__=="__main__":
  transfer=TransferClass(configurations,log,transfer_emulation=True)
  transferEnvironment=transferEnv(transfer)

  transferEnvironment.reset()
  start_time=time.time()
  final_ccs=gradient_opt(transferEnvironment)
  # final_ccs=gradient_opt_fast(transferEnvironment)
  # final_ccs=bayes_optimizer(transferEnvironment,configurations)
  end_time=time.time()
  total_bytes = np.sum(transfer.file_sizes)
  # print(f"final CC is {final_ccs[-1]}")
  print(f"total_bytes:{total_bytes} start_time:{start_time}, end_time:{end_time} ")
  transfer_throughput=int((total_bytes*8)/(np.round(end_time-start_time,1)*1000*1000))
  print(f"transfer_throughput {transfer_throughput} Mbps#############")
  print(" ###########  final CCs ",final_ccs)

  transferEnvironment.reset()
  start_time=time.time()
  # final_ccs=gradient_opt(transferEnvironment)
  final_ccs=gradient_opt_fast(transferEnvironment)
  # final_ccs=bayes_optimizer(transferEnvironment,configurations)
  end_time=time.time()
  total_bytes = np.sum(transfer.file_sizes)
  # print(f"final CC is {final_ccs[-1]}")
  print(f"total_bytes:{total_bytes} start_time:{start_time}, end_time:{end_time} ")
  transfer_throughput=int((total_bytes*8)/(np.round(end_time-start_time,1)*1000*1000))
  print(f"transfer_throughput {transfer_throughput} Mbps#############")
  print(" ###########  final CCs ",final_ccs)

  transferEnvironment.reset()
  start_time=time.time()
  # final_ccs=gradient_opt(transferEnvironment)
  # final_ccs=gradient_opt_fast(transferEnvironment)
  final_ccs=bayes_optimizer(transferEnvironment,configurations)
  end_time=time.time()
  total_bytes = np.sum(transfer.file_sizes)
  # print(f"final CC is {final_ccs[-1]}")
  print(f"total_bytes:{total_bytes} start_time:{start_time}, end_time:{end_time} ")
  transfer_throughput=int((total_bytes*8)/(np.round(end_time-start_time,1)*1000*1000))
  print(f"transfer_throughput {transfer_throughput} Mbps#############")
  print(" ###########  final CCs ",final_ccs)
  transferEnvironment.reset()
  transferEnvironment.close()

  # done=False
  # while(not done):
  #   state,score,done,_=transferEnvironment.step(2)
  #   print("state:*********",state)
  #   print(f"score{score}  done {done} ********")
  #   time.sleep(10)
  #   state,score,done,_=transferEnvironment.step(4)
  #   print("state:*********",state)
  #   print(f"score{score}  done {done} ********")
  #   time.sleep(10)
  #   state,score,done,_=transferEnvironment.step(6)
  #   print("state:*********",state)
  #   print(f"score{score}  done {done} ********")
  #   time.sleep(10)
  #   state,score,done,_=transferEnvironment.step(8)
  #   print("state:*********",state)
  #   print(f"score{score}  done {done} ********")
  #   time.sleep(30)
  #   state,score,done,_=transferEnvironment.step(16)
  #   print("state:*********",state)
  #   print(f"score{score}  done {done} ********")
  #   time.sleep(30)
  #   state,score,done,_=transferEnvironment.step(32)
  #   print("state:*********",state)
  #   print(f"score{score}  done {done} ********")
  #   time.sleep(30)
  # list_main=[]
  # for i in range(len(transferEnvironment.transferClassObject.throughput_logs)):
  #   list_main.append(transferEnvironment.transferClassObject.throughput_logs[i])

  # df = pd.DataFrame(list_main, columns = ['curr_thrpt','cc_level','cwnd','rtt','packet_loss_rate','score','date_time'])
  # # mod_df=df.dropna(axis=0, how='any')
  # mod_df=df.fillna(0)
  # record_name="record_"+datetime.datetime.now().strftime("%m_%d_%Y_%H_%M_%S")+".csv"
  # mod_df.to_csv(record_name, sep='\t', encoding='utf-8')
