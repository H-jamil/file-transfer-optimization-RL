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
from optimizer_gd_ import *

import argparse
import gym
import pybullet_envs

from lib import model
import numpy as np
import torch
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
  end_time=time.time()
  total_bytes = np.sum(transfer.file_sizes)
  print(f"total_bytes:{total_bytes} start_time:{start_time}, end_time:{end_time} ")
  transfer_throughput=int((total_bytes*8)/(np.round(end_time-start_time,1)*1000*1000))
  print(f"transfer_throughput {transfer_throughput} Mbps#############")
  print(" ###########  final CCs ",final_ccs)
  transferEnvironment.close()

  transferEnvironment.reset()
  start_time=time.time()
  final_ccs=gradient_opt_(transferEnvironment)
  end_time=time.time()
  total_bytes = np.sum(transfer.file_sizes)
  print(f"total_bytes:{total_bytes} start_time:{start_time}, end_time:{end_time} ")
  transfer_throughput=int((total_bytes*8)/(np.round(end_time-start_time,1)*1000*1000))
  print(f"transfer_throughput {transfer_throughput} Mbps#############")
  print(" ###########  final CCs ",final_ccs)
  transferEnvironment.close()

  # # time.sleep(1)
  transferEnvironment.reset()
  start_time=time.time()
  final_ccs=gradient_opt_fast(transferEnvironment)
  end_time=time.time()
  total_bytes = np.sum(transfer.file_sizes)
  print(f"total_bytes:{total_bytes} start_time:{start_time}, end_time:{end_time} ")
  transfer_throughput=int((total_bytes*8)/(np.round(end_time-start_time,1)*1000*1000))
  print(f"transfer_throughput {transfer_throughput} Mbps#############")
  print(" ###########  final CCs ",final_ccs)
  transferEnvironment.close()

  transferEnvironment.reset()
  start_time=time.time()
  final_ccs=gradient_opt_fast_(transferEnvironment)
  end_time=time.time()
  total_bytes = np.sum(transfer.file_sizes)
  print(f"total_bytes:{total_bytes} start_time:{start_time}, end_time:{end_time} ")
  transfer_throughput=int((total_bytes*8)/(np.round(end_time-start_time,1)*1000*1000))
  print(f"transfer_throughput {transfer_throughput} Mbps#############")
  print(" ###########  final CCs ",final_ccs)
  transferEnvironment.close()

  # # time.sleep(1)
  # transferEnvironment.reset()
  # start_time=time.time()
  # final_ccs=bayes_optimizer(transferEnvironment,configurations)
  # end_time=time.time()
  # total_bytes = np.sum(transfer.file_sizes)
  # print(f"total_bytes:{total_bytes} start_time:{start_time}, end_time:{end_time} ")
  # transfer_throughput=int((total_bytes*8)/(np.round(end_time-start_time,1)*1000*1000))
  # print(f"transfer_throughput {transfer_throughput} Mbps#############")
  # print(" ###########  final CCs ",final_ccs)
  # transferEnvironment.close()

  # # time.sleep(1)
  # transferEnvironment.close()
  # transferEnvironment.reset()
  # transferEnvironment.change_run_type(1)
  # start_time=time.time()
  # net = model.DDPGActor(transferEnvironment.observation_space.shape[0], transferEnvironment.action_space.n)
  # net.load_state_dict(torch.load("/home/hjamil/Documents/file-transfer-optimization-RL/saves/ddpg-ddpg-rev2/best_+29.000_305.dat"))
  # obs = transferEnvironment.reset()
  # total_reward = 0.0
  # total_steps = 0
  # while True:
  #   obs_v = torch.FloatTensor([obs])
  #   mu_v = net(obs_v)
  #   action = mu_v.squeeze(dim=0).data.numpy()
  #   action = np.clip(action, -1, 1)
  #   obs, reward, done, _ = transferEnvironment.step(action)
  #   total_reward += reward
  #   total_steps += 1
  #   if done:
  #     transferEnvironment.close()
  #     break
  #   print(obs, reward, done,transferEnvironment.transferClassObject.file_incomplete.value)

  # print("In %d steps we got %.3f reward" % (total_steps, total_reward))
  # end_time=time.time()
  # total_bytes = np.sum(transfer.file_sizes)
  # print(f"total_bytes:{total_bytes} start_time:{start_time}, end_time:{end_time} ")
  # transfer_throughput=int((total_bytes*8)/(np.round(end_time-start_time,1)*1000*1000))
  # print(f"transfer_throughput {transfer_throughput} Mbps#############")

  transferEnvironment.reset()
  transferEnvironment.close()

