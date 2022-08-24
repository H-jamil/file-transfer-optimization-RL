import os
import time
import gym
import argparse
import numpy as np
import socket
import datetime
import numpy as np
import logging as log
import multiprocessing as mp
import pandas as pd
import re
from config import configurations
from transferEnv import *
from transferEnv_ import *
from optimizer_gd import *
from stable_baselines3 import PPO
from stable_baselines3.ppo import MlpPolicy
from stable_baselines3.common.env_util import make_vec_env

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

if __name__ == "__main__":

  models_dir = f"models/transferEnv-{time.time()}"
  logdir = f"logs/transferEnv-{time.time()}"

  if not os.path.exists(models_dir):
      os.makedirs(models_dir)

  if not os.path.exists(logdir):
      os.makedirs(logdir)

  transfer=TransferClass_(configurations,log,transfer_emulation=True)
  env=transferEnv(transfer,csv_save=True)
  env.change_run_type(1)
  models_dir = "models/transferPPO_queue_mod-1661258181.581414"
  model_path = f"{models_dir}/10000"
  best_model = PPO.load(model_path, env=env)
  total_reward=0
  obs = env.reset()
  while True:
      action, _states = best_model.predict(obs)
      obs, rewards, dones, info = env.step(action)
      total_reward+=rewards
      if dones:
        break
  env.reset()
  print(f"total reward is {total_reward}")
