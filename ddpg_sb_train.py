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
from stable_baselines3 import DDPG
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
from stable_baselines3.common.env_util import make_vec_env
from SaveOnBestTrainingRewardCallback import *

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

  models_dir = f"models/TransferDDPG-0-{time.time()}"
  logdir = f"logs/TransferDDPG-0-{time.time()}"

  if not os.path.exists(models_dir):
      os.makedirs(models_dir)

  if not os.path.exists(logdir):
      os.makedirs(logdir)

  transfer=TransferClass_(configurations,log,transfer_emulation=True)
  env=transferEnv(transfer,csv_save=True)
  env.change_run_type(1)
  n_actions = env.action_space.shape[-1]
  action_noise = OrnsteinUhlenbeckActionNoise(mean=np.zeros(n_actions), sigma=0.3 * np.ones(n_actions))
  # The learning agent and hyperparameters
  env = Monitor(env, models_dir)
  model = DDPG(policy="MlpPolicy",
    env=env,
    learning_rate=0.0001,
    buffer_size=1000000,
    learning_starts=300,
    batch_size=256,
    gamma=0.99,
    action_noise=action_noise,
    gradient_steps=-1,
    train_freq=(1,"episode"),
    seed=0,
    verbose=1,
    tensorboard_log=logdir
    )
  
  callback = SaveOnBestTrainingRewardCallback(check_freq=100, log_dir=models_dir)
  TIMESTEPS = 1000
  for i in range(14):
    model.learn(total_timesteps=TIMESTEPS,reset_num_timesteps=False, tb_log_name="DDPG-0",callback=callback)
    model.save(f"{models_dir}/{TIMESTEPS*i}")



