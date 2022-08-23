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

  models_dir = f"models/transferPPO_no_error-{time.time()}"
  logdir = f"logs/transferPPO_no_error-{time.time()}"

  if not os.path.exists(models_dir):
      os.makedirs(models_dir)

  if not os.path.exists(logdir):
      os.makedirs(logdir)

  transfer=TransferClass_(configurations,log,transfer_emulation=True)
  env=transferEnv(transfer,csv_save=True)
  env.change_run_type(1)
  # The learning agent and hyperparameters
  env = Monitor(env, models_dir)
  model = PPO(
    policy=MlpPolicy,
    env=env,
    seed=0,
    batch_size=256,
    ent_coef=0.00429,
    learning_rate=7.77e-05,
    n_epochs=10,
    n_steps=8,
    gae_lambda=0.9,
    gamma=0.9999,
    clip_range=0.1,
    max_grad_norm =5,
    vf_coef=0.19,
    use_sde=True,
    policy_kwargs=dict(log_std_init=-3.29, ortho_init=False),
    verbose=1,
    tensorboard_log=logdir
    )
  # models_dir_ = "models/transferEnv2-1660806824.7421222"
  # model_path = f"{models_dir_}/1000"
  # model = PPO.load(model_path, env=env,
  #   tensorboard_log=logdir)
  callback = SaveOnBestTrainingRewardCallback(check_freq=100, log_dir=models_dir)
  TIMESTEPS = 2000
  for i in range(10):
    model.learn(total_timesteps=TIMESTEPS,reset_num_timesteps=False, tb_log_name="PPO_FixedOffset_no_error",callback=callback)
    model.save(f"{models_dir}/{TIMESTEPS*i}")



