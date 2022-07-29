import gym
from gym import spaces
import numpy as np
from transferClass import *
import random
import copy
class transferEnv(gym.Env):
  metadata={'render.modes':  []}

  def __init__(self,transferClassObject):
    self.transferClassObject=transferClassObject
    self.action_space = spaces.Discrete(int(transferClassObject.configurations["thread_limit"]))
    self.observation_space = spaces.Box(low=0, high=np.inf, shape=(3,7), dtype=np.float32)
    self.current_observation = np.zeros([3,7],dtype = np.float32)

  def reset(self):
    self.current_observation=self.transferClassObject.reset()
    self.workers,self.reporting_process=self.transferClassObject.run()
    return self.current_observation

  def step(self,action):
    info={}
    if self.transferClassObject.file_incomplete.value != 0:
      done = False
      self.transferClassObject.log.info(f"Changing concurrency to {action} ******")
      self.transferClassObject.change_concurrency([action])
      timer3s=time.time()
      while timer3s + 3.5 > time.time():
        pass
      if len(self.transferClassObject.throughput_logs)>=3:
        log_list=copy.deepcopy(self.transferClassObject.throughput_logs[-3:])
        score=[]
        for i in log_list:
          del i[-1]
          score.append(i[-1])
        log_list_array=np.array(log_list)
        try:
          score_=np.mean(score)
        except:
          score_=0
      return log_list_array,score_,done,info

    else:
      done=True
      score_=10 ** 10
      return np.zeros([3,7],dtype = np.float32),score_,done,info

  def bayes_step(self,action):
    params = [1 if x<1 else int(np.round(x)) for x in action]
    _,score_b,done_b,__=self.step(params[0])
    return score_b

  def close(self):
    for p in self.workers:
      if p.is_alive():
        p.terminate()
        p.join(timeout=0.1)

    if self.reporting_process.is_alive():
      self.reporting_process.terminate()
      self.reporting_process.join(timeout=0.1)
