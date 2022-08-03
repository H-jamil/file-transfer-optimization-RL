import gym
from gym import spaces
import numpy as np
from transferClass import *
import random
import copy
class transferEnv(gym.Env):
  metadata={'render.modes':  []}

  def __init__(self,transferClassObject,record_name="record_"+datetime.datetime.now().strftime("%m_%d_%Y_%H_%M_%S")+".csv"):
    self.transferClassObject=transferClassObject
    self.action_space = spaces.Discrete(int(transferClassObject.configurations["thread_limit"]))
    self.observation_space = spaces.Box(low=0, high=np.inf, shape=(3*6,), dtype=np.float32)
    self.record_file_name=record_name
    self.current_observation = np.zeros([3,6],dtype = np.float32).flatten()
    dummy_list=[]
    df = pd.DataFrame(dummy_list, columns = ['curr_thrpt','cc_level','cwnd','rtt','packet_loss_rate','score','date_time'])
    df.to_csv(self.record_file_name, sep='\t', encoding='utf-8',index=False)

  def reset(self):
    list_main=[]
    if len(self.transferClassObject.throughput_logs) > 0:
      for i in range(len(self.transferClassObject.throughput_logs)):
        list_main.append(self.transferClassObject.throughput_logs[i])
    df = pd.DataFrame(list_main, columns = ['curr_thrpt','cc_level','cwnd','rtt','packet_loss_rate','score','date_time'])
    # mod_df=df.dropna(axis=0, how='any')
    mod_df=df.fillna(0)
    mod_df.to_csv(self.record_file_name, mode='a', index=False, header=False, sep='\t', encoding='utf-8')
    self.current_observation=self.transferClassObject.reset().flatten()
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
        log_list_array=np.array(log_list).flatten()
        try:
          score_=np.mean(score)
        except:
          score_=0
      return log_list_array,score_,done,info

    else:
      done=True
      score_=10 ** 10
      return np.zeros([3,6],dtype = np.float32).flatten(),score_,done,info

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
