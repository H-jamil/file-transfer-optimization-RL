import gym
from gym import spaces
import numpy as np
from transferClass import *
from transferClass_ import *
from transferClass_dummy import *
import random
import copy
# from queue import Queue

MAX_RTT=50
MIN_RTT=0
MAX_THROUGHPUT=1000
MIN_THROUGHPUT=0
MAX_CC_LEVEL=32
MIN_CC_LEVEL=0
MAX_CWND=1000
MIN_CWND=0
MAX_PLR=0.4
MIN_PLR=0
MAX_SCORE=1000
MIN_SCORE=0

def get_int_cc(actionFloatValue):
  return np.abs(np.linspace(-1,1,num=5)-actionFloatValue).argmin()



class transferEnv_RA(gym.Env):
  metadata={'render.modes':  []}

  def __init__(self,transferClassObject,record_name="record_"+datetime.datetime.now().strftime("%m_%d_%Y_%H_%M_%S")+".csv",runType=0,historyLen=3,csv_save=False,start_cc=10):####runType=0 for non-RL run runType=!0 for RL run
    self.transferClassObject=transferClassObject
    # self.action_space = spaces.Discrete(int(transferClassObject.configurations["thread_limit"]))
    self.historyLen=historyLen
    self.action_space=spaces.Box(low= -1,high= 1,shape=(1,), dtype=np.float32)
    # self.action_space=spaces.Box(low= -1,high= 1,shape=(transferClassObject.configurations["thread_limit"],), dtype=np.float32)
    self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.historyLen*5,), dtype=np.float32)
    self.record_file_name=record_name
    self.csv_save=csv_save
    self.current_observation = np.zeros([self.historyLen,5],dtype = np.float32).flatten()
    self.runType=runType
    self.episode_time=None
    self.start_cc=start_cc
    self.current_cc=self.start_cc
    self.action_factors=[5,2,0,-1,-2]
    dummy_list=[]
    self.old_score=0.0
    # self.state_q=[]
    self.minRTT=1000
    df = pd.DataFrame(dummy_list, columns = ['curr_thrpt','cc_level','cwnd','rtt','packet_loss_rate','score','date_time'])
    df.to_csv(self.record_file_name, sep='\t', encoding='utf-8',index=False)
    # df = pd.DataFrame(dummy_list, columns = ['latency_gradient','latency_ratio','cwnd','packet_loss_rate','throughput'])
    # df.to_csv(self.record_file_name+'states', sep='\t', encoding='utf-8',index=False)

  def change_run_type(self,run_value):
    self.runType=run_value

  def reset(self):
    list_main=[]
    if len(self.transferClassObject.throughput_logs) > 0:
      for i in range(len(self.transferClassObject.throughput_logs)):
        list_main.append(self.transferClassObject.throughput_logs[i])
    if self.csv_save:
      df = pd.DataFrame(list_main, columns = ['curr_thrpt','cc_level','cwnd','rtt','packet_loss_rate','score','date_time'])
      mod_df=df.fillna(0)
      mod_df.to_csv(self.record_file_name, mode='a', index=False, header=False, sep='\t', encoding='utf-8')
      # df = pd.DataFrame(self.state_q, columns = ['latency_gradient','latency_ratio','cwnd','packet_loss_rate','throughput'])
      # mod_df=df.fillna(0)
      # mod_df.to_csv(self.record_file_name+'states', mode='a', index=False, header=False, sep='\t', encoding='utf-8')
    self.transferClassObject.reset()
    self.workers,self.reporting_process=self.transferClassObject.run()
    # self.state_q=[]
    self.minRTT=1000
    self.current_cc=self.start_cc
    self.old_score=0.0
    # self.close()
    if self.runType==0:
      self.runType=0
    else:
      self.runType=1
    self.episode_time=time.time()
    return np.zeros([self.historyLen,5],dtype = np.float32).flatten()

  def step(self,action):
    info={}

    if (self.episode_time + 30 <= time.time()):
      self.transferClassObject.file_incomplete.value=0
      self.transferClassObject.log.info("episode expires")
      done=True
      if self.runType==0:
        score_=10 ** 10
      else:
        score_=0
      # self.reset()
      self.close()
      return np.zeros([self.historyLen,5],dtype = np.float32).flatten(),float(score_),done,info

    # if self.transferClassObject.file_incomplete.value == 0:
    #   done=True
    #   if self.runType==0:
    #     score_=10 ** 10
    #   else:
    #     score_=0
    #   self.close()
    #   return np.zeros([self.historyLen,5],dtype = np.float32).flatten(),float(score_),done,info

    if self.transferClassObject.file_incomplete.value != 0:
      done = False
      if self.runType==0:
        self.transferClassObject.log.info(f"Changing concurrency to {action} ******")
        self.transferClassObject.change_concurrency([action])
      else:
        # action_t=np.argmax(action)
        action_t=get_int_cc(action[0])
        #####
        # if action_t ==0 : increment +5
        # if action_t ==1 : increment +2
        # if action_t ==2 : increment 0 / decrement 0 : do nothing
        # if action_t ==3 : decrement -1
        # if action_t ==3 : decrement -2
        #####
        self.current_cc+=self.action_factors[action_t]

        if self.current_cc >32:
          self.current_cc =32
        elif self.current_cc < 1:
          self.current_cc =1
        else:
          self.current_cc=self.current_cc

        self.transferClassObject.log.info(f"action is  {action_t} for action array {action[0]} and Changing concurrency to {self.current_cc}******")
        self.transferClassObject.change_concurrency([self.current_cc])

      timer3s=time.time()
      while timer3s + 3.2> time.time():
        pass
      if len(self.transferClassObject.throughput_logs)>=3:
        log_list=copy.deepcopy(self.transferClassObject.throughput_logs[-3:])
        score=[]
        for i in log_list:
          del i[-1]
          score.append(i[-1])
        if self.runType==0:
          log_list_array=np.array(log_list).flatten()
        ##########################
        else:
          log_list_=[[],[],[],[]]
          for log in log_list:
            log_list_[0].append(log[0])
            log_list_[1].append(log[2])
            log_list_[2].append(log[3])
            log_list_[3].append(log[4])
          log_list_state=[]
          latency_gradient=np.gradient(np.array(log_list_[2],dtype=float))
          log_list_state.append(latency_gradient)

          rtt_ratio=[]
          for rtt in log_list_[2]:
            if rtt <=self.minRTT:
              self.minRTT=rtt
            rtt_r = rtt/self.minRTT if self.minRTT>0 else 0.0
            rtt_ratio.append(rtt_r)
          log_list_state.append(rtt_ratio)
          cwnd_values=log_list_[1]
          log_list_state.append(cwnd_values)

          plr_values=log_list_[3]
          log_list_state.append(plr_values)
          throughput_values=log_list_[0]
          log_list_state.append(throughput_values)
          # self.state_q.append(log_list_state)
          log_list_array=np.array(log_list_state,dtype = np.float32).flatten()
          ################################
        try:
          score_=np.mean(score)
          if self.runType==1:
            reward=np.round((score_- self.old_score)*10,2)
            # if score_- self.old_score > 0.0:
            #   reward=np.round((score_- self.old_score)*10,2)
            #   # score_=(-1.0)*score_
            # elif score_- self.old_score < 0.0:
            #   reward=(score_- self.old_score)
            # else:
            #   reward=0.0
            self.old_score=score_
        except:
          score_=0.0
          reward=0.0
          self.old_score=score_
      else:
        log_list_array=np.zeros([self.historyLen,5],dtype = np.float32).flatten()
        score_=0.0
        reward=0.0
        self.old_score=score_
      self.transferClassObject.log.info(f"score {score} old_score {self.old_score} reward {reward}******")
      return log_list_array,float(reward),done,info

    # else:
    #   done=True
    #   if self.runType==0:
    #     score_=10 ** 10
    #   else:
    #     score_=0.0
    #   self.close()
    #   return np.zeros([self.historyLen,5],dtype = np.float32).flatten(),float(score_),done,info

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
