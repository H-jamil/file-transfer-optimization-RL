import os
import time
import uuid
import socket
import warnings
import datetime
import numpy as np
import logging as log
import multiprocessing as mp
from threading import Thread
from concurrent.futures import ThreadPoolExecutor
from config import configurations

configurations["thread_limit"] = configurations["max_cc"]
configurations["cpu_count"] = mp.cpu_count()

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


class TransferServiceTest:
  def __init__(self,configurations,log):
    self.log=log
    self.root=configurations["data_dir"]
    self.file_names = os.listdir(self.root) * configurations["multiplier"]
    self.file_sizes = [os.path.getsize(self.root+filename) for filename in self.file_names]
    self.file_count = len(self.file_names)
    self.num_workers = mp.Value("i", 0)
    self.file_incomplete = mp.Value("i", self.file_count)
    self.process_status = mp.Array("i", [0 for i in range(configurations["thread_limit"])])
    self.file_offsets = mp.Array("d", [0.0 for i in range(self.file_count)])
    self.file_transfer=True
    self.transfer_status=False
    manager=mp.Manager()
    self.throughput_logs=manager.list()
    self.q = manager.Queue(maxsize=self.file_count)
    for i in range(self.file_count):
      self.q.put(i)

  def worker(self,process_id,q):
    while self.file_incomplete.value >0:
      if self.process_status[process_id] == 0:
        pass
      else:
        self.log.info(f"Start Process :: {process_id}")
        while (not q.empty()) and (self.process_status[process_id] == 1):
          file_id = q.get()
          self.log.info(f"Process {process_id} get item {file_id} from queue and executing")
          ####work
          time.sleep(1)
          self.file_offsets[file_id]+=1
          ####
          if (self.file_offsets[file_id] < 8):
            q.put(file_id)
          else:
            self.file_incomplete.value =self.file_incomplete.value - 1
            self.log.info(f"Process {process_id} finished on working on file {file_id} ")
            self.process_status[process_id] = 0
            self.log.info(f"Process {process_id} shutdown itself ")
            self.log.info("Process Status Bits are: {}".format(' '.join(map(str, self.process_status[:]))))

            # self.log.info("Process Status Bits are: {}".format(' '.join(map(str, self.process_status[:]))))

    self.process_status[process_id] =0
    self.log.info(f"Process {process_id} shutdown itself ")
    self.log.info("Process Status Bits are: {}".format(' '.join(map(str, self.process_status[:]))))


  def reset(self, configurations):
    self.root=configurations["data_dir"]
    self.file_names = os.listdir(self.root) * configurations["multiplier"]
    self.file_sizes = [os.path.getsize(self.root+filename) for filename in self.file_names]
    self.file_count = len(self.file_names)
    self.num_workers = mp.Value("i", 0)
    self.file_incomplete = mp.Value("i", self.file_count)
    self.process_status = mp.Array("i", [0 for i in range(configurations["thread_limit"])])
    self.file_offsets = mp.Array("d", [0.0 for i in range(self.file_count)])
    self.file_transfer=True
    self.transfer_status=False
    manager=mp.Manager()
    self.throughput_logs=manager.list()
    self.q = manager.Queue(maxsize=self.file_count)
    for i in range(self.file_count):
        self.q.put(i)
    return np.zeros([2,2],dtype = int)

  def run(self):
    workers = [mp.Process(target=self.worker, args=(i, self.q)) for i in range(configurations["thread_limit"])]
    for p in workers:
        p.daemon = True
        p.start()
    reporting_process = mp.Process(target=self.monitor)
    reporting_process.daemon = True
    reporting_process.start()
    return workers,reporting_process

  def change_concurrency(self, params):
    self.num_workers.value = params[0]
    current_cc = np.sum(self.process_status)
    for i in range(configurations["thread_limit"]):
      if i <params[0]:
        self.process_status[i] =1
      else:
        self.process_status[i] =0
    # time.sleep(1)
    self.log.info("Process Status Bits are: {}".format(' '.join(map(str, self.process_status[:]))))
    self.log.info("Active CC: {0}".format(np.sum(self.process_status)))

  def monitor(self):
    while(True):
      time.sleep(1)
      if self.file_incomplete.value==0:
        self.transfer_status=True

      self.log.info(f"total {np.sum(self.process_status)} process are working and total {np.sum(self.file_offsets)} chunks are done: Transfer Status {self.transfer_status}")



  def print_directory_details(self):
    print("file_names:",self.file_names)
    print("file_sizes:",self.file_sizes)
    print("file_count:",self.file_count)

    print("num_workers:",self.num_workers.value)
    print("file_incomplete:",self.file_incomplete.value)

    print("process_status:",self.process_status[:])
    print("file_offsets:",self.file_offsets[:])



if __name__=="__main__":

  transfer=TransferServiceTest(configurations,log)
  transfer.print_directory_details()
  print(transfer.reset(configurations))
  workers,reporting_process=transfer.run()
  transfer.change_concurrency([2])
  time.sleep(3)
  transfer.change_concurrency([8])
  time.sleep(10)
  transfer.change_concurrency([3])
  time.sleep(5)
  transfer.change_concurrency([6])
  time.sleep(8)
  for p in workers:
    if p.is_alive():
      p.terminate()
      p.join(timeout=0.1)
  if reporting_process.is_alive():
    reporting_process.terminate()
    reporting_process.join(timeout=0.1)

