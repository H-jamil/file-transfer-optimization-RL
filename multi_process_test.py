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
HOST, PORT = configurations["receiver"]["host"], configurations["receiver"]["port"]
RCVR_ADDR = str(HOST) + ":" + str(PORT)
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
    self.HOST, self.PORT = configurations["receiver"]["host"], configurations["receiver"]["port"]
    self.RCVR_ADDR = str(HOST) + ":" + str(PORT)
    self.root=configurations["data_dir"]
    self.file_names = os.listdir(self.root) * configurations["multiplier"]
    self.file_sizes = [os.path.getsize(self.root+filename) for filename in self.file_names]
    self.file_count = len(self.file_names)
    self.chunk_size=  1 * 1000 * 1000
    self.num_workers = mp.Value("i", 0)
    self.file_incomplete = mp.Value("i", self.file_count)
    self.process_status = mp.Array("i", [0 for i in range(configurations["thread_limit"])])
    self.file_offsets = mp.Array("d", [0.0 for i in range(self.file_count)])
    self.file_transfer=mp.Value("i", 0)
    self.transfer_status=mp.Value("i", 0)
    manager=mp.Manager()
    self.throughput_logs=manager.list()
    self.q = manager.Queue(maxsize=self.file_count)
    for i in range(self.file_count):
      self.q.put(i)

  def worker(self,process_id,q):
    while (self.file_incomplete.value > 0): #and (self.process_status[process_id] == 1)
      if self.process_status[process_id] == 0:
        pass
      else:
        self.log.info(f"Start Process :: {process_id}")
        try:
          sock=socket.socket()
          sock.settimeout(3)
          sock.connect((self.HOST,self.PORT))
          self.log.info(f"Process {process_id} is connected")
          while (not q.empty()) and (self.process_status[process_id] == 1):
            try:
              file_id = q.get()
              self.log.info(f"Process {process_id} get item {file_id} from queue and executing")
            except:
              self.process_status[process_id] = 0
              self.log.info(f"Process {process_id} shutdown itself ")
              self.log.info(f"Process {process_id} failed to get item {file_id} from queue")
              break
            offset=self.file_offsets[file_id]
            to_send=self.file_sizes[file_id]- offset
            if (to_send > 0) and (self.process_status[process_id] == 1):
              filename = self.root + self.file_names[file_id]
              file = open(filename, "rb")
              msg = self.file_names[file_id] + "," + str(int(offset))
              msg += "," + str(int(to_send)) + "\n"
              sock.send(msg.encode())
              self.log.info("starting {0}, {1}, {2}".format(process_id, file_id, filename))
              ####work
              while (to_send > 0) and (self.process_status[process_id] == 1):
                block_size = min(self.chunk_size, to_send)
                if block_size > 999999:
                  sent = sock.sendfile(file=file, offset=int(offset), count=block_size)
                  offset += sent
                  to_send -= sent
                  self.file_offsets[file_id] = offset
                else:
                  to_send=0
              self.file_incomplete.value =self.file_incomplete.value - 1
              self.log.info(f"Process {process_id} finished on working on file {file_id} ")
              self.process_status[process_id] = 0
              sock.close()
              # time.sleep(1)
              self.log.info(f"Process {process_id} shutdown itself ")
              self.log.info("Process Status Bits are: {}".format(' '.join(map(str, self.process_status[:]))))

        except socket.timeout as e:
          sock.close()
          pass

        except Exception as e:
          self.log.info(f"Process {process_id} had error to send file ")
          self.process_status[process_id] = 0
          sock.close()
          self.file_incomplete.value =self.file_incomplete.value - 1
          self.log.info("Process: {0}, decreasing file count to protect program ****".format(process_id, str(e)))
          self.log.error("Process: {0}, Error: {1}".format(process_id, str(e)))
          self.log.info(f"Process {process_id} shutdown itself ")
          self.log.info("Process Status Bits are: {}".format(' '.join(map(str, self.process_status[:]))))

    self.process_status[process_id] = 0
    self.log.info(f"Process {process_id} shutdown itself outest side")
    self.log.info("Process Status Bits are: {}".format(' '.join(map(str, self.process_status[:]))))

  def tcp_stats(self):
    start = time.time()
    sent, retm = 0, 0
    try:
        data = os.popen("ss -ti").read().split("\n")
        for i in range(1,len(data)):
            if self.RCVR_ADDR in data[i-1]:
                parse_data = data[i].split(" ")
                for entry in parse_data:
                    if "data_segs_out" in entry:
                        sent += int(entry.split(":")[-1])
                    if "bytes_retrans" in entry:
                        pass
                    elif "retrans" in entry:
                        retm += int(entry.split("/")[-1])
    except Exception as e:
        print("From tcp_stat()",e)

    end = time.time()
    log.debug("Time taken to collect tcp stats: {0}ms".format(np.round((end-start)*1000)))
    return sent, retm

  def reset(self, configurations):
    self.root=configurations["data_dir"]
    self.HOST, self.PORT = configurations["receiver"]["host"], configurations["receiver"]["port"]
    self.file_names = os.listdir(self.root) * configurations["multiplier"]
    self.file_sizes = [os.path.getsize(self.root+filename) for filename in self.file_names]
    self.file_count = len(self.file_names)
    self.chunk_size=  1 * 1024 * 1024
    self.num_workers = mp.Value("i", 0)
    self.file_incomplete = mp.Value("i", self.file_count)
    self.process_status = mp.Array("i", [0 for i in range(configurations["thread_limit"])])
    self.file_offsets = mp.Array("d", [0.0 for i in range(self.file_count)])
    self.file_transfer=mp.Value("i", 0)
    self.transfer_status=mp.Value("i", 0)
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
    reporting_process = mp.Process(target=self.monitor,args=(time.time(),))
    reporting_process.daemon = True
    reporting_process.start()
    self.file_transfer.value=1
    self.transfer_status.value=0
    return workers,reporting_process

  def change_concurrency(self, params):
    self.num_workers.value = params[0]
    current_cc = np.sum(self.process_status)
    for i in range(configurations["thread_limit"]):
      if i <params[0]:
        self.process_status[i] =1
      else:
        self.process_status[i] =0
    self.log.info("Process Status Bits are: {}".format(' '.join(map(str, self.process_status[:]))))
    self.log.info("Active CC: {0}".format(np.sum(self.process_status)))

  def monitor(self,start_time):
    previous_total = 0
    previous_time = 0
    while self.file_incomplete.value > 0:
        t1 = time.time()
        time_since_begining = np.round(t1-start_time, 1)
        if time_since_begining >= 0.1:
            total_bytes = np.sum(self.file_offsets)
            thrpt = np.round((total_bytes*8)/(time_since_begining*1000*1000), 2)
            curr_total = total_bytes - previous_total
            curr_time_sec = np.round(time_since_begining - previous_time, 3)
            curr_thrpt = np.round((curr_total*8)/(curr_time_sec*1000*1000), 2)
            previous_time, previous_total = time_since_begining, total_bytes
            self.throughput_logs.append(curr_thrpt)
            m_avg = np.round(np.mean(self.throughput_logs[-60:]), 2)
            self.log.info("Throughput @{0}s: Current: {1}Mbps, Average: {2}Mbps, 60Sec_Average: {3}Mbps".format(
                time_since_begining, curr_thrpt, thrpt, m_avg))
            self.log.info(f"total {np.sum(self.process_status)} process are working ") # and total {np.sum(self.file_offsets)} chunks are done: Transfer Status {self.transfer_status.value}
            t2 = time.time()
            time.sleep(max(0, 1 - (t2-t1)))

    self.file_transfer.value=0
    self.transfer_status.value=1

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
  while(transfer.file_incomplete.value is not 0):
    if np.sum(transfer.process_status) == 0:
      transfer.change_concurrency([2])
    pass
  for p in workers:
    if p.is_alive():
      p.terminate()
      p.join(timeout=0.1)
  if reporting_process.is_alive():
    reporting_process.terminate()
    reporting_process.join(timeout=0.1)
