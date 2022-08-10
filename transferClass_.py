import os
import math
import time
import socket
import datetime
import numpy as np
import logging as log
import pandas as pd
import multiprocessing as mp
import re
from config import configurations

class TransferClass_:
  def __init__(self,configurations,log,transfer_emulation=False):
    self.log=log ### for logging
    self.HOST, self.PORT = configurations["receiver"]["host"], configurations["receiver"]["port"]
    self.RCVR_ADDR = str(self.HOST) + ":" + str(self.PORT)
    self.root=configurations["data_dir"]
    self.file_names = os.listdir(self.root) * configurations["multiplier"]
    self.file_sizes = [os.path.getsize(self.root+filename) for filename in self.file_names]
    self.file_count = len(self.file_names)
    self.configurations=configurations
    self.chunk_size=  1 * 1024 * 1024
    self.B=int(configurations["B"])
    self.K=float(configurations["K"])
    self.num_workers = mp.Value("i", 0)
    self.file_incomplete = mp.Value("i", self.file_count)
    self.process_status = mp.Array("i", [0 for i in range(configurations["thread_limit"])])
    self.file_offsets = mp.Array("d", [0.0 for i in range(self.file_count)])
    self.transfer_status=mp.Value("i", 0)
    manager=mp.Manager()
    self.q = manager.Queue(maxsize=self.file_count)
    for i in range(self.file_count):
      self.q.put(i)
    self.throughput_logs=manager.list()
    self.transfer_throughput=mp.Value("i", 0)
    if transfer_emulation ==True:
      self.transfer_emu_status=mp.Value("i",1)
    else:
      self.transfer_emu_status=mp.Value("i",0)

  def worker(self,process_id,q):
    while self.file_incomplete.value > 0:
        if (self.process_status[process_id] == 0):
          pass
        elif (self.process_status[process_id]==1) and (q.empty()):
          pass
        else:
          self.log.info(f"Start Process :: {process_id}")
          try:
            sock = socket.socket()
            sock.settimeout(3)
            sock.connect((self.HOST, self.PORT))
            if self.transfer_emu_status.value ==1:
              target, factor = 50, 10
              max_speed = (target * 1000 * 1000)/8
              second_target, second_data_count = int(max_speed/factor), 0
            while (not q.empty()) and (self.process_status[process_id] == 1):
              try:
                file_id = q.get()
                self.log.info(f"Process {process_id} get item {file_id} from queue and executing")
              except:
                self.process_status[process_id] = 0
                self.log.info(f"Process {process_id} shutdown itself ")
                self.log.info(f"Process {process_id} failed to get item {file_id} from queue")
                break
              offset = self.file_offsets[file_id]
              to_send = self.file_sizes[file_id] - offset

              if (to_send > 0) and (self.process_status[process_id] == 1):
                filename = self.root + self.file_names[file_id]
                file = open(filename, "rb")
                msg = self.file_names[file_id] + "," + str(int(offset))
                msg += "," + str(int(to_send)) + "\n"
                try:
                  sock.send(msg.encode())
                  self.log.info("starting {0}, {1}, {2}".format(process_id, file_id, filename))
                except:
                  self.log.info(f"Process {process_id} failed to send file message")

                timer100ms = time.time()

                while (to_send > 0) and (self.process_status[process_id] == 1):
                  try:
                    if self.transfer_emu_status.value==1:
                      block_size = min(self.chunk_size, second_target-second_data_count)
                      data_to_send = bytearray(int(block_size))
                      sent = sock.send(data_to_send)
                    else:
                      block_size = int(min(self.chunk_size, to_send))
                      sent = sock.sendfile(file=file, offset=int(offset), count=block_size)

                  except Exception as e:
                    sent=0
                    # if self.transfer_emu_status.value == 1:
                      # to_send=0
                    self.process_status[process_id] = 0
                    self.log.info(f"Process {process_id} shutdown itself for socket error")
                    self.log.error("Process: {0}, Error from socket: {1}".format(process_id, str(e)))

                  offset += sent
                  to_send -= sent
                  self.file_offsets[file_id] = offset

                  if self.transfer_emu_status.value == 1:
                    second_data_count += sent
                    if second_data_count >= second_target:
                      second_data_count = 0
                      while timer100ms + (1/factor) > time.time():
                        pass
                      timer100ms = time.time()

              if to_send>0:
                q.put(file_id)
              else:
                self.file_incomplete.value =self.file_incomplete.value - 1
                self.log.info(f"Process {process_id} finished on working on file {file_id} ")
            sock.close()
          except socket.timeout as e:
            pass
          except Exception as e:
            self.log.info(f"Process {process_id} had error to send file ")
            self.process_status[process_id] = 0
            self.log.error("Process: {0}, Error: {1}".format(process_id, str(e)))
            self.log.info(f"Process {process_id} shutdown itself ")
    self.process_status[process_id] = 0
    self.log.info(f"Process {process_id} shutdown itself from outest loop")
    self.log.info("Process Status Bits are: {}".format(' '.join(map(str, self.process_status[:]))))

  def monitor(self,start_time):
    previous_total = 0
    previous_time = 0
    prev_sc,prev_rc=0,0
    timer320s=time.time()
    while self.file_incomplete.value > 0:
      t1 = time.time()
      time_since_begining = np.round(t1-start_time, 1)
      if time_since_begining >= 0.1:
        total_bytes = np.sum(self.file_offsets)
        # thrpt = np.round((total_bytes*8)/(time_since_begining*1000*1000), 2)
        curr_total = total_bytes - previous_total
        curr_time_sec = np.round(time_since_begining - previous_time, 3)
        curr_thrpt = np.round((curr_total*8)/(curr_time_sec*1000*1000), 2)
        previous_time, previous_total = time_since_begining, total_bytes
        cc_level=np.sum(self.process_status)
        record_list=[] ## will record curr_thrpt,cc_level,cwnd,rtt,packet_loss_rate,score,datetime
        cwnd_list,rtt_list,curr_sc,curr_rc=self.tcp_stats()
        record_list.append(curr_thrpt)
        record_list.append(cc_level)
        try:
          if len(cwnd_list)==0:
            record_list.append(self.throughput_logs[-1][3])
          else:
            cwnd=np.round(np.nanmean(cwnd_list),1)
            if math.isnan(cwnd):
              record_list.append(self.throughput_logs[-1][3])
            else:
              record_list.append(cwnd)
        except:
          cwnd=0.0
          record_list.append(cwnd)
        try:
          if len(rtt_list)==0:
            record_list.append(self.throughput_logs[-1][4])
          else:
            rtt=np.round(np.nanmean(rtt_list),2)
            if math.isnan(rtt):
              record_list.append(self.throughput_logs[-1][4])
            else:
              record_list.append(rtt)
        except:
          rtt=0.00
          record_list.append(rtt)
        sc, rc = curr_sc - prev_sc, curr_rc - prev_rc
        lr= 0
        if sc != 0:
          lr = rc/sc if sc>rc else 0
        if lr < 0:
          lr=0
        plr_impact = self.B*lr
        cc_impact_nl = self.K**cc_level
        score = (curr_thrpt/cc_impact_nl) - (curr_thrpt * plr_impact)
        score_value = np.round(score / 1000)
        prev_sc,prev_rc=curr_sc,curr_rc
        record_list.append(lr)
        record_list.append(score_value)
        record_list.append(datetime.datetime.now())
        self.throughput_logs.append(record_list)
        self.log.info("Throughput @{0}s:{1}Mbps, rtt :{2}ms cwnd: {3} lossRate: {4} CC:{5} score:{6} ".format(
            time_since_begining, curr_thrpt,rtt,cwnd,lr,cc_level,score_value))
        t2 = time.time()
        time.sleep(max(0, 1 - (t2-t1)))
        if (timer320s + 20 <= time.time()):
          self.file_incomplete.value=0

    self.transfer_status.value=1

  def change_concurrency(self, params):
    self.num_workers.value = params[0]
    # self.num_workers.value=np.argmax(params)
    for i in range(self.configurations["thread_limit"]):
      if i <params[0]:
      # if i <self.num_workers.value:
        self.process_status[i] =1
      else:
        self.process_status[i] =0
    self.log.info("Process Status Bits from change concurrency are: {}".format(' '.join(map(str, self.process_status[:]))))
    self.log.info("Active CC: {0}".format(np.sum(self.process_status)))

  def tcp_stats(self):
    cwnd_list=[]
    rtt_list=[]
    sent, retm = 0, 0
    start = time.time()
    try:
      data = os.popen("ss -ti").read().split("\n")
      for i in range(1,len(data)):
          if self.RCVR_ADDR in data[i-1]:
              parse_data = data[i].split(" ")
              for entry in parse_data:
                if "minrtt" in entry:
                  continue
                else:
                  if "cwnd_gain" in entry:
                    continue
                  if "cwnd" in entry:
                      # cwnd_value=int(entry.split(":")[-1])
                      cwnd_list.append(int(entry.split(":")[-1]))
                  if "rtt" in entry:
                      try:
                          rtt_list.append(float(re.findall(r":(.*?)/", entry)[0]))
                      except:
                          self.log.info("rtt can't be calculated from tcp_stats() ***")
                          rtt_list.append(0)
                  if "data_segs_out" in entry:
                      sent += int(entry.split(":")[-1])
                  if "bytes_retrans" in entry:
                      continue
                  if "retrans" in entry:
                      retm += int(entry.split("/")[-1])
    except Exception as e:
      print(e)

    end = time.time()
    self.log.info("Time taken to collect tcp stats: {0}ms".format(np.round((end-start)*1000)))
    return cwnd_list,rtt_list,sent,retm

  def run(self):
    workers = [mp.Process(target=self.worker, args=(i, self.q)) for i in range(self.configurations["thread_limit"])]
    for p in workers:
      p.daemon = True
      p.start()
    reporting_process = mp.Process(target=self.monitor,args=(time.time(),))
    reporting_process.daemon = True
    reporting_process.start()
    self.transfer_status.value=0
    return workers,reporting_process

  def reset(self):
    self.num_workers = mp.Value("i", 0)
    self.file_incomplete = mp.Value("i", self.file_count)
    self.process_status = mp.Array("i", [0 for i in range(self.configurations["thread_limit"])])
    self.file_offsets = mp.Array("d", [0.0 for i in range(self.file_count)])
    self.transfer_status=mp.Value("i", 0)
    self.transfer_throughput=0
    manager=mp.Manager()
    self.q = manager.Queue(maxsize=self.file_count)
    for i in range(self.file_count):
      self.q.put(i)
    self.throughput_logs=manager.list()
    return np.zeros([3,6],dtype = np.float32)#curr_thrpt,cc_level,cwnd,rtt,packet_loss_rate,score

