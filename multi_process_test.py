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

class TransferServiceTest:
  def __init__(self,configurations):
    self.root=configurations["data_dir"]
    self.file_names = os.listdir(self.root) * configurations["multiplier"]
    self.file_sizes = [os.path.getsize(self.root+filename) for filename in self.file_names]
    self.file_count = len(self.file_names)
    self.num_workers = mp.Value("i", 0)
    self.file_incomplete = mp.Value("i", self.file_count)
    self.process_status = mp.Array("i", [0 for i in range(configurations["thread_limit"])])
    self.file_offsets = mp.Array("d", [0.0 for i in range(self.file_count)])

  def print_directory_details(self):
    print("file_names:",self.file_names)
    print("file_sizes:",self.file_sizes)
    print("file_count:",self.file_count)

    print("num_workers:",self.num_workers.value)
    print("file_incomplete:",self.file_incomplete.value)

    print("process_status:",self.process_status[:])
    print("file_offsets:",self.file_offsets[:])



if __name__=="__main__":
  transfer=TransferServiceTest(configurations)
  transfer.print_directory_details()
# configurations["thread_limit"] = configurations["max_cc"]
# root=configurations["data_dir"]
# file_names = os.listdir(root) * configurations["multiplier"]
# file_sizes = [os.path.getsize(root+filename) for filename in file_names]
# file_count = len(file_names)

# num_workers = mp.Value("i", 0)
# file_incomplete = mp.Value("i", file_count)
# process_status = mp.Array("i", [0 for i in range(configurations["thread_limit"])])
# file_offsets = mp.Array("d", [0.0 for i in range(file_count)])

# print("file_names:",file_names)
# print("file_sizes:",file_sizes)
# print("file_count:",file_count)

# print("num_workers:",num_workers.value)
# print("file_incomplete:",file_incomplete.value)

# print("process_status:",process_status[:])
# print("file_offsets:",file_offsets[:])
# time.sleep(2)
