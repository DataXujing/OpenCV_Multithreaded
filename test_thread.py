#!/usr/bin/python
#coding=utf-8
 
import threading
import time

from queue import Queue
 
q = Queue(10)
 
threadLock = threading.Lock()
class myThread(threading.Thread):
    def __init__(self,threadID,name):
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.name = name
        self.exitFlag = 0
    def run(self):
        while not self.exitFlag:
            threadLock.acquire()
            if not q.empty():
                id = q.get()
                print_time(self.name,id)
                threadLock.release()
            else:
                threadLock.release()
 
def print_time(threadName,id):
    # pass
    print("%s:%s:%s"%(threadName,time.ctime(time.time()),id))
 
# 创建3个线程
threads = []
for i in range(3):
    name =  "Thread-%d"%i
    t = myThread(i,name)
    t.start()
    threads.append(t)
 
 
for i in range(10000):
    q_name = "Queue:%d"%i
    print("================")
    print(q_name)
    q.put(q_name)
 
 
# 等待队列清空
while not q.empty():
    pass
 
# 也可以join方法，与上同效
# q.join()
 
# 通知线程，处理完之后关闭
for t in threads:
    t.exitFlag = 1
 
# 等待所有线程结束之后才退出
for t in threads:
    t.join()
 
print("Exiting Main Thread")
