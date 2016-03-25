#!/usr/bin/env python3

import sys
import threading
if sys.version_info.major==2:
  from Queue import Queue
else:
  from queue import Queue

__doc__='''
Parallel utility functions that use threading. They will be useless
useless the parallel code releases the GIL.
'''


class MonitorClosed(Exception):
  pass

class Monitor:
  '''>>> m=Monitor()
>>> q=Queue()
>>> w1=Worker(m)
>>> w2=Worker(m)
>>> threading.active_count()
3
>>> def f(x): return x**2
>>> for i,x in enumerate([2,3,4]): m.put(f,[x],{},i,q)
>>> m.close()
>>> m.put(f,[5],{},3,q)
Traceback (most recent call last):
    ...
MonitorClosed
>>> sorted([q.get(),q.get(),q.get()],key=lambda x: x[0])
[(0, 4), (1, 9), (2, 16)]
>>> threading.active_count()
1
'''
  def __init__(self):
    self.cv=threading.Condition()
    self.job=None
    self.closed=False
  def close(self):
    # Future monitor operations will raise MonitorClosed.
    with self.cv:
      self.closed=True
      self.cv.notifyAll()
  def put(self,f,s,k,i,q):
    # Blocks until the monitor is closed or the job has been started.
    # Raises MonitorClosed if the monitor has been closed. This is also
    # an indication that the job may have never been queued.
    with self.cv:
      while not self.closed and not (self.job is None):
        self.cv.wait()
      if not self.closed and self.job is None:
        self.job=(f,s,k,i,q)
        self.cv.notifyAll()
        while not self.closed and not (self.job is None):
          self.cv.wait()
    if self.closed:
      raise MonitorClosed
  def get(self):
    # Blocks until the monitor is closed or a job is taken.
    # Raises MonitorClosed if the monitor has been closed. This is also
    # an indication that the job may have been discarded.
    with self.cv:
      while not self.closed and not (self.job is not None):
        self.cv.wait()
      if not self.closed and self.job is not None:
        result=self.job
        self.job=None
        self.cv.notifyAll()
    if self.closed:
      raise MonitorClosed
    return result

class Worker(threading.Thread):
  def __init__(self,m):
    threading.Thread.__init__(self)
    self.monitor=m
    self.start()
  def run(self):
    while True:
      try:
        f,s,k,i,q=self.monitor.get()
        q.put((i,f(*s,**k)))
      except MonitorClosed:
        return

def unordered_parallel_call(F,S,K,seq=list,pool=None,thread_init=None):
  '''
F is a list of functions
S is None or an iterable of positional arguments
K is None or an iterable of keyword arguments

F is a callable
S is a list, otherwise K is a list
S is None or an iterable of positional arguments
K is None or an iterable of keyword arguments

Returns a list of out-of-order results.

>>> S=[4,7,2,4,5]
>>> sorted(unordered_parallel_call(lambda x: x**2,[[x] for x in S],None))
[4, 16, 16, 25, 49]
>>> sorted(unordered_parallel_call(lambda x: x**2,[[x] for x in S],None,pool=2))
[4, 16, 16, 25, 49]
'''
  try:
    _=len(F)
  except:
    assert not S is None or not K is None
    F=[F]*(len(S) if S else len(K))
  assert S is None or len(F)==len(S)
  assert K is None or len(F)==len(K)
  if S is None:
    S=[[]]*len(F)
  if K is None:
    K=[{}]*len(F)
  q=Queue()
  if pool is None:
    def call_f(f,s,k):
      q.put(f(*s,**k))
    allt=[]
    for f,s,k in zip(F,S,K):
      t=threading.Thread(target=call_f,args=(f,s,k))
      allt.append(t)
    if thread_init is not None:
      thread_init(allt)
    for t in allt:
      t.start()
    return seq(q.get() for i in range(len(F)))
  else:
    m=Monitor()
    allt=[Worker(m) for i in range(pool)]
    if thread_init is not None:
      thread_init(allt)
    for f,s,k in zip(F,S,K):
      m.put(f,s,k,None,q)
    m.close()
    for t in allt:
      t.join()
    return seq(q.get()[1] for i in range(len(F)))

def ordered_parallel_call(F,S,K,seq=list,pool=None,thread_init=None):
  '''
F is a list of functions
S is None or an iterable of positional arguments
K is None or an iterable of keyword arguments

F is a callable
S is a list, otherwise K is a list
S is None or an iterable of positional arguments
K is None or an iterable of keyword arguments

Returns a list of ordered results.

>>> S=[4,7,2,4,5]
>>> ordered_parallel_call(lambda x: x**2,[[x] for x in S],None)
[16, 49, 4, 16, 25]
>>> ordered_parallel_call(lambda x: x**2,[[x] for x in S],None,pool=2)
[16, 49, 4, 16, 25]
'''
  try:
    _=len(F)
  except:
    assert not S is None or not K is None
    F=[F]*(len(S) if S else len(K))
  assert S is None or len(F)==len(S)
  assert K is None or len(F)==len(K)
  if S is None:
    S=[[]]*len(F)
  if K is None:
    K=[{}]*len(F)
  q=Queue()
  if pool is None:
    def call_f(i,f,s,k):
      q.put((i,f(*s,**k)))
    allt=[]
    for i,(f,s,k) in enumerate(zip(F,S,K)):
      t=threading.Thread(target=call_f,args=(i,f,s,k))
      allt.append(t)
    if thread_init is not None:
      thread_init(allt)
    for t in allt:
      t.start()
    for t in allt:
      t.join()
    return seq(x[1] for x in sorted([q.get() for i in range(len(F))],key=lambda x: x[0]))
  else:
    m=Monitor()
    allt=[Worker(m) for i in range(pool)]
    if thread_init is not None:
      thread_init(allt)
    for i,(f,s,k) in enumerate(zip(F,S,K)):
      m.put(f,s,k,i,q)
    m.close()
    for t in allt:
      t.join()
    return seq(x[1] for x in sorted([q.get() for i in range(len(F))],key=lambda x: x[0]))

if __name__=='__main__':
  import doctest
  doctest.testmod(verbose=False)

  print('Success.')
