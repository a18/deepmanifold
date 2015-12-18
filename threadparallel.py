#!/usr/bin/env python2

from __future__ import division
from __future__ import with_statement
from __future__ import print_function

import Queue
import threading

__doc__='''
Parallel utility functions that use threading. They will be useless
useless the parallel code releases the GIL.
'''

def unordered_parallel_call(F,S,K,seq=list):
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
  q=Queue.Queue()
  def call_f(f,s,k):
    q.put(f(*s,**k))
  allt=[]
  for f,s,k in zip(F,S,K):
    t=threading.Thread(target=call_f,args=(f,s,k))
    t.start()
    allt.append(t)
  for t in allt:
    t.join()
  return seq(q.get() for i in range(len(F)))

def ordered_parallel_call(F,S,K,seq=list):
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
  q=Queue.Queue()
  def call_f(i,f,s,k):
    q.put((i,f(*s,**k)))
  allt=[]
  for i,(f,s,k) in enumerate(zip(F,S,K)):
    t=threading.Thread(target=call_f,args=(i,f,s,k))
    t.start()
    allt.append(t)
  for t in allt:
    t.join()
  return seq(x[1] for x in sorted([q.get() for i in range(len(F))],key=lambda x: x[0]))

if __name__=='__main__':
  import doctest
  doctest.testmod(verbose=True)

  print('Success.')
