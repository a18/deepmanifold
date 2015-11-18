#!/usr/bin/env python2

from __future__ import division
from __future__ import with_statement
from __future__ import print_function

import sys
import numpy as np
import scipy.optimize

def witness_fn(r,x,P,Q,rbf_var,weight):
  # r is D dim
  # x is D dim
  # P is N x D, source distribution
  # Q is M x D, target distribution
  assert r.ndim==1
  assert x.ndim==1
  assert P.ndim==2
  assert Q.ndim==2
  #print('r',r.shape,r.dtype,r.min(),r.max())
  #print('x',x.shape,x.dtype,x.min(),x.max())
  #print('P',P.shape,P.dtype,P.min(),P.max())
  #print('Q',Q.shape,Q.dtype,Q.min(),Q.max())
  N=P.shape[0]
  M=Q.shape[0]
  x=x+r
  xmP=x-P
  xmQ=x-Q
  #print('xmP',xmP.shape,xmP.dtype,xmP.min(),xmP.max())
  #print('xmQ',xmQ.shape,xmQ.dtype,xmQ.min(),xmQ.max())
  kxP=np.exp(-(xmP**2).sum(axis=1)/(2*rbf_var))
  kxQ=np.exp(-(xmQ**2).sum(axis=1)/(2*rbf_var))
  #print('kxP',kxP.shape,kxP.dtype,kxP.min(),kxP.max())
  #print('kxQ',kxQ.shape,kxQ.dtype,kxQ.min(),kxQ.max())
  assert kxP.shape==(N,)
  assert kxQ.shape==(M,)
  loss=kxP.sum()/N-kxQ.sum()/M+(r**2).sum()*weight
  grad=(-kxP.reshape(N,1)*xmP/N/rbf_var).sum(axis=0)+(kxQ.reshape(M,1)*xmQ/M/rbf_var).sum(axis=0)+2*r*weight
  print('loss',loss)
  print('grad',grad.shape,grad.dtype,grad.min(),grad.max())
  assert grad.shape==r.shape
  return loss,grad

def match_distribution(x,P,Q,weights,max_iter=5,rbf_var=1e4):
  print('x',x.shape,x.dtype,x.min(),x.max())
  print('P',P.shape,P.dtype,P.min(),P.max())
  print('Q',Q.shape,Q.dtype,Q.min(),Q.max())
  print('weights',weights)

  F=np.concatenate((P,Q),axis=0)
  print('F',F.shape,F.dtype,F.min(),F.max())
  sigma=np.std(F,axis=0)
  loc=F.mean(axis=0)
  assert sigma.min()>0
  x=(x-loc)/sigma
  P=(P-loc)/sigma
  Q=(Q-loc)/sigma
  x_0=x*sigma+loc
  
  x_result=[]
  r_result=[]
  for weight in weights:
    r=np.zeros_like(x)
    solver_type='CG'
    solver_param={'maxiter': max_iter, 'iprint': -1}
    r_opt=scipy.optimize.minimize(witness_fn,r,args=(x,P,Q,rbf_var,weight),method=solver_type,jac=True,options=solver_param).x
    print('r_opt',r_opt.shape,r_opt.dtype,r_opt.min(),r_opt.max())
    x_result.append((x+r_opt)*sigma+loc)
    r_result.append(r_opt*sigma)
  return x_0,np.asarray(x_result),np.asarray(r_result)

