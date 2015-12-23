#!/usr/bin/env python2

from __future__ import division
from __future__ import with_statement
from __future__ import print_function

import sys
import numpy as np
import scipy.optimize
import Queue
import threading
import minimize
import threadparallel

def witness_fn(r,x,P,Q,rbf_var,weight,kernel):
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
  if kernel=='l2':
    xmP=x-P
    xmQ=x-Q
    loss=0.5*((xmQ**2).sum()/M-(xmP**2).sum()/N)+(r**2).sum()*weight
    grad=xmQ.sum(axis=0)/M-xmP.sum(axis=0)/N+2*r*weight
  elif kernel=='rbf':
    if N>0:
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
    else:
      # source set is empty
      xmQ=x-Q
      kxQ=np.exp(-(xmQ**2).sum(axis=1)/(2*rbf_var))
      assert kxQ.shape==(M,)
      loss=kxQ.sum()/M+(r**2).sum()*weight
      grad=(kxQ.reshape(M,1)*xmQ/M/rbf_var).sum(axis=0)+2*r*weight
  else:
    raise ValueError('Unsupported kernel: {}'.format(kernel))
  #print('loss',loss)
  #print('grad',grad.shape,grad.dtype,grad.min(),grad.max())
  #print('r',r.shape,r.dtype,r.min(),r.max(),np.linalg.norm(r))
  assert grad.shape==r.shape
  return loss,grad

def match_distribution(x,P,Q,weights,max_iter=5,rbf_var=1e4):
  print('match_distribution()')
  print('x',x.shape,x.dtype,x.min(),x.max())
  print('P',P.shape,P.dtype,P.min(),P.max())
  print('Q',Q.shape,Q.dtype,Q.min(),Q.max())
  print('weights',weights)

  # z score
  F=np.concatenate((P,Q),axis=0)
  print('F',F.shape,F.dtype,F.min(),F.max())
  sigma=F.std()
  loc=F.mean()
  print('sigma',sigma)
  print('loc',loc)
  assert sigma>0
  x=(x-loc)/sigma
  P=(P-loc)/sigma
  Q=(Q-loc)/sigma
  x_0=x*sigma+loc
  print('x',x.shape,x.dtype,x.min(),x.max())
  print('P',P.shape,P.dtype,P.min(),P.max())
  print('Q',Q.shape,Q.dtype,Q.min(),Q.max())
  print('x_0',x_0.shape,x_0.dtype,x_0.min(),x_0.max())
  
  x_result=[]
  r_result=[]

  checkgrad=True
  parallel=10
  for weight in weights:
    r=np.zeros_like(x)

    # SciPy optimizers don't work
    #solver_type='BFGS'
    #solver_type='CG'
    #print('solver_type',solver_type)
    ##solver_param={'maxiter': max_iter, 'iprint': -1, 'gtol': 1e-7}
    #solver_param={'gtol': 1e-5}
    #r_opt=scipy.optimize.minimize(witness_fn,r,args=(x,P,Q,rbf_var,weight),method=solver_type,jac=True,options=solver_param).x
    #r_opt=scipy.optimize.fmin_cg(witness_fn_loss,r,fprime=witness_fn_grad,args=(x,P,Q,rbf_var,weight))
    if checkgrad:
      def f(*args):
        return witness_fn(*args)[0]
      def g(*args):
        return witness_fn(*args)[1]
      print('Checking gradient ...')
      print(scipy.optimize.check_grad(f,g,r[:10],*(x[:10],P[:10,:10],Q[:10,:10],rbf_var,weight,'rbf')))
    if parallel>1:
      assert (len(P) % parallel)==0
      assert (len(Q) % parallel)==0
      def witness_fn_parallel(r,x,P,Q,rbf_var,weight,kernel):
        result=threadparallel.unordered_parallel_call([witness_fn]*parallel,[(r,x,P[i*len(P)//parallel:(i+1)*len(P)//parallel],Q[i*len(Q)//parallel:(i+1)*len(Q)//parallel],rbf_var,weight,kernel) for i in range(parallel)],None)
        loss=sum(x[0] for x in result)
        grad=sum(x[1] for x in result)
        return loss,grad
      r_opt,loss_opt,iter_opt=minimize.minimize(r,witness_fn_parallel,(x,P,Q,rbf_var,weight,'rbf'),maxnumlinesearch=50,maxnumfuneval=None,red=1.0,verbose=True)
    else:
      r_opt,loss_opt,iter_opt=minimize.minimize(r,witness_fn,(x,P,Q,rbf_var,weight,'rbf'),maxnumlinesearch=50,maxnumfuneval=None,red=1.0,verbose=True)
    print('r_opt',r_opt.shape,r_opt.dtype,r_opt.min(),r_opt.max(),np.linalg.norm(r_opt))
    print(r_opt[:10])
    x_result.append((x+r_opt)*sigma+loc)
    r_result.append(r_opt*sigma)
  return x_0,np.asarray(x_result),np.asarray(r_result)

