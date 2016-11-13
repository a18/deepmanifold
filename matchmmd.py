#!/usr/bin/env python2

from __future__ import division
from __future__ import with_statement
from __future__ import print_function

import sys
import numpy as np
import scipy.optimize
import Queue
import time
import threading
import minimize
import threadparallel
import math

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
  #print('loss',loss)
  #print('grad',grad.shape,grad.dtype,grad.min(),grad.max())
  #print('r',r.shape,r.dtype,r.min(),r.max(),np.linalg.norm(r))
  assert grad.shape==r.shape
  return loss,grad

def match_distribution(x,P,Q,weights,max_iter=5,rbf_var=1e4,maxnumlinesearch=50):
  print('match_distribution()')
  print('x',x.shape,x.dtype,x.min(),x.max())
  print('P',P.shape,P.dtype)
  print('Q',Q.shape,Q.dtype)
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
  print('P',P.shape,P.dtype)
  print('Q',Q.shape,Q.dtype)
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
      print(scipy.optimize.check_grad(f,g,r[:10],*(x[:10],P[:10,:10],Q[:10,:10],rbf_var,weight)))
    if parallel>1:
      assert (len(P) % parallel)==0
      assert (len(Q) % parallel)==0
      def witness_fn_parallel(r,x,P,Q,rbf_var,weight):
        result=threadparallel.unordered_parallel_call([witness_fn]*parallel,[(r,x,P[i*len(P)//parallel:(i+1)*len(P)//parallel],Q[i*len(Q)//parallel:(i+1)*len(Q)//parallel],rbf_var,weight) for i in range(parallel)],None)
        loss=sum(x[0] for x in result)
        grad=sum(x[1] for x in result)
        return loss,grad
      r_opt,loss_opt,iter_opt=minimize.minimize(r,witness_fn_parallel,(x,P,Q,rbf_var,weight,'rbf'),maxnumlinesearch=maxnumlinesearch,maxnumfuneval=None,red=1.0,verbose=True)
    else:
      r_opt,loss_opt,iter_opt=minimize.minimize(r,witness_fn,(x,P,Q,rbf_var,weight),maxnumlinesearch=maxnumlinesearch,maxnumfuneval=None,red=1.0,verbose=True)
    print('r_opt',r_opt.shape,r_opt.dtype,r_opt.min(),r_opt.max(),np.linalg.norm(r_opt))
    print(r_opt[:10])
    x_result.append((x+r_opt)*sigma+loc)
    r_result.append(r_opt*sigma)
  return x_0,np.asarray(x_result),np.asarray(r_result)

def witness_fn3(r,x,FFT,BP,BQ,CP,CQ,N,M,L,rbf_var,weight,verbose,checkrbf):
  # K is N+M+L+1
  # r is K dim, indicator vector
  # x is K dim, indicator vector
  # F is K x D, latent space vectors
  # FFT is F F^T
  K=N+M+L+1
  assert r.shape==(K,)
  assert x.shape==(K,)
  assert FFT.shape==(K,K)

  P=np.eye(N,K)
  Q=np.concatenate([np.zeros((M,N)),np.eye(M,M+L+1)],axis=1)

  xpr=x+r
  Z=(-1.0/(2*rbf_var))
  FFTxpr=FFT.dot(xpr.T)
  xprFFTxpr=xpr.dot(FFTxpr)
  eP=Z*(xprFFTxpr+CP-2.0*xpr.dot(BP))
  eQ=Z*(xprFFTxpr+CQ-2.0*xpr.dot(BQ))

  KP=np.exp(eP)
  KQ=np.exp(eQ)
  B=FFT.dot(r)

  loss=(1.0/N)*KP.sum()-(1.0/M)*KQ.sum()+weight*(r.dot(B))
  grad=(1.0/N)*2.0*Z*(KP.reshape(N,1)*(FFTxpr-BP.T)).sum(axis=0)-(1.0/M)*2.0*Z*(KQ.reshape(M,1)*(FFTxpr-BQ.T)).sum(axis=0)+2*weight*B

  # reference implementation
  #eP=np.array([Z*((x+r-P[i]).dot(FFT).dot((x+r-P[i]).T)) for i in range(N)])
  #eQ=np.array([Z*((x+r-Q[i]).dot(FFT).dot((x+r-Q[i]).T)) for i in range(M)])
  #KP=np.exp(eP)
  #KQ=np.exp(eQ)
  #loss=(1.0/N)*KP.sum()-(1.0/M)*KQ.sum()+weight*r.dot(FFT).dot(r)
  #grad=(1.0/N)*(KP.reshape(N,1)*2.0*Z*np.array([FFT.dot((x+r-P[i]).T) for i in range(N)])).sum(axis=0)-(1.0/M)*(KQ.reshape(M,1)*2.0*Z*np.array([FFT.dot((x+r-Q[i]).T) for i in range(M)])).sum(axis=0)+2*weight*FFT.dot(r)

  if checkrbf:
    if eP.mean()<-10 or eQ.mean()<-10:
      print('WARNING: rbf_var is too small (eP.mean()={}, eQ.mean={})'.format(eP.mean(),eQ.mean()))
    print('KP',KP[:5],KP.mean(),KP.var())
    print('KQ',KQ[:5],KQ.mean(),KQ.var())
  if verbose:
    print('loss',loss)
    print('grad',grad.shape,grad.dtype,grad.min(),grad.max())
  assert grad.shape==r.shape
  return loss,grad

def witness_fn3_KQ(r,x,FFT,BQ,CQ,N,M,L,rbf_var):
  # K is N+M+L+1
  # r is K dim, indicator vector
  # x is K dim, indicator vector
  # F is K x D, latent space vectors
  # FFT is F F^T
  K=N+M+L+1
  assert r.shape==(K,)
  assert x.shape==(K,)
  assert FFT.shape==(K,K)

  Q=np.concatenate([np.zeros((M,N)),np.eye(M,M+L+1)],axis=1)

  xpr=x+r
  Z=(-1.0/(2*rbf_var))
  FFTxpr=FFT.dot(xpr.T)
  xprFFTxpr=xpr.dot(FFTxpr)
  eQ=Z*(xprFFTxpr+CQ-2.0*xpr.dot(BQ))

  KQ=np.exp(eQ)

  return KQ

def zscore_F(F):
  # in place, zero copy
  # F is K x D
  #print('F',F.shape,F.dtype,F.min(),F.max())
  assert F.ndim==2
  loc=F.mean(axis=0)
  sigma=np.empty_like(loc)
  for i in range(0,F.shape[1],10000):
    j=i+10000
    sigma[i:j]=F[:,i:j].std(axis=0)
  sigma[sigma<1e-10]=1
  F-=loc
  F/=sigma.reshape(1,-1)
  return loc,sigma

def manifold_traversal2(FFT,N,M,L,weights,max_iter=5,rbf_var=1e4,verbose=False,checkgrad=True,checkrbf=True,maxnumlinesearch=25,initialize_KQ=None):
  # returns two arrays, xpr and r
  #   xpr is optimized x+r
  #   r is optimized r
  # multiply by F to get latent space vector
  if verbose:
    print('manifold_traversal2()')
    print('FFT',FFT.shape,FFT.dtype,FFT.min(),FFT.max())
    print('N',N)
    print('M',M)
    print('L',L)
    print('weights',weights)

  #FFT=F.dot(F.T) # K x K
  xpr_result=[]
  r_result=[]
  r=np.zeros(len(FFT))
  x=np.zeros(len(FFT))
  x[-1]=1
  K=N+M+L+1
  P=np.eye(N,K)
  Q=np.concatenate([np.zeros((M,N)),np.eye(M,M+L+1)],axis=1)
  BP=FFT[:,:N] # FFT.dot(P.T) # K x N
  BQ=FFT[:,N:N+M] # FFT.dot(Q.T) # K x M
  CP=np.array([FFT[i,i] for i in range(N)]) # np.array([P[i].dot(FFT).dot(P[i].T) for i in range(N)])
  CQ=np.array([FFT[N+i,N+i] for i in range(M)]) # np.array([Q[i].dot(FFT).dot(Q[i].T) for i in range(M)])

  if not initialize_KQ is None:
    assert initialize_KQ>0 and initialize_KQ<1
    KQ=witness_fn3_KQ(r,x,FFT,BQ,CQ,N,M,L,rbf_var)
    rbf_var*=math.log(KQ.mean())/math.log(initialize_KQ)
    if verbose:
      print('Setting sigma^2 = {}'.format(rbf_var))

  for weight in weights:

    if checkgrad and weight==weights[0]:
      def f(*args):
        return witness_fn3(*args)[0]
      def g(*args):
        return witness_fn3(*args)[1]
      print('Checking gradient ...')
      est_grad=scipy.optimize.approx_fprime(r,f,math.sqrt(np.finfo(float).eps),*(x,FFT,BP,BQ,CP,CQ,N,M,L,rbf_var,weight,False,False))
      #print('est. gradient',est_grad)
      fn_grad=g(r,x,FFT,BP,BQ,CP,CQ,N,M,L,rbf_var,weight,False,True)
      #print('gradient',fn_grad)
      #print('isclose',np.isclose(est_grad,fn_grad,rtol=1e-4,atol=1e-7))
      assert np.allclose(est_grad,fn_grad,rtol=1e-4,atol=1e-5)
      #err=scipy.optimize.check_grad(f,g,r,*(x,FFT,BP,BQ,CP,CQ,N,M,L,rbf_var,weight,False,False))
      #print('gradient error',err)
      #assert err<1e-5
      print('passed.')

    t0=time.time()
    r_opt,loss_opt,iter_opt=minimize.minimize(r,witness_fn3,(x,FFT,BP,BQ,CP,CQ,N,M,L,rbf_var,weight,verbose,checkrbf),maxnumlinesearch=maxnumlinesearch,maxnumfuneval=None,red=1.0,verbose=False)
    t1=time.time()
    if verbose:
      #print('r_opt',r_opt.shape,r_opt.dtype)
      print('r_opt mean P value',r_opt[:N].mean(),r_opt[:N].var())
      print('r_opt mean Q value',r_opt[N:N+M].mean(),r_opt[N:N+M].var())
      if L>0:
        print('r_opt mean T value',r_opt[N+M:N+M+L].mean(),r_opt[N+M:N+M+L].var())
      print('r_opt X value',r_opt[-1])
      print('Optimized in {} minutes.'.format((t1-t0)/60.0))
    xpr_result.append(x+r_opt)
    r_result.append(r_opt)
    r=r_opt
  return np.asarray(xpr_result),np.asarray(r_result)

if __name__=='__main__':
  # test settings
  N=2
  M=2
  L=0
  D=2
  F=np.arange(1,11).reshape(5,2)
  rbf_var=0.5e2
  weight=0.1
  # aux vars
  FFT=F[:N+M+L+1].dot(F[:N+M+L+1].T)
  x=F[-1]
  nv=F[:N+M+L].dot(x)
  FFT[:-1,-1]=nv
  FFT[-1,:-1]=nv
  FFT[-1,-1]=x.dot(x)
  r=np.zeros(len(FFT))
  x=np.zeros(len(FFT))
  x[-1]=1
  K=N+M+L+1
  P=np.eye(N,K)
  Q=np.concatenate([np.zeros((M,N)),np.eye(M,M+L+1)],axis=1)
  BP=FFT[:,:N] # FFT.dot(P.T) # K x N
  BQ=FFT[:,N:N+M] # FFT.dot(Q.T) # K x M
  CP=np.array([FFT[i,i] for i in range(N)]) # np.array([P[i].dot(FFT).dot(P[i].T) for i in range(N)])
  CQ=np.array([FFT[N+i,N+i] for i in range(M)]) # np.array([Q[i].dot(FFT).dot(Q[i].T) for i in range(M)])
  loss,grad=witness_fn3(r,x,FFT,BP,BQ,CP,CQ,N,M,L,rbf_var,weight,False,True)
  # solution
  # eP = -1.28, -0.72
  # eQ = -0.32, -0.08
  # loss = -0.44223791352358049
  # grad = -0.01181949294952045, -0.02757881688221442, -0.04333814081490828, -0.059097464747602246, -0.074856788680296
  assert np.allclose(loss,-0.44223791352358049),loss
  assert np.allclose(grad,[-0.01181949294952045, -0.02757881688221442, -0.04333814081490828, -0.059097464747602246, -0.074856788680296]),grad

  N=6
  M=4
  L=2
  D=20
  P=np.random.random((N,D))+0.8
  Q=np.random.random((M,D))-0.8
  T=np.random.random((L,D))
  X=np.random.random((D,))
  F=np.concatenate([P,Q,T,X.reshape(1,D)])
  rbf_var=1e1
  weight=1e-2
  r=np.zeros(len(F))
  x=np.zeros(len(F))
  x[-1]=1
  FFT=F.dot(F.T) # K x K
  K=N+M+L+1
  P=np.eye(N,K)
  Q=np.concatenate([np.zeros((M,N)),np.eye(M,M+L+1)],axis=1)
  BP=FFT[:,:N] # FFT.dot(P.T) # K x N
  BQ=FFT[:,N:N+M] # FFT.dot(Q.T) # K x M
  CP=np.array([FFT[i,i] for i in range(N)]) # np.array([P[i].dot(FFT).dot(P[i].T) for i in range(N)])
  CQ=np.array([FFT[N+i,N+i] for i in range(M)]) # np.array([Q[i].dot(FFT).dot(Q[i].T) for i in range(M)])
  def f(*args):
    return witness_fn3(*args)[0]
  def g(*args):
    return witness_fn3(*args)[1]
  print('Checking gradient ...')
  err=scipy.optimize.check_grad(f,g,r,*(x,FFT,BP,BQ,CP,CQ,N,M,L,rbf_var,weight,False,True))
  print('gradient error',err)
  assert err<1e-5
  r_opt,loss_opt,iter_opt=minimize.minimize(r,witness_fn3,(x,FFT,BP,BQ,CP,CQ,N,M,L,rbf_var,weight,False,True),maxnumlinesearch=25,maxnumfuneval=None,red=1.0,verbose=True)
  print('r P',r_opt[:N],r_opt[:N].var())
  print('r Q',r_opt[N:N+M],r_opt[N:N+M].var())
  print('r T',r_opt[N+M:N+M+L],r_opt[N+M:N+M+L].var())
  print('r X',r_opt[-1])
  xhat=(x+r_opt).dot(F)
  print('xhat',xhat)
  assert sum(xhat<0)>sum(xhat>0)

  # TODO test a multimodal Q
