
import numpy

def montage(M,sep=0):
  # row X col X H X W X C
  assert M.ndim==5
  canvas=numpy.zeros((M.shape[0]*M.shape[2]+(M.shape[0]-1)*sep,M.shape[1]*M.shape[3]+(M.shape[1]-1)*sep,M.shape[4]),dtype=M.dtype)
  for i in range(M.shape[0]):
    for j in range(M.shape[1]):
      canvas[i*(M.shape[2]+sep):i*(M.shape[2]+sep)+M.shape[2],j*(M.shape[3]+sep):j*(M.shape[3]+sep)+M.shape[3]]=M[i,j]
  return canvas

