
import numpy

import SSIM_Index

def measure_PSNR(A,B,max_value):
  # input is dim1, dim2, ... dimN, channel, height, width
  outshape=A.shape[:-3]
  A=A.reshape(-1,*(A.shape[-3:]))
  B=B.reshape(-1,*(B.shape[-3:]))
  dim1,c,h,w=A.shape
  assert B.shape==A.shape
  mse=numpy.zeros(dim1)
  for i in range(dim1):
    mse[i]=((A[i]-B[i])**2).sum()/c/h/w
  psnr=(20*numpy.log10(max_value)-10*numpy.log10(mse))
  if len(outshape)==0:
    return psnr
  return psnr.reshape(*outshape)

def measure_SSIM(A,B,max_value):
  # input is dim1, dim2, ... dimN, height, width
  # output is dim1, dim2, ... dimN
  assert A.ndim>=2
  assert A.shape==B.shape
  if A.ndim==2:
    return SSIM_Index.compute_ssim(A,B,max_value)
  else:
    outshape=A.shape[:-2]
    A=A.reshape(-1,*(A.shape[-2:]))
    B=B.reshape(-1,*(B.shape[-2:]))
    dim1,h,w=A.shape
    result=numpy.zeros((dim1,))
    for i in range(dim1):
      result[i]=SSIM_Index.compute_ssim(A[i],B[i],max_value)
    return result.reshape(*outshape)
