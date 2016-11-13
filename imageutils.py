
import numpy
import PIL.Image
import PIL.ImageDraw
import PIL.ImageFont
#from matplotlib.pyplot import figure,imshow

#def plotimage(I,mark=None,mark_size=12):
#  figure()
#  if mark is None:
#    imshow(I)
#  else:
#    J=numpy.copy(I)
#    for (i,j) in mark:
#      J[max(0,i-mark_size):i+mark_size+1,j]=1
#      J[i,max(0,j-mark_size):j+mark_size+1]=1
#      J[max(0,i+1-mark_size):i+1+mark_size+1,j+1]=0
#      J[i+1,max(0,j+1-mark_size):j+1+mark_size+1]=0
#    imshow(J)

def montage(M,sep=0,canvas_value=0):
  # row X col X H X W X C
  assert M.ndim==5
  canvas=numpy.ones((M.shape[0]*M.shape[2]+(M.shape[0]-1)*sep,M.shape[1]*M.shape[3]+(M.shape[1]-1)*sep,M.shape[4]),dtype=M.dtype)*canvas_value
  for i in range(M.shape[0]):
    for j in range(M.shape[1]):
      canvas[i*(M.shape[2]+sep):i*(M.shape[2]+sep)+M.shape[2],j*(M.shape[3]+sep):j*(M.shape[3]+sep)+M.shape[3]]=M[i,j]
  return canvas

def render_text(s,size,font=PIL.ImageFont.truetype('/usr/share/fonts/truetype/droid/DroidSerif-Regular.ttf',80)):
  '''
  Returns rendered text as a H x W x 3 numpy array in the range [0,1]. The
  image will be the exact size given. If one dimension is None then the
  image will be sized to fit the given dimension.
  '''
  image=PIL.Image.new('RGB',(5,5),(255,255,255))
  draw=PIL.ImageDraw.Draw(image)
  image=PIL.Image.new('RGB',draw.textsize(s,font=font),(255,255,255))
  draw=PIL.ImageDraw.Draw(image)
  draw.text((0,0),s,(0,0,0),font=font)
  if size[0]==None:
    size=list(size)
    size[0]=int(round(image.size[1]*size[1]/float(image.size[0])))
  elif size[1]==None:
    size=list(size)
    size[1]=int(round(image.size[0]*size[0]/float(image.size[1])))
  image=image.resize((size[1],size[0]),PIL.Image.LANCZOS)
  I=numpy.array(image)/255.0
  return I

def concatenate(X,axis,canvas_value=0,gravity=(-1)):
  '''
  Given a sequence of images, concatenate them along the given axis,
  expanding the other axes as needed. If gravity is zero then the original
  data will be centered in the output domain. Negative or positive gravity
  will cause it to be flush with the lower or upper bound, respectively.
  '''
  outshape=[sum(x.shape[i] for x in X) if i==axis else max(x.shape[i] for x in X) for i in range(X[0].ndim)]
  Y=[]
  for x in X:
    newshape=list(outshape)
    newshape[axis]=x.shape[axis]
    if gravity>0:
      Y.append(numpy.pad(x,[(newshape[i]-x.shape[i],0) for i in range(x.ndim)],'constant',constant_values=canvas_value))
    elif gravity==0:
      Y.append(numpy.pad(x,[((newshape[i]-x.shape[i])//2,(newshape[i]-x.shape[i])-(newshape[i]-x.shape[i])//2) for i in range(x.ndim)],'constant',constant_values=canvas_value))
    else:
      Y.append(numpy.pad(x,[(0,newshape[i]-x.shape[i]) for i in range(x.ndim)],'constant',constant_values=canvas_value))
  return numpy.concatenate(Y,axis=axis)

