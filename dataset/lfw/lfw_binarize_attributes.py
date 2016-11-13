#!/usr/bin/env python2

import json

def read_lfw_attributes(ipath='dataset/lfw/lfw_attributes.txt'):
  # We verify that the first two attributes are person and sequence number
  # (from which the filename can be constructed).
  with open(ipath) as f:
    header=f.readline()
    attributes=f.readline().strip().split('\t')[1:]
    assert attributes[0]=='person'
    assert attributes[1]=='imagenum'
    return header,attributes,[x.strip().split('\t') for x in f.readlines()]

if __name__=='__main__':
  _,lfwattrname,lfwattr=read_lfw_attributes(ipath='lfw_attributes.txt')

  # collect, rank by decision value, take top 2/3 most confident
  S={lfwattrname[i]:sorted([j for j,x in enumerate(lfwattr) if float(x[i])>0],key=lambda k: -float(lfwattr[k][i])) for i in range(2,len(lfwattrname))}
  D={lfwattrname[i]:float(lfwattr[S[lfwattrname[i]][len(S[lfwattrname[i]])*2//3]][i]) for i in range(2,len(lfwattrname))}
  def augment_neg(attr,negattr):
    i=lfwattrname.index(attr)
    S[negattr]=sorted([j for j,x in enumerate(lfwattr) if float(x[i])<0],key=lambda k: float(lfwattr[k][i]))
    D[negattr]=float(lfwattr[S[negattr][len(S[negattr])*2//3]][i])
  augment_neg('Male','Female')
  S={k:v[:len(v)*2//3] for k,v in S.iteritems()}
  # S is a map from attrname to list of indices
  # D is a map from attrname to decision value cutoff

  M=set(S['Male'])
  F=set(S['Female'])
  G={k:'Male' if sum(1 if j in M else (-1 if j in F else 0) for j in v)>=0 else 'Female' for k,v in S.iteritems()}
  T={k:[j for j in v if j in (M if G[k]=='Male' else F)] for k,v in S.iteritems()}
  for a in sorted(S.keys()):
    print a,D[a],len(S[a]),G[a],len(T[a])
  print 'Male >=200',sorted([a for a in lfwattrname[2:] if G[a]=='Male' and len(T[a])>=200])
  print 'Female >=200',sorted([a for a in lfwattrname[2:] if G[a]=='Female' and len(T[a])>=200])
  with open('lfw_binary_attributes.json','w') as f: json.dump({'attribute_members':S,'attribute_gender':G,'attribute_gender_members':T,'male_attributes':sorted([a for a in lfwattrname[2:] if G[a]=='Male']),'female_attributes':sorted([a for a in lfwattrname[2:] if G[a]=='Female'])},f,indent=2)
