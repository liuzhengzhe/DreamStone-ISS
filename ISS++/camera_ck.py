import glob
import numpy as np
paths=glob.glob('cameras/*')
dic={}

for path in paths:
  c=np.load(path)
  print (path, c)
  num=c[0][0]
  dic[num]=0
print (len(dic.keys()))