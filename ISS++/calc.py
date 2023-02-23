import numpy as np
f=open('user_study')
ours=np.zeros(10)
bases=np.zeros(10)
our_scores=np.zeros(10)
base_scores=np.zeros(10)

cnt=0
for line in f:
  line=line.strip()
  a=line.split(' ')[0]
  b=line.split(' ')[1]
  our_score=line.split(' ')[2]
  base_score=line.split(' ')[3]
  ours[cnt]=int(a)
  bases[cnt]=int(b)
  our_scores[cnt]=float(our_score)/52*100
  base_scores[cnt]=float(base_score)/52*100
  cnt+=1

print (np.mean(ours), np.std(ours))
print (np.mean(bases), np.std(bases))

print (np.mean(our_scores), np.std(our_scores))
print (np.mean(base_scores), np.std(base_scores))