import torch
import torch
model=torch.load('../stage2/out/a chair/model20.pt')
dic=torch.load('../stage2/out/a chair/model20.pt')


for k in model['model'].keys():
  if 'decoder' in k:
    del dic['model'][k]


for k in model['model'].keys():
  data=model['model'][k]  
  if 'decoder.fc_out.weight' in k:
    dic['model'][k.replace('decoder','decoder_shape')]=data[0:1,:]
    dic['model'][k.replace('decoder','decoder_color')]=data[1:,:]
  elif 'decoder.fc_out.bias' in k:
    dic['model'][k.replace('decoder','decoder_shape')]=data[0:1]
    dic['model'][k.replace('decoder','decoder_color')]=data[1:]

  elif 'decoder' in k:

    print (k, data.shape)
    dic['model'][k.replace('decoder','decoder_shape')]=data
    dic['model'][k.replace('decoder','decoder_color')]=data
    
  
  else:
    dic['model'][k]=data


torch.save(dic,'model.pt')
