import numpy as np
import torch
from unet import unet
import cv2

def one_hot_encoding(mask):
  '''input: H x W output:  (8 x H x W) '''
  mask_onehot = np.zeros((8,256,320))
  #one-hot encoding for mask
  list_mapping = [0,32,64,96,128,160,192,224]
  for i in range(len(list_mapping)):
    label = mask==list_mapping[i]
    label.astype(np.float)
    mask_onehot[i] = label
  return mask_onehot

def reverse_one_hot_encoding(mask):
  '''input: (8 x H x W) output: H x W'''
  list_mapping = [0,32,64,96,128,160,192,224]
  mask = np.argmax(mask, axis=0)
  for i in range(mask.shape[0]):
    for j in range(mask.shape[1]):
      mask[i,j] = list_mapping[mask[i,j].astype(np.int)]
  return mask

def reverse_Normalize(array):  #For colorization
  '''input: (C x H x W) ndarray output: H x W or H x W x C '''
  array = array.transpose((1,2,0))
  array = (array+1)/2
  return array


def inference(input_img, path='/content/gdrive/My Drive/[epoch50]q1_part_e.pth'):
  '''
  input: input_img is a RGB image
         path is where saved checkpint

  output: out_img is a gray scale image
  '''
  kernel = np.ones((5, 5), np.uint8)
  # Initailza Unet for Segmentation
  model =unet()
  if torch.cuda.is_available():  # use gpu if available
    print('using cuda')
    model = model.cuda()
  model = unet()
  model.load_state_dict(torch.load(path))
  with torch.no_grad():
    model.eval()  # Set model to evaluate mode
    try:
      input_img = cv2.resize(input_img, dsize=(320, 256), interpolation=cv2.INTER_CUBIC)
      input_img = input_img.transpose((2, 0, 1))
      input_img = np.expand_dims(input_img,axis=0)
    except:
      print('image format invalid')
    input = torch.from_numpy(input_img).type(torch.FloatTensor)  # change to float torch tensor
    outputs = model(input)
    output = np.squeeze(outputs.cpu().numpy())
    output = reverse_one_hot_encoding(output)
    output = cv2.erode(output.astype('uint8'), kernel)

  return output