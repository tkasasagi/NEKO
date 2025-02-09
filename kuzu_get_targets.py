location = './data/'
from os import listdir
import pandas as pd
import torch

import math
from character_set import CharacterSet

class KuzushijiDataLoader:

  def __init__(self):
   
    chars = CharacterSet()
    num_char = chars.num_characters
    
    self.num_characters = num_char
    self.num_char = num_char
    self.chars = chars
    lines = []
    
    #Get Booklist from booklist.csv
    df = pd.read_csv('booklist.csv')
    booklist = list(df['booklist'])
    
    for book in booklist:      
      csvfile = location + book + '/' + book + "_coordinate.csv"     
      fh  = open(csvfile)
      header = fh.readline()
      for line in fh:
        lines.append(line.rstrip('\n'))
        
    obj = {}
    
    for line in lines:
      obj[line.split(',')[1]] = []
    
    
    for line in lines[1:]:
      linesp = line.split(',')
      char = linesp[0]
      if char in chars.char2ind:
        obj[linesp[1]].append([int(linesp[2]),int(linesp[3]),int(linesp[6]),int(linesp[7]),chars.char2ind[char]])
      else:
        obj[linesp[1]].append([int(linesp[2]),int(linesp[3]),int(linesp[6]),int(linesp[7]),-1])
    
    self.obj = obj

  def loadbbox_from_csv(self, target_img, img_size, full_size):

    target_tensor = torch.zeros(size=(img_size[0],self.num_char,img_size[2],img_size[3]))
    prox = torch.zeros(size=(img_size[0],1,img_size[2],img_size[3]))

    if target_img not in self.obj:
      #print("targets not found", target_img)
      #return target_tensor, prox
      return None, None

    bb_lst = self.obj[target_img]

    xresize = full_size[3]/img_size[3]
    yresize = full_size[2]/img_size[2]

    for bb in bb_lst:
      xpos, ypos, width, height, char_ind = bb
      xpos = int(xpos/xresize)
      width = int(width/xresize)
      ypos = int(ypos/yresize)
      height = int(height/yresize)

      xcenter = xpos + width//2
      ycenter = ypos + height//2

      extra_width = max(width, 10) - width

      if char_ind != -1:
        target_tensor[:,char_ind,ypos:ypos+height,xpos-extra_width//2:xpos+width+extra_width//2] += 1

      prox[:,0,ycenter-2:ycenter+2,xcenter-2:xcenter+2] += 1.0

    prox = torch.clamp(prox, 0.0, 1.0)

    return target_tensor, prox

