image_size = 576
MODELFILE = './model/epoch43.pkl'
from os import listdir
from skimage import io
from skimage.transform import resize
from tqdm import trange, tqdm
import numpy as np
import matplotlib.pyplot as plt

import pandas as pd
from character_set import CharacterSet
from PIL import Image, ImageDraw, ImageFont
from functions import *

dictionary = pd.read_csv("./resource/dictionary.csv")
datapath = "./testdata/"
f1df = pd.DataFrame()

booktp = []
bookfp = []
bookfn = []
bookf1 = []
bookgt = []
bookpd = []
bookac = []

booklist = listdir(datapath)
testlist = ["200015843", "200017458", "200021063", "200025191"]

for book in booklist:
    
    if book in testlist:
        gt = pd.read_csv("./testdata/" + book + "/" + book + "_coordinate.csv")
        size = pd.read_csv("./testdata/" + book + "/" + "imagesize.csv")
    else:
        gt = pd.read_csv("./traindata/" + book + "/" + book + "_coordinate.csv")
        size = pd.read_csv("./traindata/" + book + "/" + "imagesize.csv")
    
    cluster_file = listdir(datapath + book + "/cluster/")
    
    tp = 0
    fp = 0
    fn = 0
    fp1 = 0
    totalgt = 0
    totalpd = 0
    
    for page in range(len(cluster_file)):    
        file = cluster_file[page]
        print(file)
        
        imagesize = size[size['image'] == (file[0:-4] + ".jpg")]
        rx = imagesize['w'].iloc[0]/image_size
        ry = imagesize['h'].iloc[0]/image_size
        
        gtdf = gt[gt['Image'] == file[0:-4]]
        if gtdf.shape[0] != 0:
            totalgt += gtdf.shape[0]
            x = round(gtdf['X']/rx)
            xs = round((gtdf['X'] + gtdf['Width'])/rx)
            y = round(gtdf['Y']/ry)
            ys = round((gtdf['Y'] + gtdf['Height'])/ry)
            gtunicode = gtdf['Unicode']
            gtlist = [1] * len(x)
            
            
            pddf = pd.read_csv(datapath + book + "/cluster/" + file)
            totalpd += pddf.shape[0]
            boxsize = 5
            px = pddf['x']
            pxs = pddf['x'] + boxsize
            py = pddf['y']
            pys = pddf['y'] + boxsize
            pdunicode = pddf['Unicode']
            pdlist = [1] * len(px)
            
            for g in range(len(x)):
                gtbox = [x.iloc[g], xs.iloc[g], y.iloc[g], ys.iloc[g]]
                for p in range(len(px)):
                    pdbox = [px.iloc[p], pxs.iloc[p], py.iloc[p], pys.iloc[p]]
                    overlap = getiou(gtbox, pdbox)
                    if (overlap > 0):
                        if(gtlist[g] == 1):
                            gtlist[g] == 0
                            if pdunicode[p] == gtunicode.iloc[g]:
                                tp += 1
                            else:
                                fp1 += 1
                    
            for p in range(len(px)):
                pdbox = [px.iloc[p], pxs.iloc[p], py.iloc[p], pys.iloc[p]]
                for g in range(len(x)):
                    overlap = getiou(gtbox, pdbox)
                    if (overlap > 0):
                        if pdlist[p] == 1:
                            pdlist[p] == 0
    
    fn = sum(gtlist)
    fp = fp1 + sum(pdlist)
    f1 = f1score(tp, fp, fn)
    acc = round(tp/totalgt, 4)
    
    print("gt:" , totalgt , "pd:", totalpd, "tp:", tp, "fp:", fp, "fn:", fn, "f1:", f1, "acc:" , acc)
    
    bookgt.append(totalgt)
    bookpd.append(totalpd)
    booktp.append(tp)
    bookfp.append(fp)
    bookfn.append(fn)
    bookf1.append(f1score(tp, fp, fn))
    bookac.append(acc)

booklist.append("Sum")
sumgt = sum(bookgt)
sumpd = sum(bookpd)
sumtp = sum(bookpd)
sumfp = sum(bookpd)
sumfn = sum(bookpd)
sumf1 = f1score(sumtp, sumfp, sumfn)
sumac = sum(bookac)/(len(booklist) + len(testlist))

bookgt.append(sumgt)
bookpd.append(sumpd)
booktp.append(sumtp)
bookfp.append(sumfp)
bookfn.append(sumfn)
bookf1.append(sumf1)
bookac.append(sumac)

f1df['book'] = booklist

f1df['gt'] = bookgt
f1df['pd'] = bookpd
f1df['tp'] = booktp
f1df['fp'] = bookfp
f1df['fn'] = bookfn
f1df['f1'] = bookf1
f1df['tp/gt'] = bookac

f1df.to_csv("./f1score/" + MODELFILE[8:] + ".csv", index = False)

































