image_size = 576
MODELFILE = './model/lowest_loss576-409.pkl'

fontpath = './resource/NotoSansCJKjp-Regular.otf'
fontsize =  14

#Hiragana OCR = 1 or Alphabet OCR = 3
ocr_output =  1

from os import listdir
from skimage import io
from skimage.transform import resize
from tqdm import trange, tqdm
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
import torch
from torchvision.utils import save_image
import pandas as pd
from character_set import CharacterSet
from PIL import Image, ImageDraw, ImageFont
from functions import *

font = ImageFont.truetype(fontpath, fontsize, encoding='utf-8')
dictionary = pd.read_csv("./resource/dictionary.csv")
datapath = "./testdata/"
#booklist = listdir(datapath)
testlist = ["200015843", "200017458", "200021063", "200025191"]
booktp = []
bookfp = []
bookfn = []
bookf1 = []
bookgt = []
bookpd = []
bookac = []

#Get character list

charlist = pd.read_csv('charlist.csv')
charlist = charlist['char'].tolist()

#for book in booklist:
for book in testlist:

    print(book)
    directory = datapath + book + "/images/"
    #GT
    #if book in testlist:
    gt = pd.read_csv("./testdata/" + book + "/" + book + "_coordinate.csv")
    size = pd.read_csv("./testdata/" + book + "/" + "imagesize.csv")

    checkdir(datapath + book + "/resized/")
    resized_directory = datapath + book + "/resized/"
    
    filelist = listdir(directory)
    
    resized_filelist = listdir(datapath + book + "/resized/")
    
    # ---------------------test images resize---------------------------------------
    
    if (len(resized_filelist) != 0):
        im = io.imread(datapath + book + "/resized/" + resized_filelist[0])   
        if (im.shape[0] != image_size):
            for j in trange(len(filelist)):
                imresized = image_resize(directory + filelist[j], image_size)
                io.imsave(resized_directory + filelist[j], imresized)
    else:
       for j in trange(len(filelist)):
           imresized = image_resize(directory + filelist[j], image_size)
           io.imsave(resized_directory + filelist[j], imresized) 
           
    resized_filelist = listdir(datapath + book + "/resized/")
    
    #-------------- Detection ------------------------------------------------------
    checkdir(datapath + book + "/csv/")
    char = CharacterSet()
    model = torch.load(MODELFILE)
    model.cuda()
    
    for file in resized_filelist:
    
        x = []
        y = []
        prob = []
        char2 = []
        result_df = pd.DataFrame()
    
        image = io.imread(resized_directory + file)
     
        image = np.swapaxes(image, 0, 2) 
        image = np.swapaxes(image, 1, 2)
        image = np.reshape(image, (1, 3, image_size, image_size)) 
    
        image = torch.from_numpy(image.astype('float32')).cuda() / 255.0
    
        print(image.min(), image.max())
        
        a,b = model(image)
        a = a.cpu()
        b = b.cpu()
    
        for i in range(image_size):
            for j in range(image_size):
                max_prob, max_ind = a[:,:,i,j].max(dim=1)
                max_ind = max_ind.item()
                max_prob = max_prob.item()
                max_char = char.ind2char[int(max_ind)]
                
                if max_prob > 0.5 and b[:,:,i,j] > 0.5:
                    #print(i, j, max_prob, max_char)
                    x.append(j)
                    y.append(i)
                    prob.append(max_prob)
                    char2.append(max_char)
                
        result_df['y'] = y
        result_df['x'] = x
        result_df['prob'] = prob
        result_df['Unicode'] = char2
        result_df.to_csv(datapath + book + "/csv/" + file[0:-4] + '.csv', index = False)
    
    #--------------------------------Clustering-------------------------------------
    
    image_file = listdir(resized_directory)
    checkdir(datapath + book + "/ocr/")
    checkdir(datapath + book + "/cluster/")

    for j in trange(len(image_file)):
        file_name = image_file[j]    
    
        #Open files we need for result
        result = pd.read_csv(datapath + book + "/csv/" + file_name[0:-4] + ".csv")
        if (result.shape[0] > 0):
        
            clustering = DBSCAN(eps=3, min_samples=1)
            
            labels = clustering.fit_predict(result[['y', 'x']].values)
            
            pd.Series(labels).value_counts()
            
            result['label'] = labels
            
            xs = []
            ys = []
            chars = []
            for l, group in result.groupby('label'):
                x, y = int(group['x'].mean()), int(group['y'].mean())
                char = group['Unicode'].values[0]
        
                xs.append(round(x))
                ys.append(round(y))
                chars.append(char)
            
            cluster_result = pd.DataFrame()
            cluster_result['Unicode'] = chars
            cluster_result['x'] = xs 
            cluster_result['y'] = ys 
            cluster_result.to_csv(datapath + book + "/cluster/" + file_name[0:-4] + '.csv', index = False)
            
            cluster_result['Image'] = file_name[0:-4]
    
    
            #------------------------------ Draw Character on Image ------------------------        
            image = Image.open(datapath + book + "/resized/" + file_name)

            draw = ImageDraw.Draw(image)
            
            for x, y, char in zip(xs, ys, chars):
                character = dictionary[dictionary['unicode'] == char].iloc[0, ocr_output]
                
                color = 'rgb(255, 0, 0)' 
                 
                draw.text((x + 0, y - 0), character, fill=color, font = font)
                draw.rectangle(((x, y), (x + 2, y + 2)), fill="white")
                
            image.save(datapath + book + "/ocr/" + file_name[0:-4] + ".jpg")
    
    #------------------------- F1 Score --------------------------------------------------------  
    cluster_file = listdir(datapath + book + "/cluster/")
    
    tp = 0
    fp = 0
    fn = 0
    fp1 = 0
    totalgt = 0
    totalpd = 0
    
    for page in trange(len(cluster_file)): 
        file = cluster_file[page]
        #print(file)
        
        imagesize = size[size['image'] == (file[0:-4] + ".jpg")]
        rx = imagesize['w'].iloc[0]/image_size
        ry = imagesize['h'].iloc[0]/image_size
        
        gtdf = gt[gt['Image'] == file[0:-4]]
        
        gu = []
        gx = []
        gy = []
        gw = []
        gh = []
        for index, grow in gtdf.iterrows():
            if grow['Unicode'] in charlist:
                gu.append(grow['Unicode'])
                gx.append(grow['X'])
                gy.append(grow['Y'])
                gw.append(grow['Width'])
                gh.append(grow['Height'])
                
        gtdf = pd.DataFrame()
        gtdf['Unicode'] = gu
        gtdf['X'] = gx
        gtdf['Y'] = gy
        gtdf['Width'] = gw
        gtdf['Height'] = gh
        
        
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

f1df = pd.DataFrame()

    
f1df['book'] = testlist

f1df['gt'] = bookgt
f1df['pd'] = bookpd
f1df['tp'] = booktp
f1df['fp'] = bookfp
f1df['fn'] = bookfn
f1df['f1'] = bookf1
f1df['tp/gt'] = bookac

f1df.to_csv("./f1score/" + MODELFILE[8:] + ".csv", index = False)






    


























        
        
