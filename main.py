##### Hyperparameters ##########################################################
img_size = 576
lr = 0.0002
epoch = 100
################################################################################
from UNet import *
import numpy as np

import argparse
import torchvision
import torch
from image_folder import ImageFolderWithPaths as ImageFolder
from kuzu_get_targets import KuzushijiDataLoader
import random

import time
import sys

from tqdm import tqdm
import mlcrate as mlc
from mlcrate.torch import tonp, totensor
import lycon
from math import isnan

parser = argparse.ArgumentParser()

parser.add_argument("--batch_size",type=int,default=2)
parser.add_argument("--num_gpu",type=int,default=2)
args = parser.parse_args()

batch_size = args.batch_size

prev_test = 1

kuzu_targets = KuzushijiDataLoader()

kuzu = './data'

transforms = [torchvision.transforms.Resize((img_size,img_size)), torchvision.transforms.ToTensor()]
transforms_keep = [torchvision.transforms.Resize((32,32)), torchvision.transforms.ToTensor()]
##
kuzu_data = ImageFolder(kuzu, transform=torchvision.transforms.Compose(transforms))
img_batch = torch.utils.data.DataLoader(kuzu_data, batch_size=batch_size, shuffle=True, num_workers=2)

kuzu_keep = ImageFolder(kuzu, transform=torchvision.transforms.Compose(transforms_keep))
img_batch_keep = torch.utils.data.DataLoader(kuzu_keep, batch_size=batch_size, shuffle=True, num_workers=2)

img_sizes = {}
all_image_lst = []

#This loop is for getting file sizes and a list of all image names
print('Getting image sizes...')
try:
    img_sizes, all_image_lst = mlc.load('cache/image_sizes.pkl')
    print('Loaded image sizes from cache')
except FileNotFoundError:
    for _, (images, labels, file_locs) in enumerate(tqdm(img_batch_keep)):
        #img_sizes[file_loc] = image.size()
        #print('file loc', file_loc)
        for file_loc in file_locs:
            size2 = lycon.load(file_loc).shape  
            new_size = torch.Size([1, 3, *size2])
            # print(file_loc, new_size)
            img_sizes[file_loc] = new_size
            all_image_lst.append(file_loc)

    mlc.save([img_sizes, all_image_lst], 'cache/image_sizes.pkl')

#print("all image list", all_image_lst)
print('{} images loaded'.format(len(all_image_lst)))

#This randomly takes 50 images for the "test set".
test_images = random.sample(all_image_lst, 50)

del kuzu_keep
del img_batch_keep

# initiate Generator
#This initializes the generator object

generator = nn.DataParallel(UnetGenerator(3, kuzu_targets.num_characters, 64),
                             device_ids=[i for i in range(args.num_gpu)]).cuda()

# loss function & optimizer

recon_loss_func = nn.BCELoss()

gen_optimizer = torch.optim.Adam(generator.parameters(), lr=lr, betas=(0.9,0.999))

# TRAINING

def overlay(fg, bg, alpha=0.3):
    return fg * alpha + bg * (1-alpha)

import pandas as pd

epoch_csv = []
train_loss_csv = []
test_loss_csv = []

file = open('./{}_mse_loss'.format('unet'), 'w')
for i in range(epoch):
    train_losses = []
    test_losses = []

    bar = tqdm(total=len(img_batch) * args.batch_size)
    #print('!!! Training epoch', i)
    print('')
    print('|￣￣￣￣￣￣￣￣|')
    print('|    TRAINING    |') 
    print('|     epoch      |')
    print('|      ', i ,'       |')  
    print('| ＿＿＿_＿＿＿＿|') 
    print(' (\__/) ||') 
    print(' (•ㅅ•) || ')
    print(' / 　 づ')
    
    for iteration, (images, labels, file_locs) in enumerate(img_batch):
        loss_temp = pd.DataFrame()
        t0 = time.time()

        satel_images = images * 1.0

        img_files = [file_loc.split('/')[4].rstrip('.jpg') for file_loc in file_locs]

        # Set whole batch to test if just one image is in test set
        # TODO: Fix this behaviour
        if file_locs[0] in test_images:
            is_test = True
        else:
            is_test = False

        map_images = []
        prox_trues = []
        image_mask = [] # IDs of images in batch which have targets. We need to remove any images without targets so len(x)==len(y)

        # Loop over images in batch and run loadbbox_from_csv
        for i, (img_file, file_loc, image) in enumerate(zip(img_files, file_locs, images)):
            size = [1, *image.size()]
            map_image, prox_true = kuzu_targets.loadbbox_from_csv(img_file, img_size=size, full_size=img_sizes[file_loc])
            if map_image is not None:
                map_images.append(map_image)
                prox_trues.append(prox_true)
                image_mask.append(i)

        # Skip batch if empty
        if len(image_mask) == 0:
            continue

        # Concatenate to batch
        map_image = torch.cat(map_images)
        prox_true = torch.cat(prox_trues)

        t0 = time.time()
        gen_optimizer.zero_grad()

        x = Variable(satel_images[image_mask]).cuda(0)
        y_ = Variable(map_image).cuda(0)
        prox_true = Variable(prox_true).cuda(0)
        y, prox_pred = generator.forward(x)

        loss = recon_loss_func(y, y_) + 0.1 * recon_loss_func(prox_pred, prox_true)
        file.write(str(loss) + "\n")

        loss.backward()
        if not is_test:
            gen_optimizer.step()

        if is_test:
            test_losses.append(loss.item())
        else:
            train_losses.append(loss.item())

        bar.update(batch_size)
        bar.set_description('train_loss {:.5f} test_loss {:.5f}'.format(np.mean(train_losses), np.mean(test_losses)))
            
        #print(loss_temp)
        
        if iteration % 100 == 0:
    
            if isnan(np.mean(train_losses)):
                train_loss_csv.append(0)
            else:
                train_loss_csv.append(np.mean(train_losses))
    
            if isnan(np.mean(test_losses)):
                test_loss_csv.append(0)
            else:
                test_loss_csv.append(np.mean(test_losses))
            
            epoch_csv.append(i)
    
            loss_temp["epoch"] = epoch_csv
            loss_temp["train_loss"] = train_loss_csv
            loss_temp["test_loss"] = test_loss_csv
            loss_temp.to_csv('loss_value.csv', index = False)
            
            x = x.cpu().data
            ysum = y.cpu().data.max(dim=1,keepdim=True)[0].repeat(1,3,1,1)
            ysum_ = y_.cpu().data.max(dim=1,keepdim=True)[0].repeat(1,3,1,1)
            y = y.cpu().data[:,0:3,:,:]
            y_ = y_.cpu().data[:,0:3,:,:]

            prox_true = prox_true.cpu().data.repeat(1,3,1,1)
            prox_pred = prox_pred.cpu().data.repeat(1,3,1,1)

            overlay_real = overlay(y_, x)
            overlay_gen = overlay(y, x)
            overlay_sum = overlay(ysum, x)
            overlay_sumreal = overlay(ysum_, x)
            overlay_true_prox = overlay(prox_true,x,alpha=0.5)
            overlay_gen_prox = overlay(prox_pred,x,alpha=0.5)

            v_utils.save_image(overlay_real,"./result/image_{}_{}_real.png".format(i,iteration))
            v_utils.save_image(overlay_gen,"./result/image_{}_{}_{}_gen.png".format(i,iteration,is_test))
            v_utils.save_image(overlay_sum,"./result/image_{}_{}_sumgen.png".format(i,iteration))
            v_utils.save_image(overlay_sumreal,"./result/image_{}_{}_sumreal.png".format(i,iteration))
            v_utils.save_image(overlay_true_prox, './result/image_{}_{}_proxtrue.png'.format(i,iteration))
            v_utils.save_image(overlay_gen_prox, './result/image_{}_{}_proxgen.png'.format(i,iteration))

            if np.mean(test_losses) < prev_test:

                torch.save(generator,'./model/lowest_loss.pkl')
                prev_test = np.mean(test_losses)
            elif(np.mean(test_losses) > 0.07):
                sys.exit('!!!!!Exploded!!!!!')
        

