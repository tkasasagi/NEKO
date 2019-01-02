'''
Implementation of hierarchical softmax.  

Input: last hidden state h: (N, h_dim), 
    target: (N, k1*k2)
Output1: Loss
Output2: top character, (N,k1*k2).  top character's prob, (N,k1*k2).  
Output3: predict for a specific group
'''

import torch
import torch.nn as nn

ce = nn.CrossEntropyLoss()
sm = nn.Softmax(dim=1)
bce = nn.BCELoss()

'''
Simple method for training with very large output size vocabularies.  


'''

from Basic_blocks import * 

class HierarchicalSoftmaxUnetGenerator(nn.Module):

    def __init__(self,in_dim,num_filter, num_characters, num_character_groups, norm_type='gn'):
        super(HierarchicalSoftmaxUnetGenerator,self).__init__()
        self.in_dim = in_dim
        self.num_filter = num_filter
        act_fn = nn.LeakyReLU(0.2, inplace=True)

        self.k1 = num_character_groups
        self.k2 = num_characters // num_character_groups

        self.w1 = nn.Parameter(torch.randn(self.num_filter, self.k1))
        self.b1 = nn.Parameter(torch.randn(self.k1))
        self.w2 = nn.Parameter(torch.randn(self.k1, self.num_filter, self.k2))
        self.b2 = nn.Parameter(torch.randn(self.k1, self.k2))

        if norm_type == "bn":
            norm_func = lambda inp: nn.BatchNorm2d(inp)
        elif norm_type == "gn":
            norm_func = lambda inp: nn.GroupNorm(inp, inp)

        print("\n------Initiating U-Net------\n")

        self.down_1 = conv_block_2(self.in_dim,self.num_filter,act_fn,norm_func)
        self.pool_1 = maxpool()
        self.down_2 = conv_block_2(self.num_filter*1,self.num_filter*2,act_fn,norm_func)
        self.pool_2 = maxpool()
        self.down_3 = conv_block_2(self.num_filter*2,self.num_filter*4,act_fn,norm_func)
        self.pool_3 = maxpool()
        self.down_4 = conv_block_2(self.num_filter*4,self.num_filter*8,act_fn,norm_func)
        self.pool_4 = maxpool()

        self.bridge = conv_block_2(self.num_filter*8,self.num_filter*16,act_fn,norm_func)

        self.trans_1 = conv_trans_block(self.num_filter*16,self.num_filter*8,act_fn,norm_func)
        self.up_1 = conv_block_2(self.num_filter*16,self.num_filter*8,act_fn,norm_func)
        self.trans_2 = conv_trans_block(self.num_filter*8,self.num_filter*4,act_fn,norm_func)
        self.up_2 = conv_block_2(self.num_filter*8,self.num_filter*4,act_fn,norm_func)
        self.trans_3 = conv_trans_block(self.num_filter*4,self.num_filter*2,act_fn,norm_func)
        self.up_3 = conv_block_2(self.num_filter*4,self.num_filter*2,act_fn,norm_func)
        self.trans_4 = conv_trans_block(self.num_filter*2,self.num_filter*1,act_fn,norm_func)
        self.up_4 = conv_block_2(self.num_filter*2,self.num_filter*1,act_fn,norm_func)

        self.out_prox = nn.Sequential(
            nn.Conv2d(self.num_filter, self.num_filter,3,1,1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(self.num_filter, 1, 3, 1, 1),
            nn.Sigmoid()
        )

    def compute_h(self,input):
        down_1 = self.down_1(input)
        pool_1 = self.pool_1(down_1)
        down_2 = self.down_2(pool_1)
        pool_2 = self.pool_2(down_2)
        down_3 = self.down_3(pool_2)
        pool_3 = self.pool_3(down_3)
        down_4 = self.down_4(pool_3)
        pool_4 = self.pool_4(down_4)
        bridge = self.bridge(pool_4)
        trans_1 = self.trans_1(bridge)
        concat_1 = torch.cat([trans_1,down_4],dim=1)
        up_1 = self.up_1(concat_1)
        trans_2 = self.trans_2(up_1)
        concat_2 = torch.cat([trans_2,down_3],dim=1)
        up_2 = self.up_2(concat_2)
        trans_3 = self.trans_3(up_2)
        concat_3 = torch.cat([trans_3,down_2],dim=1)
        up_3 = self.up_3(concat_3)
        trans_4 = self.trans_4(up_3)
        concat_4 = torch.cat([trans_4,down_1],dim=1)
        up_4 = self.up_4(concat_4)
    
        return up_4

    '''Given an input x and a target, returns the loss scalar'''
    def loss(self, x, char_target, prox_target): 
        h = self.compute_h(x)
        out_prox = self.out_prox(h)

        prox_loss = bce(out_prox, prox_target)

        N = char_target.shape[0] * char_target.shape[2] * char_target.shape[3]
        y = char_target.reshape(N,)
        y1 = (y//self.k2).long()
        y2 = (y%self.k2).long()

        h = h.permute(0,2,3,1).reshape((N,self.num_filter))

        p_y1 = torch.mm(h, self.w1) + self.b1

        p_y1_sm = sm(p_y1)

        char_loss = ce(p_y1, y1)

        for c in range(self.k1):
            w2_pick = self.w2[c]
            b2_pick = self.b2[c]
            ind = (y1 == c)
            p_y2_y1 = torch.mm(h[ind], w2_pick) + b2_pick
            p_y2_y1_sm = sm(p_y2_y1)
            y2_pick = y2[ind]
            char_loss += ce(p_y2_y1, y2_pick)

        return prox_loss + char_loss

    '''For each position, returns top character indices, second output is the probability on those characters'''
    def predict(self, x):
        h = self.compute_h(x)

        prox_output = self.out_prox(h)

        N = h.shape[0] * h.shape[2] * h.shape[3]
        B = h.shape[0]
        height = h.shape[2]
        width = h.shape[3]
        h = h.permute(0,2,3,1).reshape((N,self.num_filter))

        p_y1 = torch.mm(h, self.w1) + self.b1
        p_y1_sm = sm(p_y1)
        max_prob, max_ind = torch.max(p_y1_sm,dim=1)

        top_p = torch.zeros(N,)
        top_ind = torch.zeros(N,).long()

        for c in range(self.k1):
            w2_pick = self.w2[c]
            b2_pick = self.b2[c]
            ind = (max_ind == c)

            p_y2_y1 = torch.mm(h[ind], w2_pick) + b2_pick
            p_y2_y1_sm = sm(p_y2_y1)

            max_prob_2, max_ind_2 = torch.max(p_y2_y1_sm,dim=1)

            top_p[ind] = max_prob[ind] * max_prob_2
            
            print max_prob_2
            
            top_ind[ind] = (c*self.k2 + max_ind_2)

        top_p = top_p.reshape((B,1,height,width))
        top_ind = top_ind.reshape((B,1,height,width))

        return top_p, top_ind, prox_output

if __name__ == "__main__":

    model = HierarchicalSoftmaxUnetGenerator(3, 64, num_characters=2, num_character_groups=1)

    hw = 32

    x = torch.randn(1,3,hw,hw)
    prox_target = torch.randint(0,1, size=(1,1,hw,hw))
    char_target = torch.randint(0,2, size=(1,1,hw,hw))

    l = model.loss(x,char_target,prox_target)
    l.backward()

    top_p, top_ind, prox_output = model.predict(x)

    print top_p.min(), top_p.max()
    print top_ind.min(), top_ind.max()

