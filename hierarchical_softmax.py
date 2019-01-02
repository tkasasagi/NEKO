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

if __name__ == "__main__":

    N = 500
    h_size = 32
    k1 = 20
    k2 = 200

    h = torch.randn(N, h_size)
    y = torch.randint(low=0,high=k1*k2, size=(N,))

    w1 = torch.randn(h_size, k1)
    b1 = torch.randn(k1)
    w2 = torch.randn(k1, h_size, k2)
    b2 = torch.randn(k1, k2)

    y1 = (y//k2).long()
    y2 = (y%k2).long()

    p_y1 = torch.mm(h, w1) + b1

    p_y1_sm = sm(p_y1)

    loss = ce(p_y1, y1)

    for c in range(k1):
        w2_pick = w2[c]
        b2_pick = b2[c]
        ind = (y1 == c)

        p_y2_y1 = torch.mm(h[ind], w2_pick) + b2_pick

        p_y2_y1_sm = sm(p_y2_y1)

        y2_pick = y2[ind]

        loss += ce(p_y2_y1, y2_pick)

    '''Logic for picking most likely class'''

    max_prob, max_ind = torch.max(p_y1_sm,dim=1)

    top_p = torch.zeros(N,)
    top_ind = torch.zeros(N,).long()

    for c in range(k1):
        w2_pick = w2[c]
        b2_pick = b2[c]
        ind = (max_ind == c)

        p_y2_y1 = torch.mm(h[ind], w2_pick) + b2_pick
        p_y2_y1_sm = sm(p_y2_y1)

        max_prob_2, max_ind_2 = torch.max(p_y2_y1_sm,dim=1)

        top_p[ind] = max_prob[ind] * max_prob_2
        top_ind[ind] = (c*k2 + max_ind_2)
        print top_ind.max()

    print top_p.shape, top_ind.shape, loss.shape
