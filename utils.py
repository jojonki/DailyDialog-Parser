import os
import torch
from torch.autograd import Variable
import torch.nn.functional as F
from torch import LongTensor as LT
from torch import FloatTensor as FT

STOKEN = '__eou__'


def load_data(dpath, mode):
    assert mode == 'train' or mode == 'test' or mode == 'validation'
    dial_f = os.path.join(dpath, '{}/dialogues_{}.txt'.format(mode, mode))
    act_f = os.path.join(dpath, '{}/dialogues_act_{}.txt'.format(mode, mode))
    emo_f = os.path.join(dpath, '{}/dialogues_emotion_{}.txt'.format(mode, mode))
    dlg_data, act_data, emo_data = [], [], []
    with open(dial_f, 'r') as f:
        lines = f.readlines()
        for l in lines:
            turns = [t.strip().split(' ') for t in l.split(STOKEN)]
            if turns[-1] == ['']:
                turns = turns[:-1]
            dlg_data.append(turns)
    with open(act_f, 'r') as f:
        lines = f.readlines()
        for l in lines:
            acts = [int(d) - 1 for d in l.strip().split(' ')] # -1 for range 0 - 3
            act_data.append(acts)
    with open(emo_f, 'r') as f:
        lines = f.readlines()
        for l in lines:
            emos = [int(d) for d in l.strip().split(' ')]
            emo_data.append(emos)
    return dlg_data, act_data, emo_data


def to_var(x, var_type=None):
    if var_type is not None:
        x = var_type(x)
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x)


def pad(vec, sent_len, pad_token=0):
    pad_len = max(0, sent_len - len(vec))
    vec += [pad_token] * pad_len
    vec = vec[:sent_len]
    return vec


def vectorize(w2i, d, a, e):
    ret_data = []
    sent_max_len = max([len(s) for s in d])
    hist_max_len = len(d)
    for i, u in enumerate(d):
        u_v = pad([w2i[w] for w in u], sent_max_len)
        x_v = [u_v]
        for h in d[:i][::-1]: # reverse order
            x_v.append(pad([w2i[w] for w in h], sent_max_len))
        while len(x_v) < hist_max_len: # history padding
            x_v.append(pad([], sent_max_len))
        x_v = to_var(x_v, LT)
#         x_v = torch.stack(x_v, 0)
        ret_data.append((x_v, to_var([a[i]], LT), to_var([e[i]], FT)))
    return ret_data


def save_checkpoint(state, filename='./checkpoints/checkpoint.pth.tar'):
    print('save model!', filename)
    torch.save(state, filename)
