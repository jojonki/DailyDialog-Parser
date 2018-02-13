import os
import random
import torch
import torch.nn.functional as F
import argparse
import numpy as np
from tqdm import tqdm

from utils import load_data, vectorize, save_checkpoint
from models import SimpleLSTM

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=8, help='each dialog formed one minibatch')
parser.add_argument('--start_epoch', type=int, default=0, help='initial epoch count for training')
parser.add_argument('--n_epochs', type=int, default=30, help='the number of epochs')
parser.add_argument('--resume', type=str, metavar='PATH', help='path saved params')
parser.add_argument('--task', type=int, default=1, help='default 1 for task number. 1-6')
parser.add_argument('--seed', type=int, default=1111, help='random seed')
args = parser.parse_args()

for arg in vars(args):
    print('Arg:', arg, '=', getattr(args, arg))


# Set the random seed manually for reproducibility.
torch.manual_seed(args.seed)
np.random.seed(args.seed)


def train(model, data, act_data, emo_data, start_epoch=0, n_epochs=20):
    print('Train---------------------')
    model.train()
    loss_fn = F.nll_loss
    for epoch in range(start_epoch, n_epochs):
        print('Epoch', epoch)
        indices = [i for i in range(len(data))]
        random.shuffle(indices)
        total, acc = 0, 0
        # for i, (d, a, e) in tqdm(enumerate(zip(data, act_data, emo_data)), total=len(data)):
        for i, idx in tqdm(enumerate(indices), total=len(indices)):
            d, a, e = data[idx], act_data[idx], emo_data[idx]
            batch = vectorize(w2i, d, a, e)
            x = torch.stack([turn[0] for turn in batch], 0)
            act_labels = torch.stack([turn[1] for turn in batch], 0).squeeze(1)
            preds = model(x)
            acc += torch.sum(act_labels == torch.max(preds, 1)[1]).data[0]
            total += x.size(0)
            loss = loss_fn(preds, act_labels)
            # print(loss.data[0])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if i % 500 == 0:
                print('Epoch {}, Acc {:.2f}%, loss {:.3f}'.format(epoch, 100*acc/total, loss.data[0]))

        filename = 'ckpts/SimpleLSTM-Epoch-{}.model'.format(epoch)
        save_checkpoint({
            'epoch'        : epoch,
            'state_dict'   : model.state_dict(),
            'optimizer'    : optimizer.state_dict(),
        }, filename=filename)


def test(model, data, act_data, emo_data):
    print('Test---------------------')
    model.eval()
    total, acc = 0, 0
    for i, (d, a, e) in tqdm(enumerate(zip(data, act_data, emo_data)), total=len(data)):
        batch = vectorize(w2i, d, a, e)
        x = torch.stack([turn[0] for turn in batch], 0)
        act_labels = torch.stack([turn[1] for turn in batch], 0).squeeze(1)
        preds = model(x)
        acc += torch.sum(act_labels == torch.max(preds, 1)[1]).data[0]
        total += x.size(0)

        if i % 500 == 0:
            print('Acc {:.2f}%'.format(100*acc/total))

    print('Final Acc {:.2f}%'.format(100*acc/total))


dpath = 'ijcnlp_dailydialog/'
train_data, train_act_data, train_emo_data = load_data(dpath, 'train')
test_data, test_act_data, test_emo_data  = load_data(dpath, 'test')
val_data, val_act_data, val_emo_data  = load_data(dpath, 'validation')
data = train_data + test_data + val_data

vocab = ['_PAD_'] + sorted(set(w for d in data for s in d for w in s))
print('vocab size:', len(vocab))
w2i = {w: i for i, w in enumerate(vocab)}
i2w = {i: w for w, i in enumerate(w2i)}

embd_size = 128
hidden_size = 128
class_size = 4
model = SimpleLSTM(len(w2i), embd_size, hidden_size, class_size)
optimizer = torch.optim.Adadelta(filter(lambda p: p.requires_grad, model.parameters()))
if torch.cuda.is_available():
    model.cuda()

if args.resume is not None and os.path.isfile(args.resume):
    print("=> loading checkpoint '{}'".format(args.resume))
    ckpt = torch.load(args.resume)
    args.start_epoch = ckpt['epoch'] + 1 if 'epoch' in ckpt else args.start_epoch
    model.load_state_dict(ckpt['state_dict'])
    optimizer.load_state_dict(ckpt['optimizer'])
else:
    print("=> no checkpoint found")
train(model, train_data, train_act_data, train_emo_data, args.start_epoch, args.n_epochs)
test(model, test_data, test_act_data, test_emo_data)
