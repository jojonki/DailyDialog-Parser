import random
import torch
import torch.nn.functional as F

from tqdm import tqdm
from utils import load_data, vectorize
from models import SimpleLSTM

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

def train(model, data, act_data, emo_data, n_epochs=20):
    model.train()
    loss_fn = F.nll_loss
    for epoch in range(n_epochs):
        print('Epoch', epoch)
        indices = random.shuffle([i for i in range(len(data))])
        total, acc = 0, 0
        # for i, (d, a, e) in tqdm(enumerate(zip(data, act_data, emo_data)), total=len(data)):
        for i, idx in tqdm(enumerate(indices)):
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


def test(model, data, act_data, emo_data):
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
            print('Acc {:.2f}%, loss {:.3f}'.format(100*acc/total))

    print('Final Acc {:.2f}%, loss {:.3f}'.format(100*acc/total))


train(model, train_data, train_act_data, train_emo_data)
