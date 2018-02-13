import torch.nn as nn
import torch
import torch.nn.functional as F


class WordEmbedding(nn.Module):
    '''
    In : (N, sentence_len)
    Out: (N, sentence_len, embd_size)
    '''
    def __init__(self, vocab_size, embd_size, pre_embd_w=None, is_train_embd=False):
        super(WordEmbedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embd_size)
        if pre_embd_w is not None:
            print('pre embedding weight is set')
            self.embedding.weight = nn.Parameter(pre_embd_w, requires_grad=is_train_embd)

    def forward(self, x):
        return self.embedding(x)


class SimpleLSTM(nn.Module):
    def __init__(self, vocab_size, embd_size, hidden_size, class_size, pre_embd_w=None):
        super(SimpleLSTM, self).__init__()
        self.embd_size = embd_size
        self.hidden_size = hidden_size
        self.embedding = WordEmbedding(vocab_size, embd_size, pre_embd_w)
        self.rnn = nn.GRU(embd_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, class_size)

    def forward(self, x):
        '''
        x: (bs, hist_len+1, sent_len)
        '''
        bs = x.size(0)

        x = self.embedding(x.view(bs, -1)) # (bs, -1, E)
        x, _ = self.rnn(x) # (bs, -1, H)
        x = torch.sum(x, 1) # (bs, H)
        y = self.fc(F.tanh(x.view(bs, -1))) # (bs, class_size)
        y = F.log_softmax(y, -1) # (bs, class_size)
        return y
