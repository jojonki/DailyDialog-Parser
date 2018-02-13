import os

dpath = 'ijcnlp_dailydialog/'
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
            turns = [t.strip() for t in l.split(STOKEN)]
            if turns[-1] == '':
                turns = turns[:-1]
            dlg_data.append(turns)
    with open(act_f, 'r') as f:
        lines = f.readlines()
        for l in lines:
            acts = [int(d) for d in l.strip().split(' ')]
            act_data.append(acts)
    with open(emo_f, 'r') as f:
        lines = f.readlines()
        for l in lines:
            emos = [int(d) for d in l.strip().split(' ')]
            emo_data.append(emos)
    return dlg_data, act_data, emo_data


train_data, train_act_data, train_emo_data = load_data(dpath, 'train')
test_data, test_act_data, test_emo_data    = load_data(dpath, 'test')
dev_data, dev_act_data, dev_emo_data       = load_data(dpath, 'validation')
