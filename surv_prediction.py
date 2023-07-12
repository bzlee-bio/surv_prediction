import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sksurv.metrics import concordance_index_censored

def label_modifier(y):
    """ 
    Converting duration and event label into vector type label.

    eg1. Uncensored case
        [3, 1] (duration, event) 
        duration_reference [0,2,4,6,8,10]
        --> y_extend_lab: [0,0,1,1,1,1]
        --> y_mask: [1,1,1,1,1,1]
    eg2. Censored case
        [2, 0] (duration, event)
        --> y_extend_lab: [0,1,1,1,1,1]
        --> y_mask: [1,0,0,0,0,0]

    Args:
        y (list, np.array): Label information including duration and event occurrence

    Returns:
        (y, y_extend_lab, y_mask)
    """
    y_extend_lab = []
    y_mask = []
    for _y in y:
        y_extend_lab.append((np.array(duration_reference) >= _y[0]) * 1)
        if _y[1] == 0:
            y_mask.append(~(np.array(duration_reference) >= _y[0]) * 1)
        else:
            y_mask.append(np.ones_like(y_extend_lab[-1]))
    y_extend_lab = torch.tensor(np.array(y_extend_lab).astype(float))
    y_mask = torch.tensor(np.array(y_mask).astype(int))
    return y, y_extend_lab, y_mask

class data_process(Dataset):
    """ Return batch data including x, y, y_mask, y_orig
    """
    def __init__(self, x, y, device="cpu", x_mean=None, x_std=None) -> None:
        super().__init__()
        self.x = torch.tensor(x.astype(np.float32)).to(device)
        self.y_orig = torch.tensor(y[0].astype(int)).to(device)
        self.y = y[1].to(device)
        self.y_mask = y[2].to(device)

        if x_mean is None:
            self.mean = self.x.mean(0)
            self.std = self.x.std(0)
            # self.x = (self.x - self.mean) / self.std
        else:
            self.mean = x_mean
            self.std = x_std
        
        self.x = (self.x - self.mean) / (self.std +1e-6)

    def return_mean_std(self):
        return self.mean, self.std

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx], self.y_mask[idx], self.y_orig[idx]
    
class model(nn.Module):
    def __init__(self, inp_size, num_pred_tgt, hidden=[400,500,600,800]):
        super().__init__()
        hidden = [inp_size] + hidden + [num_pred_tgt]
        self.linears = nn.ModuleList(
            [nn.Linear(hidden[i], hidden[i + 1]) for i in range(len(hidden) - 1)]
        )
        self.n_linears = len(self.linears)

    def forward(self, x):
        for i in range(self.n_linears - 1):
            x = self.linears[i](x)
            x = F.relu(x)
        x = self.linears[-1](x)
        return x    
    
def eval(model, data_loader, device):
    loss_tot = 0
    for i, batch in enumerate(data_loader):
        model.eval()
        logit = model(batch[0].to(device))
        if i==0:
            logit_tot = logit
            y_tot = batch[3]
        else:
            logit_tot = torch.concat((logit_tot, logit), 0)
            y_tot = torch.concat((y_tot, batch[3]), 0)
        loss = (loss_fn(logit,batch[1].to(device)) * batch[2].to(device)).mean()
        loss_tot+=loss
    c_idx = c_index_calc(logit_tot, y_tot)
    return loss_tot/(i+1), c_idx

def c_index_calc(logit, true_y):
    pred = []
    for i in range(logit.shape[0]):
        min_idx = torch.where(logit[i]>0)[0].min().item()
        pred.append(1/(min_idx+torch.sigmoid(logit[i][min_idx]).item()))
    event_indicator = (true_y[:, 1]==1)
    event_time = true_y[:,0]
    return concordance_index_censored(event_indicator,event_time, pred)

num_pred_nodes = 1000
hidden = [400,500,600,800]
learning_rate = 1e-5
epoch = 100

data_path = ["./data/hvgs", "./data/random_gene1", "./data/random_gene2"]
for dp in data_path:
    f_list = os.listdir(dp)
    print(f_list)
    for _f in f_list:
        
        file = os.path.join(dp, _f)
        print(file)
        df = pd.read_csv(file)
        # device = torch.device('cuda:0')
        device = torch.device('cpu')

        x = df.values[:, 3:]
        y = df.values[:, 1:3]
        seed = 42 ## shuffle random seed num
        duration_max = df.duration.max()

        ## split time point into num_pred_nodes
        duration_reference = [
            duration_max / num_pred_nodes * i for i in range(num_pred_nodes)
        ]
        
        if y[:,1].sum() <6:
            print('There is not enough event occured case')
        else:    
            x_train, x_test, y_train, y_test = train_test_split(
                x, y, random_state=42, train_size=0.8,stratify=y[:,1]
            )
            x_train, x_valid, y_train, y_valid = train_test_split(
                x_train, y_train, random_state=42, train_size=0.8,stratify=y_train[:,1]
            )
            


            print(f"x train {x_train.shape}, positive ratio {y_train[:,1].sum()/len(y_train):.3f}")
            print(f"x valid {x_valid.shape}, positive ratio {y_valid[:,1].sum()/len(y_valid):.3f}")
            print(f"x test {x_test.shape}, positive ratio {y_test[:,1].sum()/len(y_test):.3f}")


            y_train_mask = label_modifier(y_train)
            y_valid_mask = label_modifier(y_valid)
            y_test_mask = label_modifier(y_test)

            data_tr = data_process(x_train, y_train_mask)
            m, s = data_tr.return_mean_std()
            data_val = data_process(x_valid, y_valid_mask, x_mean=m, x_std=s)
            data_test = data_process(x_test, y_test_mask, x_mean=m, x_std=s)
            dl_train = DataLoader(data_tr, batch_size=64, shuffle=True)
            dl_valid = DataLoader(data_val, batch_size=64)
            dl_test = DataLoader(data_test, batch_size=64)

            loss_fn = nn.BCEWithLogitsLoss(reduction="none")
            m = model(x.shape[1], num_pred_nodes, hidden).to(device)
            optimizer = torch.optim.Adam(m.parameters(), lr=learning_rate)

            best_c = [[0]]
                    # loss_val_min = 999
            for ep in range(epoch):
                loss_tr = 0
                for i, tr_batch in enumerate(dl_train):
                    m.train()
                    optimizer.zero_grad()
                    logit = m(tr_batch[0].to(device))
                    loss = (loss_fn(logit,tr_batch[1].to(device)) * tr_batch[2].to(device)).mean()
                    # break
                    loss.backward()
                    optimizer.step()
                    loss_tr+=loss


                loss_val, c_val = eval(m, dl_valid, device)
                loss_test, c_test = eval(m, dl_test, device)
                if c_val[0]>best_c[0][0]:
                # if loss_val_min>loss_val:
                    best_c = [c_val, c_test, ep]

            with open('./perf_summary.csv','a') as f:
                best = [file, best_c[2], best_c[0][0], best_c[1][0], x_train.shape[0], x_valid.shape[0], x_test.shape[0]]
                best = [str(x) for x in best]
                f.write(','.join(best))
                f.write('\n')