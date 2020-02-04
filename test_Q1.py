import collections
import pickle
import random
from sklearn.metrics.pairwise import distance_metrics
import numpy as np
import torch
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
import math
import torch.nn.functional as F
import torch.optim as optim
from numpy import linalg as LA

use_gpu = torch.cuda.is_available()

class SGC(nn.Module):
    """
    A Simple PyTorch Implementation of Logiistic Regression.
    Assuming the features have been preprocessed with k-step graph propagation.
    """
    def __init__(self, nfeat, nclass):
        super(SGC, self).__init__()

        self.W = nn.Linear(nfeat, nclass)

    def forward(self, x):
        return self.W(x)

    
def get_model(model_opt, nfeat, nclass, cuda=True):
    if model_opt == "SGC":
        model = SGC(nfeat=nfeat,
                    nclass=nclass)
    else:
        raise NotImplementedError('model:{} is not implemented!'.format(model_opt))

    if cuda: model.cuda()
    return model

def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)

def save_pickle(file, data):
    with open(file, 'wb') as f:
        pickle.dump(data, f)

def load_pickle(file):
    with open(file, 'rb') as f:
        return pickle.load(f)

def sample_case(ld_dict, shot, n_ways):
    
    sample_class = random.sample(list(ld_dict.keys()), n_ways)
    train_input = []
    test_input = []
    test_labels = []
    train_labels = []
    
    q = random.randint(0, n_ways-1)
        
    for i in range(len(sample_class)):
        class_num = sample_class[i]
        if i == q:
            samples = random.sample(ld_dict[class_num], shot + 1)            
            train_input += samples[:shot]   
            test_input += samples[shot:]
            train_labels += [i] * shot
            test_labels += [i] * 1
        else:
            samples = random.sample(ld_dict[class_num], shot)            
            train_input += samples[:shot]   
            train_labels += [i] * shot
    train_input = np.array(train_input).astype(np.float32)
    test_input = np.array(test_input).astype(np.float32)
    train_labels = torch.tensor(train_labels).cuda()
    test_labels = torch.tensor(test_labels).cuda()
    return train_input, test_input, train_labels, test_labels

def get_adj(train_data, test_data, k, alpha, kappa):
    eps = np.finfo(float).eps
    emb_all = np.append(train_data, test_data, axis=0)
    N = emb_all.shape[0]
    metric = distance_metrics()['cosine']
    S = 1 - metric(emb_all, emb_all)
    S = torch.tensor(S)
    S = S - torch.eye(S.shape[0])
    
    if k>0:
        topk, indices = torch.topk(S, k)
        mask = torch.zeros_like(S)
        mask = mask.scatter(1, indices, 1)
        mask = ((mask+torch.t(mask))>0).type(torch.float32)      
        S    = S*mask 
    
    D       = S.sum(0)
    Dnorm   = torch.diag(torch.pow(D, -0.5))
    E   = torch.matmul(Dnorm,torch.matmul(S, Dnorm))

    E = alpha * torch.eye(E.shape[0]) + E
    E = torch.matrix_power(E, kappa)
    
    E = E.cuda()
    
    train_data = train_data - train_data.mean(0)
    train_data_norm = train_data / LA.norm(train_data, 2, 1)[:, None]
    test_data_norm = test_data / LA.norm(test_data, 2, 1)[:, None]
    features = np.append(train_data_norm, test_data_norm, axis=0)
    
    features = torch.tensor(features).cuda()
    return E, features

def sgc_precompute(features, adj, degree):
    for i in range(degree):
        features = torch.spmm(adj, features)
    
    return features

def compute_confidence_interval(data):
    """
    Compute 95% confidence interval
    :param data: An array of mean accuracy (or mAP) across a number of sampled episodes.
    :return: the 95% confidence interval for this data.
    """
    a = 1.0 * np.array(data)
    m = np.mean(a)
    std = np.std(a)
    pm = 1.96 * (std / np.sqrt(len(a)))
    return m, pm

def train_regression(model,
                     train_features, train_labels,
                     epochs=100, weight_decay=5e-6,
                     lr=0.2):

    optimizer = optim.Adam(model.parameters(), lr=lr,
                           weight_decay=weight_decay)
    accs = []
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        output = model(train_features)
        loss_train = F.cross_entropy(output, train_labels)
        loss_train.backward()
        optimizer.step()
        
    return model

def test_regression(model, test_features, test_labels):
    model.eval()
    return accuracy(model(test_features), test_labels)

def test_sgc(out_dict, shot, n_ways, k, ite=10000, epochs=1000, degree=1, weight_decay=5e-6, lr=0.001):
    accs = []
    for i in range(ite):
        train_data, test_data, train_labels, test_labels  = sample_case(out_dict, shot, n_ways)
        E, features = get_adj(train_data, test_data, k=k, alpha=0.5, kappa=3)
        model_sgc = get_model(model_opt="SGC", nfeat=features.size(1), nclass=n_ways, cuda=True)
        features = sgc_precompute(features, E, degree)
        model_sgc = train_regression(model_sgc, features[:n_ways*shot,], train_labels,
                 epochs=epochs, weight_decay=weight_decay, lr=lr)
        acc_test = test_regression(model_sgc, features[n_ways*shot:,], test_labels)
        accs.append(acc_test)
        
    accs = torch.stack(accs).cpu().detach().numpy()
    acc_mean, acc_conf = compute_confidence_interval(accs)
    return acc_mean, acc_conf

if __name__ == '__main__':
    np.random.seed(10)
    n_ways=5
    save_dir = './checkpoints/miniImagenet/WideResNet28_10_S2M2_R/last'
    out_dict = load_pickle(save_dir + '/output.plk')
    k = 0
    
    for shot in [1,2,3]:
        print(shot)
        accuracy_info_shot_Q1 = test_sgc(out_dict=out_dict, shot=shot, n_ways=n_ways, k=k)
        print('Meta Test: LAST\n{}\t{:.4f}({:.4f})'.format('GVP shot_q', *accuracy_info_shot_Q1))
        
    
    
    
