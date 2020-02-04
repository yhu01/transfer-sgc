import collections
import pickle
import random
from time import perf_counter
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
from time import perf_counter

use_gpu = torch.cuda.is_available()

class GraphConvolution(nn.Module):
    """
    A Graph Convolution Layer (GCN)
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.W = nn.Linear(in_features, out_features, bias=bias)
        self.init()

    def init(self):
        stdv = 1. / math.sqrt(self.W.weight.size(1))
        self.W.weight.data.uniform_(-stdv, stdv)

    def forward(self, x, adj):       
        support = self.W(x)
        output = torch.spmm(adj, support)
        return output
        

class GCN(nn.Module):
    """
    A Two-layer GCN.
    """
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GCN, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nclass)
        self.dropout = dropout

    def forward(self, x, adj, use_relu=True):
        x = self.gc1(x, adj)
        if use_relu:
            x = F.relu(x)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        return x
    
def get_model(model_opt, nfeat, nclass, nhid=0, dropout=0, cuda=True):
    if model_opt == "GCN":
        model = GCN(nfeat=nfeat,
                    nhid=nhid,
                    nclass=nclass,
                    dropout=dropout)
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

def sample_case(ld_dict, shot, n_ways, n_queries):
    sample_class = random.sample(list(ld_dict.keys()), n_ways)
    train_input = []
    test_input = []
    
    for each_class in sample_class:
        samples = random.sample(ld_dict[each_class], shot + n_queries)
        train_input += samples[:shot]
        test_input += samples[shot:]
    train_input = np.array(train_input).astype(np.float32)
    test_input = np.array(test_input).astype(np.float32)
    return train_input, test_input

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
    test_data = test_data - test_data.mean(0)
    test_data_norm = test_data / LA.norm(test_data, 2, 1)[:, None]
    features = np.append(train_data_norm, test_data_norm, axis=0)
    
    #features = np.append(train_data, test_data, axis=0)
    features = torch.tensor(features).cuda()
    return E, features

def get_labels(n_ways, shot, n_queries):
    train_labels = []
    test_labels = []
    classes = [i for i in range(n_ways)]
    for each_class in classes:
        train_labels += [each_class] * shot
        test_labels += [each_class] * n_queries

    train_labels = torch.tensor(train_labels).cuda()
    test_labels = torch.tensor(test_labels).cuda()
    return train_labels, test_labels

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

def train_regression(model, features, shot, n_ways, train_labels, E, epochs=100, weight_decay=5e-6, lr=0.2):

    optimizer = optim.Adam(model.parameters(), lr=lr,
                           weight_decay=weight_decay)
    accs = []
    t = perf_counter()
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        output = model(features, E)
        loss_train = F.cross_entropy(output[:shot*n_ways,], train_labels)
        loss_train.backward()
        optimizer.step()
        
    train_time = perf_counter()-t
    return model, train_time

def test_regression(model, features, shot, n_ways, test_labels, E):
    model.eval()
    output = model(features, E)
    return accuracy(output[shot*n_ways:,], test_labels)

def test_sgc(out_dict, shot, n_ways, n_queries, k, ite=10000, epochs=1000, weight_decay=5e-6, lr=0.001, dropout=0):
    accs = []
    for i in range(ite):
        train_data, test_data = sample_case(out_dict, shot, n_ways, n_queries)
        E, features = get_adj(train_data, test_data, k=k, alpha=0.5, kappa=3)
        train_labels, test_labels = get_labels(n_ways, shot, n_queries)
        model_sgc = get_model(model_opt="GCN", nfeat=features.size(1), nclass=n_ways, nhid=1024, dropout=0, cuda=True)
        model_sgc, train_time = train_regression(model_sgc, features, shot, n_ways, train_labels, E=E, epochs=epochs, weight_decay=weight_decay, lr=lr)
        acc_test = test_regression(model_sgc, features, shot, n_ways, test_labels, E)
        accs.append(acc_test)
        
    accs = torch.stack(accs).cpu().detach().numpy()
    acc_mean, acc_conf = compute_confidence_interval(accs)
    return acc_mean, acc_conf, train_time

if __name__ == '__main__':
    np.random.seed(10)
    n_ways=5
    n_queries=15
    save_dir = './checkpoints/miniImagenet/WideResNet28_10_S2M2_R/last'
    out_dict = load_pickle(save_dir + '/output.plk')
    k = 10
    accuracy_info_shot1 = test_sgc(out_dict=out_dict, shot=1, n_ways=n_ways, n_queries=n_queries, k=k)
    print('Meta Test: LAST\n{}\t{:.4f}({:.4f})'.format('GVP shot1', *accuracy_info_shot1))
    accuracy_info_shot5 = test_sgc(out_dict=out_dict, shot=5, n_ways=n_ways, n_queries=n_queries, k=k)
    print('Meta Test: LAST\n{}\t{:.4f}({:.4f})'.format('GVP shot5', *accuracy_info_shot5))    
