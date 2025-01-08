import click as ck
import pandas as pd
from deepgo.utils import Ontology, propagate_annots
import torch as th
import numpy as np
from torch import nn
from torch.nn import functional as F
from torch import optim
import copy
from torch.utils.data import DataLoader, IterableDataset, TensorDataset
from itertools import cycle
import math
from dgl.nn import GraphConv
import dgl
from deepgo.torch_utils import FastTensorDataLoader
import csv
from torch.optim.lr_scheduler import MultiStepLR
from deepgo.data import load_ppi_data, load_normal_forms
from deepgo.metrics import compute_roc
from multiprocessing import Pool
from functools import partial
from tqdm import tqdm
from sklearn.metrics import roc_curve, auc, roc_auc_score, average_precision_score, precision_recall_curve ,auc
import logging
from torch_geometric.graphgym import get_current_gpu_usage
import pickle


@ck.command()
@ck.option(
    '--data-root', '-dr', default='data',
    help='Data folder')
@ck.option(
    '--ont', '-ont', default='mf', type=ck.Choice(['mf', 'bp', 'cc']),
    help='GO subontology')
@ck.option(
    '--dataset', '-ds', default='mf', help='new dataset name'
)
@ck.option(
    '--test-data-name', '-td', default='test', type=ck.Choice(['test', 'nextprot']),
    help='Test data set name')
@ck.option(
    '--batch-size', '-bs', default=32,
    help='Batch size for training')
@ck.option(
    '--epochs', '-ep', default=100,
    help='Training epochs')
@ck.option(
    '--load', '-ld', is_flag=True, help='Load Model?')
@ck.option(
    '--device', '-d', default='cuda:0',
    help='Device')
def main(data_root, ont, test_data_name, batch_size, epochs, load, device, dataset):
    go_file = f'{data_root}/go.obo'
    model_file = f'{data_root}/{ont}/{dataset}/dgg.th'
    terms_file = f'{data_root}/{ont}/all_terms.pkl'
    out_file = f'{data_root}/{ont}/{dataset}/{test_data_name}_predictions_dgg.pkl'

    go = Ontology(go_file, with_rels=True)

    loss_func = nn.BCELoss()
    features_length = None
    features_column = 'interpros'
    ppi_graph_file = f'{dataset}/ppi_{test_data_name}.bin'    
    test_data_file = f'{test_data_name}_data.pkl'
    iprs_dict, terms_dict, graph, train_nids, valid_nids, test_nids, data, labels, test_df = load_ppi_data(
        data_root, ont, dataset, features_length, features_column, test_data_file, ppi_graph_file)
    n_terms = len(terms_dict)
    features_length = len(iprs_dict)
    go_norm_file = f'{data_root}/go.norm'
    nf1, nf2, nf3, nf4, rels_dict, zero_classes = load_normal_forms(
        go_norm_file, terms_dict)

    
    valid_labels = labels[valid_nids].numpy()
    test_labels = labels[test_nids].numpy()

    labels = labels.to(device)

    
    graph = graph.to(device)

    train_nids = train_nids.to(device)
    valid_nids = valid_nids.to(device)
    test_nids = test_nids.to(device)

    net = DeepGraphGOModel(features_length, n_terms, device).to(device)
    total = sum([param.nelement() for param in net.parameters()])
    sampler = dgl.dataloading.MultiLayerFullNeighborSampler(2)
    train_dataloader = dgl.dataloading.DataLoader(
        graph, train_nids, sampler,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=0)

    valid_dataloader = dgl.dataloading.DataLoader(
        graph, valid_nids, sampler,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=0)

    test_dataloader = dgl.dataloading.DataLoader(
        graph, test_nids, sampler,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=0)
    
    
    optimizer = th.optim.Adam(net.parameters(), lr=1e-3)
    scheduler = MultiStepLR(optimizer, milestones=[1, 3,], gamma=0.1)

    best_loss = 10000.0
    if not load:
        print('Training the model')
        for epoch in range(epochs):
            net.train()
            train_loss = 0
            train_steps = int(math.ceil(len(train_nids) / batch_size))
            with ck.progressbar(length=train_steps, show_pos=True) as bar:
                for input_nodes, output_nodes, blocks in train_dataloader:
                    bar.update(1)
                    logits = net(input_nodes, output_nodes, blocks)
                    gpu_memory = get_current_gpu_usage()
                    batch_labels = labels[output_nodes]
                    loss = F.binary_cross_entropy(logits, batch_labels)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    train_loss += loss.detach().item()
            
            train_loss /= train_steps
            
            print('Validation')
            net.eval()
            with th.no_grad():
                valid_steps = int(math.ceil(len(valid_nids) / batch_size))
                valid_loss = 0
                preds = []
                with ck.progressbar(length=valid_steps, show_pos=True) as bar:
                    for input_nodes, output_nodes, blocks in valid_dataloader:
                        bar.update(1)
                        logits = net(input_nodes, output_nodes, blocks)
                        batch_labels = labels[output_nodes]
                        batch_loss = F.binary_cross_entropy(logits, batch_labels)
                        valid_loss += batch_loss.detach().item()
                        preds = np.append(preds, logits.detach().cpu().numpy())
                valid_loss /= valid_steps
                roc_auc = compute_roc(valid_labels, preds)
                print(f'Epoch {epoch}: Loss - {train_loss}, Valid loss - {valid_loss}, AUC - {roc_auc}, gpu - {gpu_memory} s')
            if valid_loss < best_loss:
                best_loss = valid_loss
                print('Saving model')
                th.save(net.state_dict(), model_file)

            scheduler.step()
            

    # Loading best model
    print('Loading the best model')
    net.load_state_dict(th.load(model_file,map_location='cpu'))
    net.eval()
    with th.no_grad():
        test_steps = int(math.ceil(len(test_nids) / batch_size))
        test_loss = 0
        preds = []
        with ck.progressbar(length=test_steps, show_pos=True) as bar:
            for input_nodes, output_nodes, blocks in test_dataloader:
                bar.update(1)
                logits = net(input_nodes, output_nodes, blocks)
                batch_labels = labels[output_nodes]
                batch_loss = F.binary_cross_entropy(logits, batch_labels)
                test_loss += batch_loss.detach().cpu().item()
                preds.append(logits.detach().cpu().numpy())
            test_loss /= test_steps
        preds = np.concatenate(preds)
        roc_auc = compute_roc(test_labels, preds)
        print(f'Test Loss - {test_loss}, AUC - {roc_auc}')

    with open(f"{data_root}/{ont}/{dataset}/test_terms.pkl", 'rb') as file:
        defins_df = pickle.load(file)
    defins = defins_df['gos'].tolist()
    zero_terms = [term for term in zero_classes if term in defins ] #and go.get_namespace(term) == NAMESPACES[ont]]
    zero_terms_dict = {element: index for index, element in enumerate(zero_terms)}
    test_dict = {key: terms_dict[key] for key in terms_dict if key in defins}
    for i, key in enumerate(test_dict, start=0):
        test_dict[key] = i
    idx_lst = []
    for i, go_id in enumerate(test_dict):
        idx = terms_dict[go_id]
        idx_lst.append(idx)
    preds = preds[:, idx_lst]

    
    label = get_label(test_df, test_dict)
    GT = label.numpy()
    prop_annots = {}
    for go_id, j in tqdm(test_dict.items()):
        scores = preds[:, j]
        ancestors = go.get_ancestors(go_id)
        for sup_go in ancestors:
            if sup_go in prop_annots:
                prop_annots[sup_go] = np.maximum(prop_annots[sup_go], scores)
            else:
                prop_annots[sup_go] = scores
    for go_id, score in prop_annots.items():
        if go_id in test_dict:
            preds[:, test_dict[go_id]] = score


    np.save(f'{data_root}/{ont}/{dataset}/predictions_deepgozero_zero_preds.npy', preds)
    np.save(f'{data_root}/{ont}/{dataset}/predictions_deepgozero_zero_GT.npy', GT)
    pos_indices, neg_indices = sample_indices(GT)
    scores_pos, scores_neg, gts_pos, gts_neg = process_preds(np.array(preds), pos_indices, neg_indices)
    auc_ = roc_auc_score(gts_pos + gts_neg, scores_pos + scores_neg)
    auc_pr = average_precision_score(gts_pos + gts_neg, scores_pos + scores_neg)
    fmax, best_threshold = calculate_fmax(np.array(scores_pos + scores_neg),np.array(gts_pos + gts_neg))
    print("auc: %s, auprc: %s, fmax: %s, threshold: %s", auc_, auc_pr, fmax, best_threshold)
    roc_auc = compute_roc(GT, preds)
    auprc = average_precision_score(GT.flatten(), preds.flatten())

    index = process_dataframe(GT)
    ranks,scores = get_rank(preds, index)
    isHit1List = [x for x in ranks if x <= 1]
    isHit5List = [x for x in ranks if x <= 5]
    isHit10List = [x for x in ranks if x <= 10]
    hits_1 = len(isHit1List) / len(ranks)
    hits_5 = len(isHit5List) / len(ranks)
    hits_10 = len(isHit10List) / len(ranks)
    print("hit1: %s, hit5: %s, hit10: %s", hits_1, hits_5, hits_10) 



class MLPBlock(nn.Module):

    def __init__(self, in_features, out_features, bias=True, layer_norm=False, dropout=0.5, activation=nn.ReLU):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias)
        self.activation = activation()
        self.layer_norm = nn.LayerNorm(out_features) if layer_norm else None
        self.dropout = nn.Dropout(dropout) if dropout else None

    def forward(self, x):
        x = self.activation(self.linear(x))
        if self.layer_norm:
            x = self.layer_norm(x)
        if self.dropout:
            x = self.dropout(x)
        return x


class DeepGraphGOModel(nn.Module):

    def __init__(self, nb_iprs, nb_gos, device, hidden_dim=1024):
        super().__init__()
        self.nb_gos = nb_gos
        self.net1 = MLPBlock(nb_iprs, hidden_dim)
        self.conv1 = GraphConv(hidden_dim, hidden_dim)
        self.conv2 = GraphConv(hidden_dim, hidden_dim)
        input_length = hidden_dim
        self.net2 = nn.Sequential(
            nn.Linear(hidden_dim, nb_gos),
            nn.Sigmoid())

        
    def forward(self, input_nodes, output_nodes, blocks, residual=True):
        g1 = blocks[0]
        g2 = blocks[1]
        features = g1.ndata['feat']['_N']
        x = self.net1(features)
        x = self.conv1(g1, x)
        x = self.conv2(g2, x)
        logits = self.net2(x)
        return logits

def compute_roc(labels, preds):
    # Compute ROC curve and ROC area for each class
    fpr, tpr, _ = roc_curve(labels.flatten(), preds.flatten())
    roc_auc = auc(fpr, tpr)

    return roc_auc

def calculate_fmax(predictions, ground_truth):
    precision, recall, thresholds = precision_recall_curve(ground_truth.flatten(), predictions.flatten())
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-5)
    fmax = np.max(f1_scores)
    best_threshold = thresholds[np.argmax(f1_scores)]

    return fmax, best_threshold

def get_label(df, terms_dict):
    labels = th.zeros((len(df), len(terms_dict)), dtype=th.float32)
    for i, row in enumerate(df.itertuples()):
        for go_id in row.prop_annotations: # prop_annotations for full model
            if go_id in terms_dict:
                g_id = terms_dict[go_id]
                labels[i, g_id] = 1
    return labels

def sample_indices(df):
    pos_indices = []
    neg_indices = []

    for row in range(len(df)):
        values = df[row]
        pos_index = np.where(values == 1)[0]
        neg_index = np.where(values == 0)[0]
        nums_to_sample = len(pos_index)
        neg_sample_indices = np.random.choice(neg_index, size=nums_to_sample, replace=False)

        pos_indices.append(pos_index)
        neg_indices.append(neg_sample_indices)

    return pos_indices, neg_indices

def process_preds(df, pos_indices, neg_indices):
    scores_pos = []
    scores_neg = []
    gts_pos = []
    gts_neg = []

    for row_idx in range(len(df)):
        pos_idx = pos_indices[row_idx]
        neg_idx = neg_indices[row_idx]

        for col_idx in pos_idx:
            score_pos = df[row_idx, col_idx]
            GT_pos = 1
            scores_pos.append(score_pos)
            gts_pos.append(GT_pos)

        for col_idx in neg_idx:
            score_neg = df[row_idx, col_idx]
            GT_neg = 0
            scores_neg.append(score_neg)
            gts_neg.append(GT_neg)

    return scores_pos, scores_neg, gts_pos, gts_neg

def process_dataframe(df):
    indices_group = []

    for row in range(len(df)):
        indice_row = []
        values = df[row]
        pos_rels = np.where(values == 1)[0]
        neg_rels = np.where(values == 0)[0]
            
        for pos_rel in pos_rels:
            nums_to_sample = 49
            neg_sample_indices = np.random.choice(neg_rels, size=nums_to_sample, replace=True)
            indice_row.append([pos_rel] + list(neg_sample_indices))
        indices_group.append(indice_row)

    return indices_group

def get_rank(df, indices):
    ranks = []
    scores = []
    for row_idx in range(len(df)):
        row_values = df[row_idx]
        row_indices = indices[row_idx]
        for items in row_indices:
            score = row_values[items]
            rank = np.argwhere(np.argsort(score)[::-1] == 0) + 1
            ranks.append(rank)
            scores.append(score)
    return ranks, scores

if __name__ == '__main__':
    main()
