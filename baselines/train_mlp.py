import click as ck
import pandas as pd
import torch as th
import numpy as np
from torch import nn
from torch.nn import functional as F
from torch import optim
from sklearn.metrics import roc_curve, auc, matthews_corrcoef
import copy
from torch.utils.data import DataLoader, IterableDataset, TensorDataset
from itertools import cycle
import math
from deepgo.aminoacids import to_onehot, MAXLEN
from dgl.nn import GraphConv
import dgl
from deepgo.torch_utils import FastTensorDataLoader
import csv
from torch.optim.lr_scheduler import MultiStepLR
from deepgo.models import MLPModel
from deepgo.data import load_data
from deepgo.utils import Ontology, propagate_annots
from multiprocessing import Pool
from functools import partial
from deepgo.metrics import compute_roc
from sklearn.metrics import roc_curve, auc, matthews_corrcoef, roc_auc_score, average_precision_score, precision_recall_curve
from torchprofile import profile_macs
from torch_geometric.graphgym import get_current_gpu_usage
import time


@ck.command()
@ck.option(
    '--data-root', '-dr', default='data',
    help='Data folder')
@ck.option(
    '--ont', '-ont', default='mf', type=ck.Choice(['mf', 'bp', 'cc']),
    help='GO subontology')
@ck.option(
    '--dataset', '-ds', default='mf_both', help='dataset name'
)
@ck.option(
    '--model-name', '-m', type=ck.Choice([
        'mlp', 'mlp_esm']),
    default='mlp',
    help='Prediction model name')
@ck.option(
    '--test-data-name', '-td', default='test', type=ck.Choice(['test', 'nextprot']),
    help='Test data set name')
@ck.option(
    '--batch-size', '-bs', default=37,
    help='Batch size for training')
@ck.option(
    '--epochs', '-ep', default=256,
    help='Training epochs')
@ck.option(
    '--load', '-ld', is_flag=True, help='Load Model?')
@ck.option(
    '--device', '-d', default='cuda:0',
    help='Device')
def main(data_root, ont, dataset, model_name, test_data_name, batch_size, epochs, load, device):
    go_file = f'{data_root}/go.obo'
    model_file = f'{data_root}/{ont}/{dataset}/{model_name}.th'
    terms_file = f'{data_root}/{ont}/all_terms.pkl'
    out_file = f'{data_root}/{ont}/{dataset}/{test_data_name}_predictions_{model_name}.pkl'

    go = Ontology(go_file, with_rels=True)
    loss_func = nn.BCELoss()

    if model_name.find('esm') != -1:
        features_length = 2560
        features_column = 'esm2'
    else:
        features_length = None 
        features_column = 'interpros'

    test_data_file = f'{test_data_name}_data_{dataset}.pkl'
    iprs_dict, terms_dict, train_data, valid_data, test_data, test_df = load_data(
        data_root, ont, terms_file, dataset, features_length, features_column, test_data_file=test_data_file)
    n_terms = len(terms_dict)
    if features_column == 'interpros':
        features_length = len(iprs_dict)
    net = MLPModel(features_length, n_terms, device).to(device)
    total = sum([param.nelement() for param in net.parameters()])
    
    train_features, train_labels = train_data
    valid_features, valid_labels = valid_data
    test_features, test_labels = test_data
    
    train_loader = FastTensorDataLoader(
        *train_data, batch_size=batch_size, shuffle=True)
    valid_loader = FastTensorDataLoader(
        *valid_data, batch_size=batch_size, shuffle=False)
    test_loader = FastTensorDataLoader(
        *test_data, batch_size=batch_size, shuffle=False)

    valid_labels = valid_labels.detach().cpu().numpy()
    test_labels = test_labels.detach().cpu().numpy()
    
    optimizer = th.optim.Adam(net.parameters(), lr=1e-3)
    
    best_loss = 10000.0
    if not load:
        print('Training the model')
        for epoch in range(epochs):
            net.train()
            time_start = time.time()
            train_loss = 0
            train_steps = int(math.ceil(len(train_labels) / batch_size))
            with ck.progressbar(length=train_steps, show_pos=True) as bar:
                for batch_features, batch_labels in train_loader:
                    bar.update(1)
                    batch_features = batch_features.to(device)
                    batch_labels = batch_labels.to(device)
                    logits = net(batch_features)
                    gpu_memory = get_current_gpu_usage()
                    loss = F.binary_cross_entropy(logits, batch_labels)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    train_loss += loss.detach().item()
            
            train_loss /= train_steps
            
            print('Validation')
            net.eval()
            with th.no_grad():
                valid_steps = int(math.ceil(len(valid_labels) / batch_size))
                valid_loss = 0
                preds = []
                with ck.progressbar(length=valid_steps, show_pos=True) as bar:
                    for batch_features, batch_labels in valid_loader:
                        bar.update(1)
                        batch_features = batch_features.to(device)
                        batch_labels = batch_labels.to(device)
                        logits = net(batch_features)
                        batch_loss = F.binary_cross_entropy(logits, batch_labels)
                        valid_loss += batch_loss.detach().item()
                        preds = np.append(preds, logits.detach().cpu().numpy())
                valid_loss /= valid_steps
                roc_auc = compute_roc(valid_labels, preds)
                time_elapsed = time.time() - time_start
                print(f'Epoch {epoch}: Loss - {train_loss}, Valid loss - {valid_loss}, AUC - {roc_auc}, gpu - {gpu_memory} in {time_elapsed}s')
            if valid_loss < best_loss:
                best_loss = valid_loss
                print('Saving model')
                th.save(net.state_dict(), model_file)

        
    # Loading best model
    print('Loading the best model')
    net.load_state_dict(th.load(model_file,map_location='cpu'))
    net.eval()
    with th.no_grad():
        test_steps = int(math.ceil(len(test_labels) / batch_size))
        test_loss = 0
        preds = []
        with ck.progressbar(length=test_steps, show_pos=True) as bar:
            for batch_features, batch_labels in test_loader:
                bar.update(1)
                batch_features = batch_features.to(device)
                batch_labels = batch_labels.to(device)
                logits = net(batch_features)
                batch_loss = F.binary_cross_entropy(logits, batch_labels)
                test_loss += batch_loss.detach().cpu().item()
                preds.append(logits.detach().cpu().numpy())
            test_loss /= test_steps
        preds = np.concatenate(preds)
        roc_auc = compute_roc(test_labels, preds)
        print(f'Test Loss - {test_loss}, AUC - {roc_auc}')

    GT = test_labels
    np.save(f'{data_root}/{ont}/{dataset}/predictions_deepgozero_zero_preds.npy', preds)
    np.save(f'{data_root}/{ont}/{dataset}/predictions_deepgozero_zero_GT.npy', GT)
    # preds, GT = mask_train(preds, GT, go_list)
    pos_indices, neg_indices = sample_indices(GT)
    scores_pos, scores_neg, gts_pos, gts_neg = process_preds(np.array(preds), pos_indices, neg_indices)
    auc_ = roc_auc_score(gts_pos + gts_neg, scores_pos + scores_neg)
    auc_pr = average_precision_score(gts_pos + gts_neg, scores_pos + scores_neg)
    fmax, best_threshold = calculate_fmax(np.array(scores_pos + scores_neg),np.array(gts_pos + gts_neg))
    print("auc: %s, auprc: %s, fmax: %s, threshold: %s", auc_, auc_pr, fmax, best_threshold)
    roc_auc = compute_roc(test_labels, preds)
    auprc = average_precision_score(test_labels.flatten(), preds.flatten())

    index = process_dataframe(GT)
    ranks,scores = get_rank(preds, index)
    isHit1List = [x for x in ranks if x <= 1]
    isHit5List = [x for x in ranks if x <= 5]
    isHit10List = [x for x in ranks if x <= 10]
    hits_1 = len(isHit1List) / len(ranks)
    hits_5 = len(isHit5List) / len(ranks)
    hits_10 = len(isHit10List) / len(ranks)
    print("hit1: %s, hit5: %s, hit10: %s", hits_1, hits_5, hits_10) 

    
def compute_roc(labels, preds):
    # Compute ROC curve and ROC area for each class
    fpr, tpr, _ = roc_curve(labels.flatten(), preds.flatten())
    roc_auc = auc(fpr, tpr)

    return roc_auc

def calculate_fmax(predictions, ground_truth):
    precision, recall, thresholds = precision_recall_curve(ground_truth.flatten(), predictions.flatten())
    f1_scores = 2 * (precision * recall) / (precision + recall)
    fmax = np.max(f1_scores)
    best_threshold = thresholds[np.argmax(f1_scores)]

    return fmax, best_threshold

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
            neg_sample_indices = np.random.choice(neg_rels, size=nums_to_sample, replace=False)
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
