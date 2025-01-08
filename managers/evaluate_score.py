import os
import numpy as np
import torch
import pdb
from sklearn import metrics
import torch.nn.functional as F
from torch.utils.data import DataLoader
import dgl
import networkx as nx
import matplotlib.pyplot as plt
from tqdm import tqdm
import pickle

class Evaluate_score():
    def __init__(self, params, graph_classifier, data):
        self.params = params
        self.graph_classifier = graph_classifier
        self.data = data

    def calculate_fmax(self, predictions, ground_truth):
        precision, recall, thresholds = metrics.precision_recall_curve(ground_truth.flatten(), predictions.flatten())
        f1_scores = 2 * (precision * recall) / (precision + recall + 0.000000000001)
        fmax = np.max(f1_scores)
        best_threshold = thresholds[np.argmax(f1_scores)]
    
        return fmax, best_threshold

    def eval(self, save=True):
        pos_scores = []
        pos_labels = []
        neg_scores = []
        neg_labels = []
        pos_embeds = []
        ranks = []
        dataloader = DataLoader(self.data, batch_size=self.params.batch_size, shuffle=False, num_workers=self.params.num_workers, collate_fn=self.params.collate_fn)

        self.graph_classifier.eval()
        with torch.no_grad():
            for b_idx, batch in tqdm(enumerate(dataloader)):

                data_pos, targets_pos, data_neg, targets_neg = self.params.move_batch_to_device(batch, self.params.device)
                score_pos, embed_pos = self.graph_classifier(data_pos)
                score_neg, embed_neg = self.graph_classifier(data_neg)


                pos_scores += score_pos.squeeze(1).detach().cpu().tolist()
                neg_scores += score_neg.squeeze(1).detach().cpu().tolist()
                pos_labels += targets_pos.tolist()
                neg_labels += targets_neg.tolist()
                pos_embeds += embed_pos.squeeze(1).detach().cpu().tolist()


        if len(pos_labels)> 0:

            auc = metrics.roc_auc_score(pos_labels + neg_labels, pos_scores + neg_scores)
            auc_pr = metrics.average_precision_score(pos_labels + neg_labels, pos_scores + neg_scores)
            fpr, tpr, thresholds = metrics.roc_curve(pos_labels + neg_labels, pos_scores + neg_scores)
            roc_auc = metrics.auc(fpr, tpr)
            precision, recall, thresholds = metrics.precision_recall_curve(pos_labels + neg_labels, pos_scores + neg_scores)
            print('fpr:', fpr, 'tpr:', tpr, 'thresholds:', thresholds, 'roc_auc:', roc_auc, 'precision:', precision, 'recall:', recall)
            fmax, best_threshold = self.calculate_fmax(np.array(pos_scores + neg_scores),np.array(pos_labels + neg_labels))
        else:
            auc = 0
            auc_pr = 0

        if save:
            pos_test_triplets_path = os.path.join(self.params.main_dir, 'data/{}/{}prot.txt'.format(self.params.dataset, self.data.file_name))
            with open(pos_test_triplets_path) as f:
                pos_triplets = [line.split() for line in f.read().split('\n')[:-1]]
            pos_file_path = os.path.join(self.params.main_dir, 'data/{}/{}_predictions.txt'.format(self.params.dataset, self.data.file_name))
            print(pos_test_triplets_path, pos_file_path)
            with open(pos_file_path, "w") as f:
                for ([s, r, o], score, label) in zip(pos_triplets, pos_scores, pos_labels):
                    f.write('\t'.join([s, r, o, str(score), str(label)]) + '\n')

            pos_embed_path = os.path.join(self.params.main_dir, 'data/{}/{}_pos_embeds.txt'.format(self.params.dataset, self.data.file_name))
            with open(pos_embed_path, 'w') as f:
                for ([s, r, o], score, embed) in zip(pos_triplets, pos_scores, pos_embeds):
                    f.write('\t'.join([s, r, o, str(score), str(embed)]) + '\n')

            neg_test_triplets_path = os.path.join(self.params.main_dir, 'data/{}/neg_{}_0.txt'.format(self.params.dataset, self.data.file_name))
            with open(neg_test_triplets_path) as f:
                neg_triplets = [line.split() for line in f.read().split('\n')[:-1]]
            neg_file_path = os.path.join(self.params.main_dir, 'data/{}/neg_{}_{}_predictions.txt'.format(self.params.dataset, self.data.file_name, self.params.constrained_neg_prob))
            with open(neg_file_path, "w") as f:
                for ([s, r, o], score, label) in zip(neg_triplets, neg_scores, neg_labels):
                    f.write('\t'.join([s, r, o, str(score), str(label)]) + '\n')

        return {'auc': auc, 'auc_pr': auc_pr, 'fmax': fmax, 'threshold': best_threshold}
