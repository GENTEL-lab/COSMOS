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

class Evaluator():
    def __init__(self, params, graph_classifier, data):
        self.params = params
        self.graph_classifier = graph_classifier
        self.data = data

    def draw_graph(self, data_pos, save_path='./figure'): 
        """
        graph visualization for debugging
        """ 
        indices_head = torch.nonzero(((data_pos[0].ndata['id'] == 1)), as_tuple=True)  
        indices_head_list = list(indices_head[0].cpu().numpy())  
        indices_tail = torch.nonzero(((data_pos[0].ndata['id'] == 2)), as_tuple=True)  
        indices_tail_list = list(indices_tail[0].cpu().numpy())  
 
        nx_g = dgl.to_networkx(data_pos[0], node_attrs=['id'])  
        
        node_colors = {}  
        plt.figure(figsize=(50, 50))  
        for node in nx_g.nodes():  
            if node in indices_head_list:  
                node_colors[node] = 'red'  
            elif node in indices_tail_list:  
                node_colors[node] = 'yellow'  
            else:  
                node_colors[node] = 'blue'  
                
        nx.draw(nx_g, with_labels=True, node_color=list(node_colors.values()), alpha=0.5)  
        plt.savefig(save_path)

    def write_and_print_edges(data_pos, save_path="./figure"):  
        """
        for the dense graph that's hard to visualize, print the edges for analysis
        """
        src_ids, dst_ids = data_pos.edges()  
        
        with open(save_path, "w") as file:  
            file.write("\nEdges:\n")  
            for src_id, dst_id in zip(src_ids, dst_ids):  
                file.write(str(src_id) + " -> " + str(dst_id) + "\n")  

        print(data_pos.edges())

    def eval(self, save=False):
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
    
        else:
            auc = 0
            auc_pr = 0

        if save:
            pos_test_triplets_path = os.path.join(self.params.main_dir, 'data/{}/{}prot.txt'.format(self.params.dataset, self.data.file_name))
            with open(pos_test_triplets_path) as f:
                pos_triplets = [line.split() for line in f.read().split('\n')[:-1]]
            pos_file_path = os.path.join(self.params.main_dir, 'data/{}/{}_predictions.txt'.format(self.params.dataset, self.data.file_name))
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

        return {'auc': auc, 'auc_pr': auc_pr}
