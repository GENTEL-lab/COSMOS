import pdb
import os
import math
import random
import argparse
import numpy as np
import time as time

from graph_utils import incidence_matrix, get_edge_count
from dgl_utils import _bfs_relational
from data_utils import process_files, save_to_file
from tqdm import tqdm 
from goatools.obo_parser import GODag
from goatools.gosubdag.gosubdag import GoSubDag

def get_active_relations(adj_list):
    act_rels = []
    for r, adj in enumerate(adj_list):
        if len(adj.tocoo().row.tolist()) > 0:
            act_rels.append(r)
    return act_rels


def get_avg_degree(adj_list):
    adj_mat = incidence_matrix(adj_list)
    degree = []
    for node in range(adj_list[0].shape[0]):
        degree.append(np.sum(adj_mat[node, :]))
    return np.mean(degree)


def get_splits(adj_list, nodes, valid_rels=None, valid_ratio=0.1, test_ratio=0.1):
    '''
    Get train/valid/test splits of the sub-graph defined by the given set of nodes. The relations in this subbgraph are limited to be among the given valid_rels.
    '''

    # Extract the subgraph
    subgraph = [adj[nodes, :][:, nodes] for adj in adj_list]

    # Get the relations that are allowed to be sampled
    active_rels = get_active_relations(subgraph)
    common_rels = list(set(active_rels).intersection(set(valid_rels)))

    print('Average degree : ', get_avg_degree(subgraph))
    print('Nodes: ', len(nodes))
    print('Links: ', np.sum(get_edge_count(subgraph)))
    print('Active relations: ', len(common_rels))

    # get all the triplets satisfying the given constraints
    all_triplets = []
    for r in common_rels:
        for (i, j) in zip(subgraph[r].tocoo().row, subgraph[r].tocoo().col):
            all_triplets.append([nodes[i], nodes[j], r])
    all_triplets = np.array(all_triplets)

    # delete the triplets which correspond to self connections
    ind = np.argwhere(all_triplets[:, 0] == all_triplets[:, 1])
    all_triplets = np.delete(all_triplets, ind, axis=0)
    print('Links after deleting self connections : %d' % len(all_triplets))

    # get the splits according to the given ratio
    np.random.shuffle(all_triplets)
    train_split = int(math.ceil(len(all_triplets) * (1 - valid_ratio - test_ratio)))
    valid_split = int(math.ceil(len(all_triplets) * (1 - test_ratio)))

    train_triplets = all_triplets[:train_split]
    valid_triplets = all_triplets[train_split: valid_split]
    test_triplets = all_triplets[valid_split:]

    return train_triplets, valid_triplets, test_triplets, common_rels


def get_subgraph(adj_list, hops, max_nodes_per_hop, protein_list, id2entity, roots_num):
    '''
    Samples a subgraph around randomly chosen root nodes upto hops with a limit on the nodes selected per hop given by max_nodes_per_hop
    '''

    # collapse the list of adj mattricees to a single matrix
    A_incidence = incidence_matrix(adj_list)

    idx = np.random.choice(range(len(A_incidence.tocoo().row)), size=roots_num, replace=False)
    coo_matrix = A_incidence.tocoo()
    roots = {coo_matrix.row[id] for id in idx} | {coo_matrix.col[id] for id in idx}
    root_lst = list(roots)

    # get the neighbor nodes within a limit of hops
    bfs_generator = _bfs_relational(A_incidence, roots, max_nodes_per_hop)
    lvls = list()
    for _ in range(1):
        lvls.append(next(bfs_generator))

    nodes = list(roots) + list(set().union(*lvls))
  
    return nodes

def get_SPUG_subgraph(adj_list, hops, max_nodes_per_hop, train_prot_nodes, roots_num):
    '''
    Samples a subgraph around randomly chosen root nodes upto hops with a limit on the nodes selected per hop given by max_nodes_per_hop
    '''

    # collapse the list of adj mattricees to a single matrix
    A_incidence = incidence_matrix(adj_list)

    # choose a set of random root nodes
    #
    idx = np.random.choice(range(len(A_incidence.tocoo().row)), size=roots_num, replace=False)

    roots = set([A_incidence.tocoo().row[id] for id in idx] + [A_incidence.tocoo().col[id] for id in idx]) 

    # get the neighbor nodes within a limit of hops
    bfs_generator = _bfs_relational(A_incidence, roots, max_nodes_per_hop)
    lvls = list()
    for _ in range(hops):
        lvls.append(next(bfs_generator))

    nodes = list(roots) + list(set().union(*lvls))

    return nodes


def mask_nodes(adj_list, nodes, protein_list, go_list, mask_type):
    # mask the nodes according to the situation
    if mask_type == "none":     
        print("no need for mask")   
        return adj_list  
  
    masked_adj_list = [adj.copy() for adj in adj_list]  
      
    if mask_type == "both": 
        print("mask protein and GO") 
        nodes_to_process = nodes  
    elif mask_type == "protein":  
        print("mask protein") 
        protein_set = set(protein_list)  
        nodes_to_process = [node for node in tqdm(nodes) if node in protein_set]  
    elif mask_type == "GO":  
        print("mask GO") 
        GO_list = set(go_list)  
        nodes_to_process = [node for node in tqdm(nodes) if node in GO_list]  
    else:  
        raise ValueError("Invalid mask_type")

    print("the mask nodes is", len(nodes), len(nodes_to_process))  
    
    for node in tqdm(nodes_to_process):
        for i, adj in enumerate(masked_adj_list):
            adj.data[adj.indptr[node]:adj.indptr[node + 1]] = 0
            adj = adj.tocsc()
            adj.data[adj.indptr[node]:adj.indptr[node + 1]] = 0
            adj = adj.tocsr()
            masked_adj_list[i] = adj
    for adj in masked_adj_list:
        adj.eliminate_zeros()
    return masked_adj_list


def separate_keys(id2entity):  
    go_keys = []  
    protein_keys = []  
      
    for key, value in id2entity.items():  
        if value.startswith("GO:"):  
            go_keys.append(key)  
        else:  
            protein_keys.append(key)  
              
    return go_keys, protein_keys

def remove_duplicate_lines(file1, file2, output_file):  
    with open(file1, 'r') as f1:  
        file1_lines = set(f1.readlines())  
  
    with open(file2, 'r') as f2:  
        file2_lines = set(f2.readlines())  

    duplicate_lines = file1_lines.intersection(file2_lines)  
  
    with open(output_file, 'w') as out:  
        for line in file2_lines:  
            if line not in duplicate_lines:  
                out.write(line) 

def double_check_subdataset(train_dir, test_dir):
    train_train = train_dir + "/train.txt"
    train_valid = train_dir + "/valid.txt"
    train_test = train_dir + "/test.txt"
    test_train = test_dir + "/train.txt"
    test_valid = test_dir + "/valid.txt"
    test_test = test_dir + "/test.txt"
    
    remove_duplicate_lines(train_train, test_train, test_train)
    remove_duplicate_lines(train_train, test_valid, test_valid)
    remove_duplicate_lines(train_train, test_test, test_test)
    
    remove_duplicate_lines(train_valid, test_train, test_train)
    remove_duplicate_lines(train_valid, test_valid, test_valid)
    remove_duplicate_lines(train_valid, test_test, test_test)
    
    remove_duplicate_lines(train_test, test_train, test_train)
    remove_duplicate_lines(train_test, test_valid, test_valid)
    remove_duplicate_lines(train_test, test_test, test_test)


def get_ancestors(go_list):
    original_go_set = set(go_list)
    obo_file = "/cluster/home/zzmmmmm/project/neo_kg/utils/go-basic.obo"
    go_dag = GODag(obo_file, optional_attrs=['relationship'])

    for go_id in tqdm(go_list):
        gosubdag_r0 = GoSubDag([go_id], go_dag, relationships=None, prt=None)
        if go_id in gosubdag_r0.rcntobj.go2ancestors.keys():
            ancestors = gosubdag_r0.rcntobj.go2ancestors[go_id]
            original_go_set.update(ancestors)

    return original_go_set


def main(params):
    print("start processing file")
    adj_list, triplets, entity2id, relation2id, id2entity, id2relation = process_files(files)
    go_list, protein_list = separate_keys(id2entity)


    print("start getting subgraph")
    meta_train_nodes = get_subgraph(adj_list, params.hops, params.max_nodes_per_hop, protein_list, id2entity, params.n_roots)  # list(range(750, 8500))  #
    train_prot_nodes = set(meta_train_nodes).intersection(set(protein_list))
    train_go_nodes = set(meta_train_nodes).intersection(set(go_list))
    train_go_nodes_name = [id2entity[key] for key in train_go_nodes if key in id2entity]
    masked_train_nodes_w_ancestors_name = get_ancestors(train_go_nodes_name)
    masked_train_nodes_w_ancestors = [entity2id[key] for key in masked_train_nodes_w_ancestors_name if key in entity2id]
    masked_nodes = set(meta_train_nodes).union(masked_train_nodes_w_ancestors)
    print('In train set, num nodes:', len(masked_nodes), 'protein nodes:', len(train_prot_nodes), 'GO nodes:', len(masked_train_nodes_w_ancestors))
    print("start masking nodes")
    masked_adj_list = mask_nodes(adj_list, masked_nodes, protein_list, go_list, params.mask_type)
    print("start getting test subgraph")
    if params.mask_type == 'protein':
        meta_test_nodes = get_subgraph(masked_adj_list, params.hops_test + 1, params.max_nodes_per_hop_test, protein_list, id2entity, params.n_roots)
    elif params.mask_type == 'GO':
        meta_test_nodes = get_SPUG_subgraph(masked_adj_list, params.hops_test + 1, params.max_nodes_per_hop_test, train_prot_nodes, params.n_roots)
    elif params.mask_type == 'none':
        meta_test_nodes = get_SPUG_subgraph(masked_adj_list, params.hops_test + 1, params.max_nodes_per_hop_test, train_prot_nodes, params.n_roots)
    else:
        raise ValueError("Invalid mask_type")
    print('In test set, protein nodes: ', len(set(meta_test_nodes).intersection(set(go_list))), 'GO nodes: ', len(set(meta_test_nodes).intersection(set(go_list))))

    print('Common GO nodes among the two disjoint datasets: ', set(meta_train_nodes).intersection(set(meta_test_nodes)).intersection(set(go_list)))
    print('Common protein nodes among the two disjoint datasets: ', set(meta_train_nodes).intersection(set(meta_test_nodes)).intersection(set(protein_list)))
    tmp = [adj[meta_train_nodes, :][:, meta_train_nodes] for adj in masked_adj_list]
    print('Residual edges (should be zero) : ', np.sum(get_edge_count(tmp)))

    print("================")
    print("Train graph stats")
    print("================")
    train_triplets, valid_triplets, test_triplets, train_active_rels = get_splits(adj_list, meta_train_nodes, range(len(adj_list)))
    print("================")
    print("Meta-test graph stats")
    print("================")
    meta_train_triplets, meta_valid_triplets, meta_test_triplets, meta_active_rels = get_splits(masked_adj_list, meta_test_nodes, train_active_rels)

    print("================")
    print('Extra rels (should be empty): ', set(meta_active_rels) - set(train_active_rels))

    data_dir = os.path.join(params.main_dir, 'data/{}'.format(params.new_dataset))
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    save_to_file(data_dir, 'train.txt', train_triplets, id2entity, id2relation)
    save_to_file(data_dir, 'valid.txt', valid_triplets, id2entity, id2relation)
    save_to_file(data_dir, 'test.txt', test_triplets, id2entity, id2relation)

    meta_data_dir = os.path.join(params.main_dir, 'data/{}'.format(params.new_dataset + '_meta'))
    if not os.path.exists(meta_data_dir):
        os.makedirs(meta_data_dir)

    save_to_file(meta_data_dir, 'train.txt', meta_train_triplets, id2entity, id2relation)
    save_to_file(meta_data_dir, 'valid.txt', meta_valid_triplets, id2entity, id2relation)
    save_to_file(meta_data_dir, 'test.txt', meta_test_triplets, id2entity, id2relation)


if __name__ == '__main__':
    start_time = time.time()

    parser = argparse.ArgumentParser(description='Save adjacency matrtices and triplets')

    parser.add_argument("--dataset", "-d", type=str, default="ProtKG",
                        help="Dataset string")
    parser.add_argument("--new_dataset", "-nd", type=str, default="fb_v3",
                        help="Dataset string")
    parser.add_argument("--n_roots", "-n", type=int, default="1",
                        help="Number of roots to sample the neighborhood from")
    parser.add_argument("--hops", "-H", type=int, default="2",
                        help="Number of hops to sample the neighborhood")
    parser.add_argument("--max_nodes_per_hop", "-m", type=int, default="300",
                        help="Number of nodes in the neighborhood")
    parser.add_argument("--hops_test", "-HT", type=int, default="2",
                        help="Number of hops to sample the neighborhood")
    parser.add_argument("--max_nodes_per_hop_test", "-mt", type=int, default="300",
                        help="Number of nodes in the neighborhood")
    parser.add_argument("--seed", "-s", type=int, default="1",
                        help="Numpy random seed")
    parser.add_argument("--mask_type", "-t", type=str, default="none",
                        help="protein/GO/both/none")
    parser.add_argument("--slink", "-sl", type=str, default="top3both",
                        help="soft link for the original file")

    params = parser.parse_args()

    np.random.seed(params.seed)
    random.seed(params.seed)

    params.main_dir = os.path.join(os.path.relpath(os.path.dirname(os.path.abspath(__file__))), '..')

    files = {
        # 'train': '/home/zhouzm/deepgozero/data/train_filtered_pg_rel_cc_top3.txt',
        'train': os.path.join(params.main_dir, f'data/{params.dataset}/{params.slink}'),
        'valid': os.path.join(params.main_dir, 'data/{}/valid'.format(params.dataset)),
        'test': os.path.join(params.main_dir, 'data/{}/test'.format(params.dataset))
    }

    main(params)

    end_time = time.time()
    print('Took %f second' % (end_time - start_time))