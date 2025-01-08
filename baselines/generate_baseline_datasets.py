import argparse
import os
import pickle
import pandas as pd
from tqdm import tqdm
import logging
import torch
from transformers import AutoTokenizer, AutoModel

# Ignore warnings
logging.getLogger().setLevel(logging.ERROR)
warnings.filterwarnings('ignore')

# Argument parser
parser = argparse.ArgumentParser(description='Script description')
parser.add_argument('--dataset', dest='dataset', type=str, required=True, help='The dataset name')
parser.add_argument('--ont', dest='ont', type=str, required=True, help='The ontology name')
parser.add_argument('--device', '-d', default='cuda:3', help='Device')
args = parser.parse_args()
DATASET = args.dataset
ONT = args.ont
device = args.device

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(f"../esm")
model = AutoModel.from_pretrained(f"../esm").to(device)

# Function to select protein-GO relations
def select_protein_go_relations(all_rels_file, PG_rels_file):
    with open(all_rels_file, 'r') as input_file, open(PG_rels_file, 'w') as output_file:
        output_file.write("protein\trel\tGO\n")
        for line in input_file:
            line = line.strip()
            elements = line.split('\t')
            if not elements[0].startswith("GO:") and elements[2].startswith("GO:"):
                output_file.write(line + '\n')

# Function to generate dictionary from protein-GO relations
def generate_dict(input_file, output_type):
    ProtKG = {}
    with open(input_file, "r") as f:
        for line in tqdm(f):
            row = line.strip().split("\t")
            protein = row[0]
            go = row[2]
            if not protein.startswith("GO:") and go.startswith("GO:"):
                if protein in ProtKG:
                    ProtKG[protein].append(go)
                else:
                    ProtKG[protein] = [go]
    with open(f"../data/{DATASET}/{output_type}_dict.pkl", "wb") as file:
        pickle.dump(ProtKG, file)
    return ProtKG

# Function to get result dataframe
def get_result_dataframe(protein_lst, data_file, ProtKG):
    results = []

    accessions_set = set(data_file['accessions'])  # Create a set of all accessions
    protein_data = {}
    with open('../data/protein_name_2_seq.txt', 'r') as file:
        lines = file.readlines()
        for i in range(0, len(lines), 2):
            protein_id = lines[i].strip()
            protein_seq = lines[i+1].strip()
            protein_data[protein_id] = protein_seq

    for protein in tqdm(protein_lst):
        mask = data_file['accessions'].apply(lambda x: protein in x) 
        filtered_rows = data_file[mask]

        if not filtered_rows.empty:
            interpros = filtered_rows.iloc[0]['interpros']
            prop_annos = filtered_rows.iloc[0]['prop_annotations']
            sequences = filtered_rows.iloc[0]['sequences']
            results.append({"Protein": protein, "interpros": interpros, "prop_annotations": prop_annos, "sequences": sequences})
        else:
            sequences = protein_data[protein]
            interpros = []
            prop_annos = ProtKG[protein]
            results.append({"Protein": protein, "interpros": interpros, "prop_annotations": prop_annos, "sequences": sequences})

    results_df = pd.DataFrame(results)
    return results_df

# Function to do GO annotations and properties
def do_go_annos_prop(ProtKG_dict, ancestor_file):
    ancestor_data = {}
    with open(ancestor_file, "r") as f:
        for line in tqdm(f):
            elements = line.strip().split(" ")
            key = elements[0]
            value = elements[2]
            if key in ancestor_data:
                ancestor_data[key].append(value)
            else:
                ancestor_data[key] = [value]

    go_set = set(value for values_list in ProtKG_dict.values() for value in values_list)
    for key, value_list in ProtKG_dict.items():
        updated_values = set(value_list)  
        
        for value in value_list:
            if value in ancestor_data:
                ancestor_values = ancestor_data[value]
                ancestor_in_go_set = list(set(ancestor_values).intersection(go_set))
                updated_values.update(ancestor_in_go_set)    
        
        ProtKG_dict[key] = list(updated_values) 

    return ProtKG_dict

# Function to generate embedding
def generate_embedding(protein_id_list, protein_seq_list):
    emb_list = []
    list_lengths = []
    sequences = protein_seq_list

    logging.info("start to generate train embedding...")
    for id, seq in tqdm(zip(protein_id_list, sequences)):
        sequence_example = seq
        sequence_example = sequence_example[:640]
        tokens = tokenizer.tokenize(sequence_example)
        inputs = tokenizer(sequence_example, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        outputs = model(**inputs)
        emb = outputs['last_hidden_state'][:, 0][0].detach().cpu().numpy()
        emb_list.append(emb)
        prot_id = id
    return emb_list

# Main execution
if __name__ == "__main__":
    output_dir = f'../data/{ONT}/{DATASET}'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Select protein-GO relations for training, validation, and testing
    select_protein_go_relations(f"../data/{DATASET}/train.txt", f"../data/{DATASET}/trainprot.txt")
    select_protein_go_relations(f"../data/{DATASET}/valid.txt", f"../data/{DATASET}/validprot.txt")
    select_protein_go_relations(f"../data/{DATASET}/test.txt", f"../data/{DATASET}/testprot.txt")

    # Load data and generate dictionaries
    with open("../data/swissprot_exp.pkl", "rb") as f:
        data_file = pickle.load(f)

    pg_dict_train_dict = generate_dict(PG_rels_file_train, 'train')
    pg_dict_valid_dict = generate_dict(PG_rels_file_valid, 'valid')
    pg_dict_test_dict = generate_dict(PG_rels_file_test_test, 'test')

    # Perform GO annotations and properties
    ProtKG_dict_w_ancs_train = do_go_annos_prop(pg_dict_train_dict, ancestor_file)
    ProtKG_dict_w_ancs_valid = do_go_annos_prop(pg_dict_valid_dict, ancestor_file)
    ProtKG_dict_w_ancs_test = do_go_annos_prop(pg_dict_test_dict, ancestor_file)

    # Generate result dataframes and embeddings
    train_result_df = get_result_dataframe(train_protein_lst, data_file, ProtKG_dict_w_ancs_train)
    valid_result_df = get_result_dataframe(valid_protein_lst, data_file, ProtKG_dict_w_ancs_valid)
    test_result_df = get_result_dataframe(test_protein_lst, data_file, ProtKG_dict_w_ancs_test)

    # Save result dataframes and embeddings
    with open(f'{output_dir}/train_data_{DATASET}.pkl', 'wb') as f:
        pickle.dump(train_result_df, f)

    with open(f'{output_dir}/valid_data_{DATASET}.pkl', 'wb') as f:
        pickle.dump(valid_result_df, f)

    with open(f'{output_dir}/test_data_{DATASET}.pkl', 'wb') as f:
        pickle.dump(test_result_df, f)

    # Generate and save embeddings
    train_emb_file = f'{output_dir}/train_emb_{DATASET}.pkl'
    train_esm_emb = generate_embedding(train_result_df['Protein'], train_result_df['sequences'])
    with open(train_emb_file, 'wb') as f:
        pickle.dump(train_esm_emb, f)
        train_result_df['esm2'] = train_esm_emb

    valid_emb_file = f'{output_dir}/valid_emb_{DATASET}.pkl'
    valid_esm_emb = generate_embedding(valid_result_df['Protein'], valid_result_df['sequences'])
    with open(valid_emb_file, 'wb') as f:
        pickle.dump(valid_esm_emb, f)
        valid_result_df['esm2'] = valid_esm_emb

    test_emb_file = f'{output_dir}/test_emb_{DATASET}.pkl'
    test_esm_emb = generate_embedding(test_result_df['Protein'], test_result_df['sequences'])
    with open(test_emb_file, 'wb') as f:
        pickle.dump(test_esm_emb, f)
        test_result_df['esm2'] = test_esm_emb