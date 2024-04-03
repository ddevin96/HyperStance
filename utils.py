import os, shutil
import pickle
import pandas as pd
import hypernetx as hnx
import json

### get_embeddings("Hello, my dog is cute", tokenizer, model)
def get_embeddings(text, tokenizer, model):
    encoded_input = tokenizer(text, truncation=True, max_length=512, return_tensors='tf')
    output_embeddings = model(encoded_input)
    return output_embeddings.pooler_output

### generate file of adjacency matrix from hypergraph file
### hgToIncidenceMatrix("data/processed/dataset/hg.hgf", "data/processed/dataset/adj.pkl")
def hgToIncidenceMatrix(inputFile, outputFile):
    l = []
    with open(inputFile, 'r') as f:
        lines = f.readlines()
        for line in lines:
            l.append(set(map(int, line.split(','))))
    H = hnx.Hypergraph(l)
    with open(outputFile, 'wb') as f:
        pickle.dump(H.incidence_matrix().toarray(), f, pickle.HIGHEST_PROTOCOL)

### create files for the architecture
### * id_map.csv - mapping of unique id to original id
### * textembs.pkl - array of text embeddings
### * hg.hgf - hypergraph file
### * hg.edgelabel - edge labels | the label is the link_id
### * adjacency.pkl - adjacency matrix
def create_files(input_folder, output_folder, tokenizer, model):
    spinos = pd.read_pickle('data/spinos/SPINOS_official_dataset.pkl')
    spinos['index'] = spinos.index
    annotations = ['stance_not_inferrable', 'favor', 'undecided', 's_favor', 'against', 's_against'] # [0, 1, 2, 3, 4, 5] | -1 if not present

    for file in os.listdir(input_folder):
        file = file.split(".")[0]
        # ignoring submission - comments only
        if file.endswith("_comments"):
            filename = file.split("_comments")[0]
            
            out = output_folder + "/" + filename
            if not os.path.exists(out):
                print(f"\n\nCreating dataset for {filename}")
                os.makedirs(out)
            else:
                # if folder already exists, we don't want to overwrite it
                print(f"{filename} folder already exists - skipping dataset")
                continue

            data = pd.read_csv(input_folder + "/" + file + ".csv")

            # if dataset is empty, we don't want to create the files
            if data.empty:
                print(f"Empty dataset for {filename} - skipping dataset")
                continue

            data = data.dropna()
            data = data[data.author != "AutoModerator"]
            # assign unique id to each message and set label_id
            data['id_num'] = data['id'].astype('category').cat.codes
            data['label'] = data['id'].map(spinos['annotation']).fillna(-1).astype(str)
            data['label_id'] = data['label'].apply(lambda x: annotations.index(x) if x in annotations else -1)
            data = data.drop(columns=['label'])
            data[['id_num', 'id', 'label_id']].drop_duplicates().to_csv(out + '/id_map.csv', index=False)
            
            # array of text embeddings sorted by id_num
            text_embs = []
            sorted_data_by_id_num = data.sort_values(by='id_num')
            l = len(sorted_data_by_id_num)
            l_tenth = l // 10
            for index, row in sorted_data_by_id_num.iterrows():
                if index == 0 or index % l_tenth == 0:
                    print(f"Processing {index}/{l}")
                text_embs.append(get_embeddings(row['body'], tokenizer, model))
            with open(out + '/textembs.pkl', 'wb') as f:
                pickle.dump(text_embs, f, pickle.HIGHEST_PROTOCOL)

            # save hg and edge label (link_id) to file          
            with open(out + "/hg.hgf", 'w') as f:
                with open(out + '/hg.edgelabel', 'w') as f2:
                    for name, group in data.groupby('link_id'):
                        f2.write(str(name) + '\n')
                        if len(group['link_id'].values) > 1:
                            g = group['id_num'].unique()
                            g.sort()
                            f.write(','.join(map(str, g)))
                            f.write('\n')
                        else:
                            f.write(str(group['id_num'].values[0]) + '\n')

            # create adjacency matrix from hypergraph file
            hgToIncidenceMatrix(out + "/hg.hgf", out + "/matrix.pkl")
            print("-------------------\n")
    remove_empty_folders(output_folder)

### remove empty folders from a directory in a recursive way
def remove_empty_folders(path):
    if not os.path.isdir(path):
        return

    # remove empty subfolders
    files = os.listdir(path)
    if len(files):
        for f in files:
            fullpath = os.path.join(path, f)
            if os.path.isdir(fullpath):
                remove_empty_folders(fullpath)

    # if folder empty, delete it
    files = os.listdir(path)
    if len(files) == 0:
        print(f"Removing empty folder: {path}")        
        os.rmdir(path)

## change name of a file in each subfolder
def change_name(folder, old_name, new_name):
    for subfolder in os.listdir(folder):
        if os.path.isdir(folder + "/" + subfolder):
            for file in os.listdir(folder + "/" + subfolder):
                if file.endswith(old_name):
                    os.rename(folder + "/" + subfolder + "/" + file, folder + "/" + subfolder + new_name)
                    print(f"Renamed {file} to {new_name} in {subfolder}")    

### change id_map with associated labels from spinos
### if id is present in spinos, label_id is set to the index of the label in annotations from 0 to 5
### otherwise, label_id is set to -1 
def change_labels():
    data = pd.read_pickle('data/spinos/SPINOS_official_dataset.pkl')
    data['index'] = data.index
    annotations = ['stance_not_inferrable', 'favor', 'undecided', 's_favor', 'against', 's_against'] # [0, 1, 2, 3, 4, 5]
    for subfolder in os.listdir("data/processed"):
            if os.path.isdir("data/processed" + "/" + subfolder):
                for file in os.listdir("data/processed" + "/" + subfolder):
                    if file == "id_map.csv":
                        id_map = pd.read_csv('data/processed/' + subfolder + '/id_map.csv')
                        id_map['label'] = id_map['id'].map(data['annotation']).fillna(-1).astype(str)
                        id_map['label_id'] = id_map['label'].apply(lambda x: annotations.index(x) if x in annotations else -1)
                        id_map = id_map.drop(columns=['label'])
                        id_map.to_csv('data/processed/' + subfolder + '/id_map.csv', index=False)


### process MT_CSD data
def process_MT_CSD(input_folder):
    for subfolder in os.listdir(input_folder):
        if os.path.isdir(input_folder + "/" + subfolder):
            for file in os.listdir(input_folder + "/" + subfolder):
                if file == "text.csv":
                    data = pd.read_csv(input_folder + "/" + subfolder + "/text.csv")
                    output_folder = input_folder + "/" + subfolder + "_processed"
                    if not os.path.exists(output_folder):
                        os.makedirs(output_folder)
                    # read json file
                    with open(input_folder + "/" + subfolder + "/train.json") as f:
                        train_json = json.load(f)
                    with open(input_folder + "/" + subfolder + "/test.json") as f:
                        test_json = json.load(f)
                    with open(input_folder + "/" + subfolder + "/valid.json") as f:
                        valid_json = json.load(f)

                    output_conversation_folder = output_folder + "/conversation"
                    counter_conversation = 0
                    # if not os.path.exists(output_conversation_folder + str(counter_conversation)):
                    #     os.makedirs(output_conversation_folder + str(counter_conversation+1))
                    hm = {}
                    counter = 0
                    hgf_line = ""
                    hgf_he_starter = "1-1"
                    hgf_body = ""
                    for index, row in data.iterrows():
                        # if counter_conversation == '4':
                        #     exit()
                        # if row['id'] is convertable to int, it's first message in the conversation
                        if row['id'].isdigit():
                            if row['id'] != counter_conversation:
                                # new conversation
                                counter_conversation = row['id']
                                real_count = int(counter_conversation)-1
                                if not os.path.exists(output_conversation_folder + str(real_count)):
                                    os.makedirs(output_conversation_folder + str(real_count))
                                counter = 0
                                hm = {}
                                hgf_line = hgf_line[:-1]
                                # print(hgf_line)
                                hgf_body += hgf_line + "\n"
                                with open(output_conversation_folder + str(real_count) + "/hg.hgf", 'w') as f:
                                    # strip empty lines
                                    hgf_body = os.linesep.join([s for s in hgf_body.splitlines() if s])
                                    f.write(hgf_body)
                                if row['id'] not in hm:
                                    hm[row['id']] = counter
                                    counter += 1
                                hgf_line = "" + str(hm[str(counter_conversation)]) + ","
                                print(f"Conversation {counter_conversation}")
                                continue

                                # row['text] do something

                        elif len(row['id'].split('-')) == 2:
                            if row['id'] != hgf_he_starter:
                                if hgf_line != "" + str(hm[str(counter_conversation)]) + ",":
                                    hgf_line = hgf_line[:-1]
                                    hgf_body += hgf_line + "\n"
                                    # print(hgf_line)
                                    hgf_he_starter = row['id']
                                    hgf_line = "" + str(hm[str(counter_conversation)]) + ","

                        if row['id'] not in hm:
                            hm[row['id']] = counter
                            counter += 1
                        
                        hgf_line += str(hm[row['id']]) + ","

                    if os.path.isdir(output_conversation_folder + "0"):
                        shutil.rmtree(output_conversation_folder + "0")
  