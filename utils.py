import os
import pickle
import pandas as pd

### get_embeddings("Hello, my dog is cute", tokenizer, model)
def get_embeddings(text, tokenizer, model):
    encoded_input = tokenizer(text, truncation=True, max_length=512, return_tensors='tf')
    output_embeddings = model(encoded_input)
    return output_embeddings.pooler_output

### create 
### * id_map.csv - mapping of unique id to original id
### * textembs.pkl - array of text embeddings
### * hg.hgf - hypergraph file
### * hg.edgelabel - edge labels | the label is the link_id
def create_files(input_folder, output_folder, tokenizer, model):
    for file in os.listdir(input_folder):
        file = file.split(".")[0]
        # ignoring submission - comments only
        if file.endswith("_comments"):
            filename = file.split("_comments")[0]
            
            out = output_folder + "/" + filename
            if not os.path.exists(out):
                print(f"Creating dataset for {filename}")
                os.makedirs(out)
            else:
                # if folder already exists, we don't want to overwrite it
                print(f"{filename} folder already exists - skipping dataset")
                continue

            data = pd.read_csv(input_folder + "/" + file + ".csv")
            data = data.dropna()
            data = data[data.author != "AutoModerator"]
            # assign unique id to each message
            data['id_num'] = data['id'].astype('category').cat.codes
            data[['id_num', 'id']].drop_duplicates().to_csv(out + '/id_map.csv', index=False)
            
            # array of text embeddings sorted by id_num
            text_embs = []
            sorted_data_by_id_num = data.sort_values(by='id_num')
            l = len(sorted_data_by_id_num)
            for index, row in sorted_data_by_id_num.iterrows():
                if index == 0 or index % 10 == 0:
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
            break