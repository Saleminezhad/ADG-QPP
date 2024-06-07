import faiss
from sentence_transformers import SentenceTransformer,  util
import torch
import time
import json


device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')

base_queries_filepath = "/home/abbas/ADG-QPP/msmarco-data/queries.train.tsv"
output_path = "/home/abbas/ADG-QPP/NNQ/output"


top_k = 100 + 1

base_queries_name = base_queries_filepath.split('/')[-1].split('.')[1]

model_name = "msmarco-MiniLM-L6-cos-v5"
model = SentenceTransformer(model_name)
embedding_dimension_size = model.get_sentence_embedding_dimension()
# embedding_dimension_size = 384

base_queries = {}
queries_list = []
with open(base_queries_filepath, 'r', encoding='utf-8') as fIn:
    for line in fIn:
        qid, query = line.strip().split("\t")
        base_queries[qid] = query
        queries_list.append(query)
        
print("Number of Queries to be indexed:", len(queries_list))


index = faiss.IndexFlatIP(embedding_dimension_size)

index= faiss.read_index("/home/abbas/ADG-QPP/msmarco-data/indices/train_all-MiniLM-L6-v2_queries.index")
print("index loaded")


for year in ["hard", "2019", "2020"]:
    target_queries_filepath = f"/home/abbas/ADG-QPP/msmarco-data/trec_data/{year}/{year}_queries"
    target_queries_name = target_queries_filepath.split('/')[-1]

    target_queries = {}
    queries_list2 = []
                
    with open(target_queries_filepath, 'r', encoding='utf-8') as fIn:
        for line in fIn:
            qid, query = line.strip().split("\t")
            target_queries[qid] = query
            queries_list2.append(query.strip())
    xq = model.encode(queries_list2)

    start_time = time.time()
    D, I = index.search(xq, top_k)



    print(f'Search time: {time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time))}')

    with open(output_path + f'/top_{top_k - 1}_{base_queries_name}-{target_queries_name}_{model_name.split("/")[-1]}_matched_queries.tsv', 'w', encoding='utf-8') as fOut:
        t_qids = list(target_queries.keys())
        b_qids = list(base_queries.keys())
        fOut.write(f'qid\tquery\tmatched_qid\tmatched_query\trank\n')

        for qid in range(len(I)):
            for rank in range(top_k - 1):
                fOut.write(f'{t_qids[qid]}\t{target_queries[t_qids[qid]]}\t{b_qids[I[qid][rank]]}\t{base_queries[b_qids[I[qid][rank]]]}\t{rank + 1}\n')

        
    qids=[]
    queries=[]
    with open(target_queries_filepath, 'r', encoding='utf8') as fIn:
        for line in fIn:
            qid, query = line.strip().split("\t")
            qids.append(qid)
            queries.append(query)
            
    NNQ_path = output_path + f'/top_{top_k - 1}_{base_queries_name}-{target_queries_name}_{model_name.split("/")[-1]}_matched_queries.tsv'

    NNQ_data = {}
    with open(NNQ_path, 'r', encoding='utf-8') as fIn:
        for line in fIn:
            qid, query, matched_qid, matched_query, rank = line.strip().split("\t")
            if qid not in NNQ_data:
                NNQ_data[qid] = {}
                
            NNQ_data[qid][rank] = matched_qid + "\t" + matched_query
            NNQ_data[qid]["q_text"] = query
            




    NNQ_data_emd_similarity = {}

    for key in NNQ_data.keys():

        if key != "qid":
            NNQ_data_emd_similarity[key] = {}
            NNQ_data_emd_similarity[key]['q_text'] = {}
            NNQ_data_emd_similarity[key]['q_text']['q_text'] = NNQ_data[key]['q_text']
            embedding_1= model.encode(NNQ_data[key]['q_text'], convert_to_tensor=True)
            NNQ_data_emd_similarity[key]['q_text']['emb'] = embedding_1.tolist()

            for rank in NNQ_data[key].keys():
                if rank != "q_text":
                    NNQ_data_emd_similarity[key][rank] = {}
                    NNQ_data_emd_similarity[key][rank]["id"] = NNQ_data[key][rank].split("\t")[0]
                    NNQ_data_emd_similarity[key][rank]["query"] = NNQ_data[key][rank].split("\t")[1]
                    q_gen = NNQ_data[key][rank].split("\t")[1]
                    embedding_2 = model.encode(q_gen, convert_to_tensor=True)
                    sem_sim = util.pytorch_cos_sim(embedding_1, embedding_2).item()
                    NNQ_data_emd_similarity[key][rank]["sim"] = round(sem_sim, 4)
                    NNQ_data_emd_similarity[key][rank]["emb"] = embedding_2.tolist()
                    
                    
    with open(f"/home/abbas/ADG-QPP/NNQ/output/top_100_emb_sim_train_{year}_queries_{model_name}_matched_queries.json", 'w') as json_file:
        json.dump(NNQ_data_emd_similarity, json_file)
