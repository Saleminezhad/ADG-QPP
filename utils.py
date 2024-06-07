import math
import os , numpy
from scipy.stats import pearsonr,kendalltau,spearmanr
import collections
import  faiss
from sentence_transformers import SentenceTransformer, util
import networkx as nx

# from sentence_transformers import SentenceTransformer, LoggingHandler
import logging
import os
import torch
from shutil import copyfile
import numpy as np
import json
import collections
import math
import rbo
# import torch
# from utils import RBO_func, correlation_eval, map_values

def rbo_function(list1, list2, p=0.9):
    # tail recursive helper function
    def helper(ret, i, d):
        l1 = set(list1[:i]) if i < len(list1) else set(list1)
        l2 = set(list2[:i]) if i < len(list2) else set(list2)
        a_d = len(l1.intersection(l2))/i
        term = math.pow(p, i) * a_d
        if d == i:
            return ret + term
        return helper(ret + term, i + 1, d)
    k = max(len(list1), len(list2))
    x_k = len(set(list1).intersection(set(list2)))
    summation = helper(0, 1, k)
    return ((float(x_k)/k) * math.pow(p, k)) + ((1-p)/p * summation)

def rbo_function2(list1, list2, p=0.9):
    k = max(len(list1), len(list2))
    x_k = len(set(list1).intersection(set(list2)))
    summation = 0
    for i in range(1, k+1):
        l1 = set(list1[:i]) if i <= len(list1) else set(list1)
        l2 = set(list2[:i]) if i <= len(list2) else set(list2)
        a_d = len(l1.intersection(l2)) / i
        term = math.pow(p, i) * a_d
        summation += term
    return ((float(x_k)/k) * math.pow(p, k)) + ((1-p)/p * summation)


def correlation_eval(performance_path, RBO_path):
    performance_file=open(performance_path,'r').readlines()
    performance_dic={}
    for line in performance_file:
        qid,performance=line.rstrip().split()
        performance_dic[qid]=float(performance)


    qpp_file=open(RBO_path,'r').readlines()

    qpp_dic={}
    for line in qpp_file:
        qid,qpp_metrics=line.rstrip().split()
        qpp_metrics=float(qpp_metrics)
        qpp_dic[qid] = qpp_metrics
    actual=[]
    predicted=[]
    for key, value in performance_dic.items():
        predicted.append(qpp_dic[key])
        actual.append(value)

    return pearsonr(actual,predicted)[0],kendalltau(actual,predicted)[0],spearmanr(actual,predicted)[0] 
    
    
def correlation_eval_all_run(performance_path, total_run, year,version_name,epsilon, alpha,var, K):
    noisy_dic = {}
    for n in range(total_run):
        RBO_output_path = f'/home/abbas/Dense-QPP/NNQ/temp_test/{epsilon}_{alpha}_{var}_{version_name}/{n}_{epsilon}_{year}_{alpha}_{var}_{version_name}_rbo_noisy_run_fil_{K}'

        noisy_run=open(RBO_output_path,'r').readlines()
        
        for line in noisy_run:
            qid, score=line.rstrip().split('\t')
            if qid not in noisy_dic:
                noisy_dic[qid] = float(score)
            else:
                noisy_dic[qid] =  noisy_dic[qid] + float(score) 
            

    for i in noisy_dic:
        noisy_dic[i] = noisy_dic[i]/(total_run)
    
    performance_file=open(performance_path,'r').readlines()
    performance_dic={}
    for line in performance_file:
        qid,performance=line.rstrip().split()
        performance_dic[qid]=float(performance)

    actual=[]
    predicted=[]
    for key, value in performance_dic.items():
        predicted.append(noisy_dic[key])
        actual.append(value)

    return pearsonr(actual,predicted)[0],kendalltau(actual,predicted)[0],spearmanr(actual,predicted)[0] 
    

def RBO_func(orig_run_path, noisy_runf_path, RBO_output_path, year, p, K):
    
    original_run=open(f'/home/abbas/Dense-QPP/run/org_{year}.tsv','r').readlines()
    noisy_run=open(noisy_runf_path,'r').readlines()

    # noisy_run=open('/home/abbas/Dense-QPP/run/marco_unifiednoise_0.05','r').readlines()


    noisy_dic=collections.defaultdict(list)
    for line in noisy_run:
        qid,docid,rank=line.rstrip().split('\t')
        noisy_dic[qid].append(docid)

    org_dic=collections.defaultdict(list)
    for line in original_run:
        qid,docid,rank = line.rstrip().split('\t')[:3]
        org_dic[qid].append(docid)


    rank_org_ar=[]
    rank_noise_ar=[]

    with open(RBO_output_path,'w') as out:
        for qid in org_dic:
            rank_org_ar=[]
            rank_noise_ar=[]
            for doc in org_dic[qid][:K]:
                rank_org=org_dic[qid].index(doc)
                try:
                    rank_noise=noisy_dic[qid].index(doc)
                    rank_noise_ar.append(int(rank_noise))
                except:
                    pass
                rank_org_ar.append(int(rank_org))
            if year == "dev": 
                sm=rbo.RankingSimilarity(org_dic[qid][:K], noisy_dic[qid][:K]).rbo(p=1)
            if year in ["2019", "2020", "hard"]:
                sm = rbo_function2(org_dic[qid][:K], noisy_dic[qid][:K], p)

            out.write(qid+'\t'+str(sm)+'\n')

def map_values(value):
    if value <= -0.2663605:
        return 0
    elif value >= 0.26297867:
        return 0.05
    else:
        # Perform linear interpolation between 0 and 0.05
        slope = 0.05 / (0.26297867 - (-0.2663605))
        intercept = 0
        return slope * (value - (-0.2663605)) + intercept
    


def retrieve_noisy_doc_v2(NNQ_doc_path, noisy_run_path, main_query_path, index, model, alpha):
    
    qids=[]
    queries=[]
    with open(main_query_path, 'r', encoding='utf8') as fIn:
        for line in fIn:
            qid, query = line.strip().split("\t")
            qids.append(qid)
            queries.append(query)



    xq = model.encode(queries)
    top_k = 1000
    
    with open(NNQ_doc_path, 'r', encoding='utf-8') as fIn:
        org_doc_id_text = json.load(fIn)

    noisy_std_coeffs = []
    noisy_avg_emb = []
    for qid in qids:
        # n
        doc_id_list = org_doc_id_text[qid]['doc_ids']
        doc_text_list = org_doc_id_text[qid]['doc_texts']

        embeddings = model.encode(doc_text_list)

        # Stack all embeddings into a numpy array
        stacked_arrays = np.stack(embeddings)

        # Calculate the standard deviation
        std_dev = np.std(stacked_arrays, axis=0)
        noisy_std_coeffs.append(std_dev)

        average_emb = np.mean(embeddings,axis=0)

        map_values_vectorized = np.vectorize(map_values)
        normalized_emb = map_values_vectorized(average_emb)

        noisy_avg_emb.append(normalized_emb)    



    with open(noisy_run_path, 'w') as out:
        for i in range(len(xq)):
            noise_tensor =np.random.normal(noisy_avg_emb[i], noisy_std_coeffs[i] , embeddings.shape[1])                    
            xq[i, :] = xq[i, :] + alpha * noise_tensor 
    
        _, I = index.search(xq, top_k)  # search

        for q_id in range(len(I)):

            for rank in range(1,top_k +1):
                out.write(qids[q_id]+'\t'+str( I[q_id][rank-1])+'\t'+str(rank)+'\n')
                

def retrieve_noisy_doc_v2_load_emb(NNQ_doc_path, noisy_run_path, main_query_path, index, model, alpha, var):
    
    qids=[]
    queries=[]
    with open(main_query_path, 'r', encoding='utf8') as fIn:
        for line in fIn:
            qid, query = line.strip().split("\t")
            qids.append(qid)
            queries.append(query)


    xq = model.encode(queries)
    top_k = 1000
    
    with open(NNQ_doc_path, 'r', encoding='utf-8') as fIn:
        org_doc_id_text = json.load(fIn)
        

    
    noisy_std_coeffs = []
    noisy_avg_emb = []
    for qid in qids:
        # n
        doc_id_list = org_doc_id_text[qid]['doc_ids']
        doc_text_list = org_doc_id_text[qid]['doc_texts']
        embeddings = np.array(org_doc_id_text[qid]['doc_emb'])

        # embeddings = model.encode(doc_text_list)

        # Stack all embeddings into a numpy array
        stacked_arrays = np.stack(embeddings)

        # Calculate the standard deviation
        std_dev = np.std(stacked_arrays, axis=0)
        noisy_std_coeffs.append(std_dev)

        average_emb = np.mean(embeddings,axis=0)

        map_values_vectorized = np.vectorize(map_values)
        normalized_emb = map_values_vectorized(average_emb)

        noisy_avg_emb.append(normalized_emb)    



    with open(noisy_run_path, 'w') as out:
        for i in range(len(xq)):
            noise_tensor =np.random.normal(noisy_avg_emb[i], noisy_std_coeffs[i] , embeddings.shape[1])                    
            xq[i, :] = xq[i, :] + alpha * noise_tensor 
    
        _, I = index.search(xq, top_k)  # search

        for q_id in range(len(I)):

            for rank in range(1,top_k +1):
                out.write(qids[q_id]+'\t'+str( I[q_id][rank-1])+'\t'+str(rank)+'\n')
                
                

def retrieve_noisy_doc_v1_load_emb(NNQ_doc_path, noisy_run_path, main_query_path, index, model, alpha, var):
    
    qids=[]
    queries=[]
    with open(main_query_path, 'r', encoding='utf8') as fIn:
        for line in fIn:
            qid, query = line.strip().split("\t")
            qids.append(qid)
            queries.append(query)


    xq = model.encode(queries)
    
    top_k = 1000
    
    with open(NNQ_doc_path, 'r', encoding='utf-8') as fIn:
        org_doc_id_text = json.load(fIn)
        

    
    noisy_std_coeffs = []
    noisy_avg_emb = []
    for qid in qids:
        # n
        doc_id_list = org_doc_id_text[qid]['doc_ids']
        doc_text_list = org_doc_id_text[qid]['doc_texts']
        embeddings = np.array(org_doc_id_text[qid]['doc_emb'])

        # embeddings = model.encode(doc_text_list)

        # Stack all embeddings into a numpy array
        stacked_arrays = np.stack(embeddings)

        # Calculate the standard deviation
        std_dev = np.std(stacked_arrays, axis=0)
        noisy_std_coeffs.append(std_dev)

        average_emb = np.mean(embeddings,axis=0)

        map_values_vectorized = np.vectorize(map_values)
        normalized_emb = map_values_vectorized(average_emb)

        noisy_avg_emb.append(normalized_emb)    



    with open(noisy_run_path, 'w') as out:
        for i in range(len(xq)):
            noise_tensor =np.random.normal(0, alpha*noisy_std_coeffs[i] , xq.shape[1])
            xq[i, :] = xq[i, :] +  noise_tensor
    
        _, I = index.search(xq, top_k)  # search

        for q_id in range(len(I)):

            for rank in range(1,top_k +1):
                out.write(qids[q_id]+'\t'+str( I[q_id][rank-1])+'\t'+str(rank)+'\n')




def retrieve_noisy_doc_v3_load_emb(NNQ_doc_path, noisy_run_path, main_query_path, index, model, alpha, var):
    
    qids=[]
    queries=[]
    with open(main_query_path, 'r', encoding='utf8') as fIn:
        for line in fIn:
            qid, query = line.strip().split("\t")
            qids.append(qid)
            queries.append(query)


    xq = model.encode(queries)
    top_k = 1000
    
    with open(NNQ_doc_path, 'r', encoding='utf-8') as fIn:
        org_doc_id_text = json.load(fIn)
        

    
    noisy_std_coeffs = []
    noisy_avg_emb = []
    for qid in qids:
        # n
        doc_id_list = org_doc_id_text[qid]['doc_ids']
        doc_text_list = org_doc_id_text[qid]['doc_texts']
        embeddings = np.array(org_doc_id_text[qid]['doc_emb'])

        # embeddings = model.encode(doc_text_list)

        # Stack all embeddings into a numpy array
        stacked_arrays = np.stack(embeddings)

        # Calculate the standard deviation
        std_dev = np.std(stacked_arrays, axis=0)
        noisy_std_coeffs.append(std_dev)

        average_emb = np.mean(embeddings,axis=0)

        map_values_vectorized = np.vectorize(map_values)
        normalized_emb = map_values_vectorized(average_emb)

        noisy_avg_emb.append(normalized_emb)    



    with open(noisy_run_path, 'w') as out:
        for i in range(len(xq)):
            noise_tensor =np.random.normal(0, var , 384)
            xq[i, :] = xq[i, :] + noisy_std_coeffs[i] * noise_tensor
    
        _, I = index.search(xq, top_k)  # search

        for q_id in range(len(I)):

            for rank in range(1,top_k +1):
                out.write(qids[q_id]+'\t'+str( I[q_id][rank-1])+'\t'+str(rank)+'\n')
                
                
                

def retrieve_noisy_doc_v3_network(org_doc_id_text, noisy_run_path, main_query_path, index, alpha, var, epsilon, metric_name, network_size, year):
    
    
    qids=[]
    queries=[]
    with open(main_query_path, 'r', encoding='utf8') as fIn:
        for line in fIn:
            qid, query = line.strip().split("\t")
            qids.append(qid)
            queries.append(query)


    # xq = model.encode(queries)
    top_k = 1000
    main_query_emb_path = f'/data3/suren/Dense-QPP/test_data/trec_data/{year}/{year}_queries_emb.npy'
    xq = np.load(main_query_emb_path)
    # with open(NNQ_doc_path, 'r', encoding='utf-8') as fIn:
    #     org_doc_id_text = json.load(fIn)
        

    network_measurement = []
    noisy_std_coeffs = []
    noisy_avg_emb = []
    for ii, qid in enumerate(qids):
        # n
        doc_id_list = org_doc_id_text[qid]['doc_ids']
        doc_text_list = org_doc_id_text[qid]['doc_texts']
        embeddings = np.array(org_doc_id_text[qid]['doc_emb'])
        
        orig_emb = xq[ii]
        
        emblist = []
        emblist.append(orig_emb)
        for rank in range(1,network_size +1):
            emblist.append(embeddings[rank-1])
        emblist = np.array(emblist)
        adj_matrix = util.pytorch_cos_sim(emblist, emblist).tolist()
        
        G = nx.Graph()
        for i in range(len(adj_matrix)):
            for j in range(len(adj_matrix[i])):
                if adj_matrix[i][j] >= epsilon:
                    G.add_edge(str(i), str(j), weight=np.round(adj_matrix[i][j], 2))
                    
        for i in range(len(G.nodes())):
            G.nodes[str(i)]["vector"]=emblist[i]
            
        # degree_centrality_l = nx.degree_centrality(G)
        # network_measurement.append(degree_centrality_l[0])
        result = metric_main(metric_name, G)
        network_measurement.append(result)
        
        # # Stack all embeddings into a numpy array
        # stacked_arrays = np.stack(embeddings)

        # # Calculate the standard deviationß
        # std_dev = np.std(stacked_arrays, axis=0)
        # noisy_std_coeffs.append(std_dev)

        # average_emb = np.mean(embeddings,axis=0)

        # map_values_vectorized = np.vectorize(map_values)
        # normalized_emb = map_values_vectorized(average_emb)

        # noisy_avg_emb.append(normalized_emb)    



    with open(noisy_run_path, 'w') as out:
        for i in range(len(xq)):
            noise_tensor =np.random.normal(0, var ,  xq.shape[1])
            xq[i, :] = xq[i, :] + network_measurement[i] * noise_tensor
    
        _, I = index.search(xq, top_k)  # search

        for q_id in range(len(I)):

            for rank in range(1,top_k +1):
                out.write(qids[q_id]+'\t'+str( I[q_id][rank-1])+'\t'+str(rank)+'\n')