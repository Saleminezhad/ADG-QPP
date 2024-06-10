
from scipy.stats import pearsonr,kendalltau,spearmanr
from sentence_transformers import util
import networkx as nx
import numpy as np
from networkx.algorithms.community import greedy_modularity_communities
import numpy as np


# import torch
# from utils import RBO_func, correlation_eval, map_values
    
def correlation_eval_all_run(performance_path, total_run, year,var, K, epsilon, metric_name, network_size):
    noisy_dic = {}
    for n in range(total_run):
        # RBO_output_path = f'/home/abbas/Dense-QPP/NNQ/temp_test_query/{metric_name}_{network_size}_{epsilon}_{alpha}_{var}_{version_name}/{n}_{year}_{alpha}_{var}_{version_name}_rbo_noisy_run_fil_{K}'
        RBO_output_path = f'/RBO/{metric_name}_{network_size}_{epsilon}_{var}/{n}_{year}_{var}_rbo_noisy_run_fil_{K}'

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
    
def correlation_eval_doc_all_run(performance_path, total_run, year,var, K, epsilon, metric_name, network_size):
    
    noisy_dic = {}
    for n in range(total_run):
        RBO_output_path = f'/RBO/{metric_name}_{network_size}_{epsilon}_{var}/{n}_{year}_{var}_rbo_noisy_run_fil_{K}'

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


def retrieve_noisy_query_network_v1_emb(NNQ_data_emd_similarity, noisy_run_path, main_query_path, index, var, epsilon, metric_name, network_size, year):
    
    qids=[]
    queries=[]
    with open(main_query_path, 'r', encoding='utf8') as fIn:
        for line in fIn:
            qid, query = line.strip().split("\t")
            qids.append(qid)
            queries.append(query)
            
# ############
    main_query_emb_path = f'/msmarco-data/trec_data/{year}/{year}_queries_emb.npy'
    xq = np.load(main_query_emb_path)


    top_k = 1000
    network_measurement = []


    
    for qid in qids:
        # n 

        orig_emb = NNQ_data_emd_similarity[qid]['q_text']['emb']
        emblist = []
        emblist.append(orig_emb)
        for rank in range(1, network_size +1):
            emb = NNQ_data_emd_similarity[qid][str(rank)]["emb"]
            emblist.append(emb)
        adj_matrix = util.pytorch_cos_sim(emblist, emblist).tolist()

        G = nx.Graph()
        for i in range(len(adj_matrix)):
            for j in range(len(adj_matrix[i])):
                if adj_matrix[i][j] >= epsilon:
                    G.add_edge(str(i), str(j), weight=np.round(adj_matrix[i][j], 2))
                    
        for i in range(len(G.nodes())):
            G.nodes[str(i)]["vector"]=emblist[i]

        result = metric_main(metric_name, G)
        network_measurement.append(result)

    
    with open(noisy_run_path, 'w') as out:
        for i in range(len(xq)):
            noise_tensor =np.random.normal(0, var , xq.shape[1])
            xq[i, :] = xq[i, :] + network_measurement[i] * noise_tensor
    
        _, I = index.search(xq, top_k)  # search

        for q_id in range(len(I)):

            for rank in range(1,top_k +1):
                out.write(qids[q_id]+'\t'+str( I[q_id][rank-1])+'\t'+str(rank)+'\n')
                
                
#########################################################################################################################################################################
# proximity-based metrics
def SNS(G, node="0"):
    """Calculate the normalized Network Size (NS), which is the number of nodes in the graph 
    normalized by the maximum number of nodes if known, otherwise by itself."""
    total_nodes = len(G)
    max_nodes = len(G)  # assuming the current graph is at maximum node capacity
    return total_nodes / max_nodes if max_nodes > 0 else 0



# node-based metrics

def DC(G, node="0"):
    """Calculate Adaptive Degree Centrality (ADC) as the sum of edge weights, interpreted as disturbance-resistant connections."""
    return nx.degree_centrality(G)[node]


def CC(G, node="0"):
    """Calculate Adaptive Closeness Centrality (ACC), possibly weighted by disturbance effects on edges."""
    return nx.closeness_centrality(G, node, distance='weight')

def PR(G, node="0"):
    """Evaluate Adaptive PageRank (APR), which may inherently account for disturbances if edge weights are affected."""
    pagerank_dict = nx.pagerank(G, weight='weight')
    return pagerank_dict.get(node, 0)


# edge-based metrics

def CS(G,  node="0"):
    """Calculate normalized Adaptive Edge Count (AEC), normalized by the total possible number of edges."""
    total_nodes = len(G)
    total_possible_edges = total_nodes * (total_nodes - 1) / 2
    current_edge_count = G.number_of_edges()
    return current_edge_count / total_possible_edges if total_possible_edges > 0 else 0

def QCS(G,  node="0" ):
    max_weight=1
    """Calculate normalized Adaptive Edge Weight Sum (AEWS), normalized by the maximum possible sum of edge weights.
    
    Args:
    - G (nx.Graph): The graph for which AEWS is calculated.
    - max_weight (float): The maximum possible weight an edge can have, used for normalization.
    
    Returns:
    - float: The normalized AEWS.
    """
    max_possible_weight_sum = max_weight * len(G.edges())  # Assuming each edge could potentially have 'max_weight'
    current_weight_sum = sum(data['weight'] for _, _, data in G.edges(data=True))
    return current_weight_sum / max_possible_weight_sum if max_possible_weight_sum > 0 else 0

def AQC(G, node="0"):
    """Calculate Adaptive Edge Weight Average (AEWA), which is the average weight of all edges in the graph."""
    weights = [data['weight'] for u, v, data in G.edges(data=True)]
    return np.mean(weights) if weights else 0

# def MQC(G, node="0"):
#     """Identify Adaptive Edge Weight Max (AEWM), which is the maximum edge weight in the graph."""
#     weights = [data['weight'] for u, v, data in G.edges(data=True)]
#     return max(weights) if weights else 0

def RPI(G, node="0"):
    """Calculate Adaptive Inverse Edge Frequency (AIEF), focusing on the uniqueness and significance of connections post-disturbance."""
    unique_edges = {data['weight'] for _, data in G[node].items()}
    total_edges = sum(1 for _ in G.edges(data=True))
    return len(unique_edges) / total_edges if total_edges > 0 else 0


# cluster based metrics



def ICC(G, node = "0"):
    clusters = [list(community) for community in greedy_modularity_communities(G)]

    """Calculate the Inter-Cluster Connectivity for a given set of clusters in graph G.
    
    Args:
    - G (nx.Graph): The graph containing the clusters.
    - clusters (list of lists): List of clusters, where each cluster is a list of nodes.
    
    Returns:
    - float: The Inter-Cluster Connectivity value.
    """
    max_weights = []
    for i in range(len(clusters)):
        for j in range(i + 1, len(clusters)):
            max_weight = max([G[u][v]['weight'] for u in clusters[i] for v in clusters[j] if G.has_edge(u, v)], default=0)
            max_weights.append(max_weight)
    return np.mean(max_weights) if max_weights else 0

def CCW(G, node="0"):
    """Calculate Adaptive Edge Weight Average for the most semantically cohesive cluster (centroid Segment)."""
    clusters = get_clusters(G)
    centroid = centroid_cluster(G, clusters)
    if centroid:
        subgraph = G.subgraph(centroid)
        weights = [data['weight'] for u, v, data in subgraph.edges(data=True)]
        return np.mean(weights) if weights else 0
    return 1


def detect_clusters(G):
    """Detect clusters in the graph using the greedy modularity community detection method.
    
    Args:
    - G (nx.Graph): The graph for which to detect clusters.
    
    Returns:
    - dict: A dictionary with nodes as keys and their respective cluster indices as values.
    """
    communities = list(greedy_modularity_communities(G))
    label_dict = {}
    for idx, community in enumerate(communities):
        for node in community:
            label_dict[node] = idx
    return label_dict

def get_cluster_nodes(label_dict, cluster_index):
    """Get nodes belonging to a specific cluster index.
    
    Args:
    - label_dict (dict): Dictionary of node-cluster label pairs.
    - cluster_index (int): The index of the cluster for which nodes are required.
    
    Returns:
    - list: List of nodes belonging to the specified cluster.
    """
    return [node for node, index in label_dict.items() if index == cluster_index]


def metric_main(metric_function_name, G, node ="0"):
    """Execute the specified metric function on the graph G."""
    # Retrieve the function from globals
    func = globals().get(metric_function_name)
    
    # Check if the function exists and is callable
    if callable(func):
        return func(G, node)
    else:
        return "Function not found."

