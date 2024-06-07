import faiss
import os
import json, time
import torch
from utils import RBO_func, correlation_eval
from utils_network import retrieve_noisy_query_network_v1_emb, correlation_eval_all_run
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

def main(var, K, p, total_run, epsilon, metric_name, network_size, model_name):
    

    data_folder='/home/abbas/ADG-QPP/msmarco-data/indices/'
    index= faiss.read_index(os.path.join(data_folder,model_name))
    
    correlation_list = ""
    years  = [ "hard", "2019", "2020"]

    for year in years:
        
        main_query_path = f'/home/abbas/ADG-QPP/msmarco-data/trec_data/{year}/{year}_queries'

        NNQ_query_path_emb =f"/home/abbas/ADG-QPP/NNQ/output/top_100_emb_sim_train_{year}_queries_msmarco-MiniLM-L6-cos-v5_matched_queries.json"
        with open(NNQ_query_path_emb, 'r') as json_file:
            NNQ_data_emd_similarity = json.load(json_file)
        
        
        noisy_run_path = f'/home/abbas/ADG-QPP/run/{metric_name}_{network_size}_{epsilon}_{var}/{metric_name}_{year}_{var}_Noisyq_run.tsv'
        
        # write a code to check if a folder exits or not, if not create it.
        folder_path = f'/home/abbas/ADG-QPP/run/{metric_name}_{network_size}_{epsilon}_{var}'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        
        if not os.path.exists(f'/home/abbas/ADG-QPP/RBO/{metric_name}_{network_size}_{epsilon}_{var}'):
            os.makedirs(f'/home/abbas/ADG-QPP/RBO/{metric_name}_{network_size}_{epsilon}_{var}')
            
        
        start_time = time.time()

        for n in range(total_run):

            retrieve_noisy_query_network_v1_emb(NNQ_data_emd_similarity, noisy_run_path, main_query_path, index, var, epsilon, metric_name, network_size, year)


            RBO_output_path = f'/home/abbas/ADG-QPP/RBO/{metric_name}_{network_size}_{epsilon}_{var}/{n}_{year}_{var}_rbo_noisy_run_fil_{K}'

            orig_run_path = f'/home/abbas/ADG-QPP/run/org_{year}.tsv'

            RBO_func(orig_run_path, noisy_run_path, RBO_output_path, year, p, K )
            
            performance_path = f'/home/abbas/ADG-QPP/performance/ndcg_cut_10_dl{year}.tsv'
            pears, kendal, spear = correlation_eval(performance_path, RBO_output_path)

            correlation_list += f'{n} {year} {pears} {kendal} {spear} \n'
            
        pears, kendal, spear = correlation_eval_all_run(performance_path, total_run, year,var, K, epsilon, metric_name, network_size)
        correlation_list += f'total_run {year} {pears} {kendal} {spear} \n'
        
        end_time1 = time.time()
        print(f"{metric_name} {year}  :", end_time1 - start_time)
    
    if not os.path.exists(f"/home/abbas/ADG-QPP/correlations/{metric_name}"):
        os.makedirs(f"/home/abbas/ADG-QPP/correlations/{metric_name}")

    with open(f"/home/abbas/ADG-QPP/correlations/{metric_name}/{metric_name}_{network_size}_{epsilon}_{var}.txt", 'w') as out:
        out.write(correlation_list)
            
            

if __name__ == "__main__":

    # Check if GPU is available
    if torch.cuda.is_available():
        print("CUDA GPU is available and being used.")
    else:
        print("CUDA GPU is not available. CPU is being used.")
        
    metrics = [
        "SNS",
        "PR", "CC","DC",
        "CS", "QCS", "AQC","RPI",
        "ICC", "CCW"
    ]
    
    var = 0.06
    epsilon = 0.3
    K=1000
    p = 0.95
    model_name =  "msmarco-MiniLM-L6-cos-v5"

    network_size = 10 
    total_run = 10
    for metric_name in metrics:
        main(var, K, p, total_run, epsilon, metric_name, network_size, model_name)

                
            