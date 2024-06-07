First we should build indice using the msmarco datasets. first index is for the collection.tsv that is for dense retriever to retrieve documents. 
the other index is used for the fining the nearest queries using the msmarco-data2/queries.train.tsv 
put the indices in /msmarco-data/indices

for finding the nearest queries use the find_topK_nearest_query.py to find and generate the nearest queries file in NNQ/output folder

after that use the query_main.py to run the total model and the outputs will be stored in correlations folder

