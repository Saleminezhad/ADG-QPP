
# ADG-QPP: Robust Query Performance Prediction for Dense Retrievers via Adaptive Disturbance Generation

This repository contains the implementation of ADG-QPP, a Query Performance Prediction (QPP) method designed specifically for dense neural retrievers. The underlying foundation of ADG-QPP is to measure query performance based on its degree of robustness towards perturbations.

## Table of Contents
- [Introduction](#introduction)
- [Prerequisites](#prerequisites)
- [Data Preparation](#data-preparation)
- [Building Indices](#building-indices)
- [Finding Nearest Queries](#finding-nearest-queries)
- [Running the Model](#running-the-model)
- [Results](#results)
- [References](#references)
- [Contact](#contact)

## Introduction

ADG-QPP is designed to address the limitations of traditional QPP methods by perturbing the query by injecting disturbance into its neural embedding representation. Through extensive experiments, we demonstrate that ADG-QPP outperforms state-of-the-art baselines in terms of Kendall τ, Spearman ρ, and Pearson’s ρ correlations.

## Prerequisites

- Python 3.8 or higher
- PyTorch
- Faiss
- NumPy
- Pandas
- Scikit-learn

You can install the required packages using the following command:

```bash
pip install -r requirements.txt
```

## Data Preparation

First, download the MS MARCO datasets for the collections and queries.

```bash
# Download collection.tsv and queries.train.tsv from MS MARCO
wget -P msmarco-data/ https://msmarco.blob.core.windows.net/msmarcoranking/collection.tar.gz
wget -P msmarco-data/ https://msmarco.blob.core.windows.net/msmarcoranking/queries.train.tsv
```

Extract the collection data:

```bash
tar -xvf msmarco-data/collection.tar.gz -C msmarco-data/
```

## Building Indices

We need to build two indices using the MS MARCO datasets. One index is for `collection.tsv` for the dense retriever to retrieve documents, and the other is for `queries.train.tsv` to find the nearest queries. Put the indices in `/msmarco-data/indices`.

```bash
python build_indices.py --collection_path msmarco-data/collection.tsv --output_dir msmarco-data/indices --index_type document
python build_indices.py --collection_path msmarco-data/queries.train.tsv --output_dir msmarco-data/indices --index_type query
```

## Finding Nearest Queries

Use the `find_topK_nearest_query.py` script to find and generate the nearest queries file in the `NNQ/output` folder.

```bash
python find_topK_nearest_query.py --query_index msmarco-data/indices/query_index --output_dir NNQ/output
```

## Running the Model

After generating the nearest queries file, use `query_main.py` to run the complete model. The outputs will be stored in the `correlations` folder.

```bash
python query_main.py --query_file NNQ/output/nearest_queries.tsv --output_dir correlations
```

## Results

The outputs of the model will be stored in the `correlations` folder, where you can find the results of the query performance prediction in terms of various correlation metrics.


```

## requirements.txt

```txt
faiss-gpu==1.7.2
numpy==1.26.3
pandas==2.1.4
scikit-learn==1.3.0
scipy==1.11.4
torch==2.1.2
transformers==4.32.1
sentence-transformers==2.2.2
rbo==0.1.3
```

Simply copy and paste the above content into your respective `README.md` and `requirements.txt` files.
