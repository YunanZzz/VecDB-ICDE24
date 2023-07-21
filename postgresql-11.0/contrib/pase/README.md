# Target
This repository contains code used for comparing performance of PASE and Faiss.

# Pase
Pase is an extension of PostgreSQL used for Approximate Nearest Neighbor Search. 
It has implemented two index: IVFFlat and HNSW. We implemented a new index IVFPQ in
this repository.

## Prerequisite

1. Install PostgreSQL11.0 (Pase may have conflicts with newer versions of PG)
2. Install Openmp

## Pase Building

1. Download Pase under folder `contrib` of PG
2. cd pase
3. `make USE_PGXS=`
4. `make install`
5. Start PG
6. Input `create extension pase;` in psql command line


## Modified From:

- [Pase: PostgreSQL Ultra-High Dimensional Approximate Nearest Neighbor Search Extension](https://github.com/alipay/PASE)

# Faiss

- [Faiss: A Library for Efficient Similarity Search and Clustering of Dense Vectors](https://github.com/facebookresearch/faiss)


