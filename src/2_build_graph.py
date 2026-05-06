import pandas as pd
import numpy as np
import scipy.sparse as sp
import os
import time
from preprocessing import load_and_clean_data
from graph_builder import TextGCNGraph

def main():
    print("=== STEP 2: Offline Graph Construction (With Semantic Upgrade) ===")
    start_time = time.time()
    
    # 1. Load the dataset
    data_path = "../data/dataset1_tweets_combined.csv" 
    print(f"Loading data from {data_path}...")
    cleaned_df = load_and_clean_data(data_path)
    
    # 2. Load the embeddings for the Semantic Upgrade
    print("Loading document embeddings for Semantic Graph Upgrade...")
    doc_features = np.load("../data/doc_embeddings.npy")
    
    # 3. Build the Graph components
    graph_builder = TextGCNGraph(cleaned_df)
    
    # Calculate Edges
    tfidf_matrix = graph_builder.build_tfidf_edges()
    pmi_edges = graph_builder.build_pmi_edges(window_size=20)
    
    # Calculate Lexical Edges
    jaccard_edges = graph_builder.build_jaccard_edges(threshold=0.2)
    
    # Calculate Semantic Edges (The New Upgrade)
    # We use a strict threshold (0.85) to ensure we only connect tweets with highly similar meanings
    semantic_edges = graph_builder.build_semantic_doc_edges(doc_features, threshold=0.85)
    
    # 4. Assemble the Master Adjacency Matrix
    A_matrix = graph_builder.build_adjacency_matrix(pmi_edges, jaccard_edges, semantic_edges)
    
    # 5. Save the Sparse Matrix to Hard Drive
    save_dir = "../data"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    save_path = os.path.join(save_dir, "A_matrix.npz")
    sp.save_npz(save_path, A_matrix)
    
    elapsed_time = time.time() - start_time
    print(f"\n[SUCCESS] Upgraded Adjacency Matrix saved to: {save_path}")
    print(f"Final Matrix Shape: {A_matrix.shape}")
    print(f"Total Edges (nnz): {A_matrix.nnz}")
    print(f"Graph Construction Time: {elapsed_time:.2f} seconds.")

if __name__ == "__main__":
    main()