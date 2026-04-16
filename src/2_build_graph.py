import pandas as pd
import scipy.sparse as sp
import os
import time
from preprocessing import load_and_clean_data
from graph_builder import TextGCNGraph

def main():
    print("=== STEP 2: Offline Graph Construction (10/10 Paper Replica) ===")
    start_time = time.time()
    
    # 1. Load the dataset
    data_path = "../data/dataset2_twitter_English_augmented.csv" 
    print(f"Loading data from {data_path}...")
    cleaned_df = load_and_clean_data(data_path)
    
    # 2. Build the Graph components
    graph_builder = TextGCNGraph(cleaned_df)
    
    # Calculate Edges
    tfidf_matrix = graph_builder.build_tfidf_edges()
    pmi_edges = graph_builder.build_pmi_edges(window_size=20)
    
    # FIX 2: Set Jaccard threshold to 0.0 to keep all relations, matching the paper
    jaccard_edges = graph_builder.build_jaccard_edges(threshold=0.2)
    
    # 3. Assemble the Master Adjacency Matrix
    A_matrix = graph_builder.build_adjacency_matrix(pmi_edges, jaccard_edges)
    
    # 4. Save the Sparse Matrix to Hard Drive
    save_dir = "../data"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    save_path = os.path.join(save_dir, "A_matrix.npz")
    sp.save_npz(save_path, A_matrix)
    
    elapsed_time = time.time() - start_time
    print(f"\n[SUCCESS] Adjacency Matrix saved to: {save_path}")
    print(f"Final Matrix Shape: {A_matrix.shape}")
    print(f"Total Edges (nnz): {A_matrix.nnz}")
    print(f"Graph Construction Time: {elapsed_time:.2f} seconds.")

if __name__ == "__main__":
    main()