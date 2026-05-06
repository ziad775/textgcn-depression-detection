import pandas as pd
import numpy as np
import scipy.sparse as sp
import os
import time
from imblearn.over_sampling import SMOTE
from preprocessing import load_and_clean_data
from graph_builder import HomogeneousGraphBuilder

def main():
    print("=== STEP 2: SMOTE Data Synthesis & Graph Construction ===")
    start_time = time.time()
    
    # 1. Load the Imbalanced Dataset (Make sure balance_data=False in preprocessing!)
    data_path = "../data/dataset1_tweets_combined.csv" 
    print(f"Loading raw labels from {data_path}...")
    
    # NOTE: Ensure your load_and_clean_data function returns the full imbalanced set here!
   # CORRECT
    cleaned_df = load_and_clean_data(data_path)
    raw_labels = cleaned_df['label'].values
    
    # 2. Load the Imbalanced Embeddings from Step 1
    print("Loading original document embeddings...")
    doc_features = np.load("../data/doc_embeddings.npy")
    
    print(f"\nOriginal Data Shape: {doc_features.shape}")
    print(f"Original Label Distribution: \n{pd.Series(raw_labels).value_counts()}")
    
    # ==========================================
    # THE SMOTE ALGORITHM
    # ==========================================
    print("\n[INITIATING SMOTE] Generating synthetic clinical vectors...")
    smote = SMOTE(random_state=42)
    
    # This generates brand new 1536-dimensional arrays for the minority class!
    balanced_features, balanced_labels = smote.fit_resample(doc_features, raw_labels)
    
    print(f"-> SMOTE Complete! New Data Shape: {balanced_features.shape}")
    print(f"-> New Label Distribution: \n{pd.Series(balanced_labels).value_counts()}")
    
    # 3. Save the new SMOTE-Balanced features so Step 3 can use them
    np.save("../data/balanced_doc_embeddings.npy", balanced_features)
    np.save("../data/balanced_labels.npy", balanced_labels)
    
    # ==========================================
    # GRAPH CONSTRUCTION
    # ==========================================
    # 4. Pass the balanced features to our new Homogeneous Graph Builder
    graph_builder = HomogeneousGraphBuilder(balanced_features, threshold=0.85)
    A_matrix = graph_builder.build_adjacency_matrix()
    
    # 5. Save the Matrix
    sp.save_npz("../data/A_matrix.npz", A_matrix)
    
    elapsed_time = time.time() - start_time
    print(f"\n[SUCCESS] Homogeneous SMOTE Graph Construction Time: {elapsed_time:.2f} seconds.")

if __name__ == "__main__":
    main()