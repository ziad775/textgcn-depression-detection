import pandas as pd
import numpy as np
import os
from preprocessing import load_and_clean_data
from embedder import EmotionEmbedder

def main():
    print("=== STEP 1: Offline Feature Extraction ===")
    
    # 1. Load the dataset
    data_path = "../data/twitter_English.csv"
    print(f"Loading data from {data_path}...")
    df = load_and_clean_data(data_path)
    
    # 2. Extract Embeddings
    print("\nFiring up MentalBERT...")
    embedder = EmotionEmbedder(model_name="mental/mental-bert-base-uncased")
    embedded_df = embedder.process_dataset(df)
    
    # 3. Extract the raw math (The X Matrix for documents)
    # np.vstack stacks the individual arrays into one massive 2D grid
    doc_features = np.vstack(embedded_df['doc_embedding'].values)
    
    # 4. Save to Hard Drive
    save_dir = "../data"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    save_path = os.path.join(save_dir, "doc_embeddings.npy")
    np.save(save_path, doc_features)
    
    print(f"\n[SUCCESS] Document embeddings saved to: {save_path}")
    print(f"Final Matrix Shape: {doc_features.shape}")
    print("Your GPU is now safe. You never have to run MentalBERT on this specific data again!")

if __name__ == "__main__":
    main()