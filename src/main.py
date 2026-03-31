import pandas as pd
import scipy.sparse as sp
from preprocessing import load_and_clean_data
from embedder import EmotionEmbedder
from graph_builder import TextGCNGraph

def main():
    print("=== TextGCN Pipeline Executing ===")
    
    # --- PHASE 1: Preprocessing ---
    print("\n[PHASE 1] Data Preprocessing")
    data_path = "../data/dummy_dataset.csv"
    cleaned_df = load_and_clean_data(data_path)
    
    # --- PHASE 2: Emotion Representation ---
    print("\n[PHASE 2] Emotion Representation Extraction")
    embedder = EmotionEmbedder(model_name="mental/mental-bert-base-uncased")
    embedded_df = embedder.process_dataset(cleaned_df)
    
    # --- PHASE 3: Graph Construction ---
    print("\n[PHASE 3] Graph Construction")
    graph_builder = TextGCNGraph(embedded_df)
    
    # 3A: TF-IDF (Doc-Word edges)
    tfidf_matrix = graph_builder.build_tfidf_edges()
    
    # 3B: PMI (Word-Word edges)
    pmi_edges = graph_builder.build_pmi_edges(window_size=20)
    
    # 3C: Master Adjacency Matrix (The A Matrix)
    A_matrix = graph_builder.build_adjacency_matrix(pmi_edges)

    # Verification Check
    print("\n--- Phase 3 Verification ---")
    print(f"Is Adjacency Matrix Sparse? : {sp.issparse(A_matrix)}")
    
    print("\n[PIPELINE STATUS] Ready for Phase 4: Spektral GCN")

if __name__ == "__main__":
    main()