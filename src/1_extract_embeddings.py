import os
import torch
import gc
import numpy as np
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
from sklearn.decomposition import PCA
from huggingface_hub import login
from preprocessing import load_and_clean_data
from dotenv import load_dotenv

def main():
    print("=== STEP 1: Tri-Brain Transformer Ensemble Extraction ===")

    load_dotenv() 
    HUGGING_API_KEY = os.getenv("HUGGINGFACE_API_KEY")
    login(token=HUGGING_API_KEY)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Executing on device: {device}")
    
    data_path = "../data/dataset2_twitter_English.csv"
    output_path = "../data/doc_embeddings.npy"
    
    print(f"Loading and cleaning data from {data_path}...")
    df = load_and_clean_data(data_path) 
    texts = df['cleaned_text'].astype(str).tolist()
    
    # ==========================================
    # MODEL 1: RoBERTa (Syntax & General Context)
    # ==========================================
    print("\nLoading Brain 1: roberta-base...")
    tok_rob = AutoTokenizer.from_pretrained("roberta-base")
    mod_rob = AutoModel.from_pretrained("roberta-base").to(device)
    mod_rob.eval()
    
    # ==========================================
    # MODEL 2: MentalBERT (Clinical Undertones)
    # ==========================================
    print("Loading Brain 2: mental/mental-bert-base-uncased...")
    tok_men = AutoTokenizer.from_pretrained("mental/mental-bert-base-uncased")
    mod_men = AutoModel.from_pretrained("mental/mental-bert-base-uncased").to(device)
    mod_men.eval()

    # ==========================================
    # MODEL 3: GoEmotions (Psychological Affect)
    # ==========================================
    print("Loading Brain 3: SamLowe/roberta-base-go_emotions...")
    tok_emo = AutoTokenizer.from_pretrained("SamLowe/roberta-base-go_emotions")
    mod_emo = AutoModel.from_pretrained("SamLowe/roberta-base-go_emotions").to(device)
    mod_emo.eval()

    ensemble_embeddings = []
    
    print(f"\nBeginning 2304-Dimensional Feature Extraction for {len(texts)} texts...")
    
    with torch.no_grad():
        for text in tqdm(texts, desc="Processing Tweets"):
            # 1. RoBERTa Extraction
            in_rob = tok_rob(text, return_tensors="pt", truncation=True, max_length=512).to(device)
            out_rob = mod_rob(**in_rob)
            cls_rob = out_rob.last_hidden_state[:, 0, :].cpu().numpy().flatten()
            
            # 2. MentalBERT Extraction
            in_men = tok_men(text, return_tensors="pt", truncation=True, max_length=512).to(device)
            out_men = mod_men(**in_men)
            cls_men = out_men.last_hidden_state[:, 0, :].cpu().numpy().flatten()

            # 3. GoEmotions Extraction
            in_emo = tok_emo(text, return_tensors="pt", truncation=True, max_length=512).to(device)
            out_emo = mod_emo(**in_emo)
            cls_emo = out_emo.last_hidden_state[:, 0, :].cpu().numpy().flatten()
            
            # THE TRI-BRAIN FUSION (768 + 768 + 768 = 2304 dimensions)
            fused_embedding = np.concatenate((cls_rob, cls_men, cls_emo))
            ensemble_embeddings.append(fused_embedding)
            
            del in_rob, out_rob, in_men, out_men, in_emo, out_emo
            
    final_matrix = np.vstack(ensemble_embeddings)
    print(f"\nExtraction complete! Raw Matrix Shape: {final_matrix.shape}")

    # ==========================================
    # PCA COMPRESSION (RAM Protector)
    # ==========================================
    print("\n[INITIATING PCA] Compressing array to save CPU memory...")
    pca = PCA(n_components=0.95, random_state=42)
    compressed_matrix = pca.fit_transform(final_matrix)
    
    new_dim = compressed_matrix.shape[1]
    mem_saved = 100 - ((new_dim / 2304) * 100)
    
    print(f"-> Array compressed from 2304 down to {new_dim} dimensions.")
    print(f"-> Retained 95% of clinical data while saving {mem_saved:.1f}% RAM!")

    # Save the compressed matrix
    np.save(output_path, compressed_matrix)
    print(f"[SUCCESS] Compressed Embeddings saved to {output_path}")
    
    # Cleanup
    del mod_rob, mod_men, mod_emo
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

if __name__ == "__main__":
    main()