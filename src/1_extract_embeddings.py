import os
import torch
import gc
import numpy as np
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
from huggingface_hub import login
from preprocessing import load_and_clean_data
from dotenv import load_dotenv

def main():
    # ==========================================
    # THE MASTER TOGGLE SWITCH
    # ==========================================
    # Set to True for Quad-Brain (3072 dims, includes Sarcasm detection).
    # Set to False for Tri-Brain (2304 dims, higher clinical precision).
    ENABLE_IRONY_BRAIN = False
    # ==========================================

    mode_name = "Quad-Brain" if ENABLE_IRONY_BRAIN else "Tri-Brain"
    dim_size = 3072 if ENABLE_IRONY_BRAIN else 2304
    
    print(f"=== STEP 1: {mode_name} Transformer Ensemble Extraction (RAW) ===")

    load_dotenv() 
    HUGGING_API_KEY = os.getenv("HUGGINGFACE_API_KEY")
    if HUGGING_API_KEY:
        login(token=HUGGING_API_KEY)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Executing on device: {device}")
    
    data_path = "../data/dataset1_tweets_combined.csv"
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

    # ==========================================
    # MODEL 4: Irony (OPTIONAL)
    # ==========================================
    if ENABLE_IRONY_BRAIN:
        print("Loading Brain 4: cardiffnlp/twitter-roberta-base-irony...")
        tok_iro = AutoTokenizer.from_pretrained("cardiffnlp/twitter-roberta-base-irony")
        mod_iro = AutoModel.from_pretrained("cardiffnlp/twitter-roberta-base-irony").to(device)
        mod_iro.eval()
    else:
        print("Brain 4 (Irony) is DISABLED via Master Toggle.")

    ensemble_embeddings = []
    
    print(f"\nBeginning {dim_size}-Dimensional RAW Feature Extraction for {len(texts)} texts...")
    
    with torch.no_grad():
        for text in tqdm(texts, desc="Processing Tweets"):
            # Max length 128 applied to safely clip long inputs without crashing short-text models
            
            # 1. RoBERTa
            in_rob = tok_rob(text, return_tensors="pt", truncation=True, max_length=512).to(device)
            out_rob = mod_rob(**in_rob)
            cls_rob = out_rob.last_hidden_state[:, 0, :].cpu().numpy().flatten()
            
            # 2. MentalBERT
            in_men = tok_men(text, return_tensors="pt", truncation=True, max_length=512).to(device)
            out_men = mod_men(**in_men)
            cls_men = out_men.last_hidden_state[:, 0, :].cpu().numpy().flatten()

            # 3. GoEmotions
            in_emo = tok_emo(text, return_tensors="pt", truncation=True, max_length=512).to(device)
            out_emo = mod_emo(**in_emo)
            cls_emo = out_emo.last_hidden_state[:, 0, :].cpu().numpy().flatten()
            
            # Collect the core 3 brains
            fused_components = [cls_rob, cls_men, cls_emo]
            
            # 4. Irony (Conditional)
            if ENABLE_IRONY_BRAIN:
                in_iro = tok_iro(text, return_tensors="pt", truncation=True, max_length=512).to(device)
                out_iro = mod_iro(**in_iro)
                cls_iro = out_iro.last_hidden_state[:, 0, :].cpu().numpy().flatten()
                fused_components.append(cls_iro)
                
                # Cleanup specific to irony
                del in_iro, out_iro
            
            # THE FUSION
            # This dynamically concatenates either 3 arrays or 4 arrays based on the toggle
            fused_embedding = np.concatenate(fused_components)
            ensemble_embeddings.append(fused_embedding)
            
            # Manual memory management
            del in_rob, out_rob, in_men, out_men, in_emo, out_emo
            
    final_matrix = np.vstack(ensemble_embeddings)
    
    print(f"\nExtraction complete! Final Matrix Shape: {final_matrix.shape}")
    print(f"Warning: Each document node now has {final_matrix.shape[1]} features.")

    np.save(output_path, final_matrix)
    print(f"[SUCCESS] Raw Embeddings saved to {output_path}")
    
    # Final Cleanup
    del mod_rob, mod_men, mod_emo
    if ENABLE_IRONY_BRAIN:
        del mod_iro
        
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

if __name__ == "__main__":
    main()
