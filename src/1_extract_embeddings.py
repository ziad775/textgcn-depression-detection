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
    print("=== STEP 1: Dual-Brain Transformer Ensemble Extraction ===")

    load_dotenv() 

    HUGGING_API_KEY = os.getenv("HUGGINGFACE_API_KEY")
    
    login(token=HUGGING_API_KEY)

    # 1. Hardware Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Executing on device: {device}")
    
    # 2. Load the Dataset using the synchronized cleaning pipeline
    data_path = "../data/dataset2_twitter_English.csv"
    output_path = "../data/doc_embeddings.npy"
    
    print(f"Loading and cleaning data from {data_path}...")
    # This guarantees Step 1 drops the exact same junk rows as Step 2 and 3!
    df = load_and_clean_data(data_path) 
    texts = df['cleaned_text'].astype(str).tolist()
    
    # ==========================================
    # MODEL 1: The Generalist (RoBERTa)
    # ==========================================
    print("\nLoading Model 1: roberta-base (General Syntax & Slang)...")
    tokenizer_roberta = AutoTokenizer.from_pretrained("roberta-base")
    model_roberta = AutoModel.from_pretrained("roberta-base").to(device)
    model_roberta.eval() # Lock model for inference only
    
    # ==========================================
    # MODEL 2: The Specialist (MentalBERT)
    # ==========================================
    print("Loading Model 2: mental/mental-bert-base-uncased (Clinical Undertones)...")
    tokenizer_mental = AutoTokenizer.from_pretrained("mental/mental-bert-base-uncased")
    model_mental = AutoModel.from_pretrained("mental/mental-bert-base-uncased").to(device)
    model_mental.eval()

    ensemble_embeddings = []
    
    print(f"\nBeginning 1536-Dimensional Feature Extraction for {len(texts)} texts...")
    
    # Disable gradient calculation to save massive amounts of RAM
    with torch.no_grad():
        for text in tqdm(texts, desc="Processing Tweets"):
            
            # --- Extract from RoBERTa ---
            inputs_rob = tokenizer_roberta(text, return_tensors="pt", truncation=True, max_length=512).to(device)
            outputs_rob = model_roberta(**inputs_rob)
            # Grab the [CLS] token (index 0)
            cls_rob = outputs_rob.last_hidden_state[:, 0, :].cpu().numpy().flatten()
            
            # --- Extract from MentalBERT ---
            inputs_men = tokenizer_mental(text, return_tensors="pt", truncation=True, max_length=512).to(device)
            outputs_men = model_mental(**inputs_men)
            # Grab the [CLS] token (index 0)
            cls_men = outputs_men.last_hidden_state[:, 0, :].cpu().numpy().flatten()
            
            # --- THE ENSEMBLE FUSION ---
            # Glue the two 768-dimensional arrays into one 1536-dimensional array
            fused_embedding = np.concatenate((cls_rob, cls_men))
            ensemble_embeddings.append(fused_embedding)
            
            # --- MEMORY PROTECTOR ---
            del inputs_rob, outputs_rob, inputs_men, outputs_men
            
    # 3. Save the Master Matrix
    print("\nExtraction complete! Saving matrix to disk...")
    final_matrix = np.vstack(ensemble_embeddings)
    np.save(output_path, final_matrix)
    
    print(f"[SUCCESS] Ensemble Embeddings saved to {output_path}")
    print(f"Final Matrix Shape: {final_matrix.shape} (Expected: {len(texts)}, 1536)")
    
    # Final cleanup
    del model_roberta, model_mental
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

if __name__ == "__main__":
    main()