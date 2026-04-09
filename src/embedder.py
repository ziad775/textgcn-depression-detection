import torch
from transformers import AutoTokenizer, AutoModel
import pandas as pd
import numpy as np

class EmotionEmbedder:
    def __init__(self, model_name="mental/mental-bert-base-uncased"):
        """
        Initializes the MentalBERT model natively in PyTorch for maximum stability.
        """
        print(f"\n--- Initializing PyTorch Emotion Embedder ---")
        print(f"Downloading/Loading model: {model_name}")
        
        # Hardware Check: Bridge PyTorch to your Nvidia GPU
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Compute Device Detected: {self.device.type.upper()}")
        
        # Load Hugging Face Tokenizer and Model (Forcing the token check)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        
        # Lock the model weights (we are extracting features, not training BERT)
        self.model.eval() 

    def get_document_embedding(self, text: str) -> np.ndarray:
        """
        Passes text through MentalBERT and extracts the final [CLS] token state.
        """
        # 1. Tokenize: Convert text into BERT's vocabulary numbers
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        
        # 2. Route data to the GPU
        inputs = {key: val.to(self.device) for key, val in inputs.items()}
        
        # 3. Forward Pass: Push the data through the neural network
        with torch.no_grad():
            outputs = self.model(**inputs)
            
        # 4. Extract the [CLS] token (Batch 0, Token 0, All 768 Features)
        cls_embedding = outputs.last_hidden_state[0, 0, :].cpu().numpy()
        
        return cls_embedding

    def process_dataset(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Iterates through the dataframe and applies the embedding function.
        """
        print("Extracting Document Node Embeddings [X Matrix Foundation]...")
        embeddings = []
        
        for text in df['cleaned_text']:
            emb = self.get_document_embedding(text)
            embeddings.append(emb)
            
        # Add the mathematical vectors as a new column in our dataframe
        df['doc_embedding'] = embeddings
        return df