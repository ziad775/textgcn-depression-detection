import pandas as pd
import re

def clean_text(text: str) -> str:
    """
    Cleans raw social media text while STRICTLY preserving emojis and emoticons,
    as dictated by the TextGCN depression detection methodology.
    """
    if not isinstance(text, str):
        return ""

    # 1. Convert to lowercase
    text = text.lower()
    
    # 2. Remove URLs (http:// or https://)
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    
    # 3. Remove user mentions (@username)
    text = re.sub(r'\@\w+', '', text)
    
    # 4. Remove HTML tags (like <br>)
    text = re.sub(r'<.*?>', '', text)
    
    # 5. REMOVED the aggressive punctuation stripper. 
    # MentalBERT's tokenizer handles punctuation and emoticons perfectly.
    
    # 6. Remove extra whitespace/tabs/newlines
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text 

def load_and_clean_data(file_path: str) -> pd.DataFrame:
    """
    Loads the raw CSV dataset and applies the cleaning function to the text column.
    """
    print(f"Loading data from {file_path}...")
    
    # Load the CSV using Pandas
    df = pd.read_csv(file_path)
    
    # --- THE FIX FOR REAL DATA ---
    # Rename 'tweet' to 'text' so the rest of the pipeline works.
    # Also fixes the 'lable' typo to standard 'label' for the training script.
    df.rename(columns={'tweet': 'text', 'lable': 'label'}, inplace=True)
    
    if 'text' not in df.columns:
        raise ValueError("Dataset must contain a 'text' column (or 'tweet').")
        
    print("Cleaning social media posts (preserving emojis)...")
    
    # Apply our clean_text function to every single row in the 'text' column
    df['cleaned_text'] = df['text'].apply(clean_text)
    
    # Drop rows that became empty after cleaning
    df = df[df['cleaned_text'].astype(bool)]
    
    print(f"Successfully cleaned {len(df)} posts.")
    return df