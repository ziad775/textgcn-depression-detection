import pandas as pd
import re
import emoji

def clean_text(text: str) -> str:
    """
    Cleans raw social media text while STRICTLY preserving emojis and emoticons.
    """
    if not isinstance(text, str):
        return ""

    # NEW: Translate emojis into text before cleaning!
    # By using spaces as delimiters, 💔 becomes " broken_heart "
    #text = emoji.demojize(text, delimiters=(" ", " "))    

    text = text.lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'\@\w+', '', text)
    text = re.sub(r'\#\w+', '', text)
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text 

def load_and_clean_data(file_path: str) -> pd.DataFrame:
    """
    Loads the raw CSV dataset, auto-detects column names, and applies cleaning.
    """
    print(f"Loading data from {file_path}...")
    df = pd.read_csv(file_path)
    
    # 1. Auto-detect the TEXT column
    text_cols = ['text', 'tweet', 'post', 'content', 'message']
    for col in text_cols:
        if col in df.columns:
            df.rename(columns={col: 'text'}, inplace=True)
            break
            
    # 2. Auto-detect the LABEL column
    label_cols = ['label', 'lable', 'target', 'class', 'sentiment', 'depression']
    for col in label_cols:
        if col in df.columns:
            df.rename(columns={col: 'label'}, inplace=True)
            break
            
    # Safety Check
    if 'text' not in df.columns or 'label' not in df.columns:
        print(f"\n[ERROR] AVAILABLE COLUMNS IN CSV: {df.columns.tolist()}")
        raise ValueError("Could not find the text or label columns! Please check the column names above.")
        
    print("Cleaning social media posts (preserving emojis)...")
    df['cleaned_text'] = df['text'].apply(clean_text)
    
    # Drop rows that became empty after cleaning
    df = df[df['cleaned_text'].astype(bool)]
    
    print(f"Successfully cleaned {len(df)} posts.")
    return df