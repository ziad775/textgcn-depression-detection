import pandas as pd
import re
import emoji

def clean_text(text: str) -> str:
    """Cleans raw social media text, preserves emojis, and destroys Mojibake noise."""
    if not isinstance(text, str):
        return ""

    # 1. Strip URLs FIRST so they don't count as "real words"
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'pic\.twitter\.com\S+', '', text) # Explicitly catch twitter image links
    
    # 2. THE ALPHABET FILTER
    # If there are no English letters left after removing URLs, this is a garbage tweet.
    if not re.search(r'[a-zA-Z]', text):
        return "" # Returns empty, which will be dropped in the next step

    # 3. Preserve Emojis
    text = emoji.demojize(text, delimiters=(" ", " "))    
    text = text.lower()
    
    # 4. Standard Cleaning
    text = re.sub(r'\@\w+', '', text)
    text = re.sub(r'\#\w+', '', text)
    text = re.sub(r'<.*?>', '', text)
    
    # 5. PUNCTUATION CRUSHER
    # Converts multiple repetitive symbols (e.g., ??????? or .......) into a single symbol
    text = re.sub(r'([?.\!])\1+', r'\1', text)

    # 6. Clean up white spaces
    text = re.sub(r'\s+', ' ', text).strip()
    
    # 7. Final Safety Check (Must be at least 2 characters long)
    if len(text) < 2:
        return ""
        
    return text 

def load_and_clean_data(file_path: str) -> pd.DataFrame:
    """Loads the raw CSV dataset, applies text cleaning, and removes ghost nodes."""
    print(f"Loading data from {file_path}...")
    df = pd.read_csv(file_path)
    
    # Auto-detect Columns
    for col in ['text', 'tweet', 'post', 'content', 'message']:
        if col in df.columns:
            df.rename(columns={col: 'text'}, inplace=True)
            break
            
    for col in ['label', 'lable', 'target', 'class', 'sentiment', 'depression']:
        if col in df.columns:
            df.rename(columns={col: 'label'}, inplace=True)
            break
            
    print("Cleaning social media posts (Filtering Encoding Noise)...")
    original_len = len(df)
    
    # Apply the aggressive cleaning function
    df['cleaned_text'] = df['text'].apply(clean_text)
    
    # Drop all rows that became empty strings (The Mojibake / Ghost Nodes)
    df = df[df['cleaned_text'].astype(bool)]
    
    deleted_rows = original_len - len(df)
    
    print(f"-> Successfully cleaned {len(df)} posts.")
    print(f"-> [NOISE FILTER] Deleted {deleted_rows} garbage/empty tweets from the graph.")
    print(f"Current Class Distribution:\n{df['label'].value_counts()}")
    
    return df

    '''
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
    '''