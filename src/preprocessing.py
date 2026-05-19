import pandas as pd
import re
import emoji
import nltk
from nltk.corpus import stopwords

# Run this once to ensure the dictionary is downloaded on your machine
nltk.download('stopwords', quiet=True)

# ==========================================
# CLINICAL NLP CONFIGURATION
# ==========================================
# 1. Define the Clinical Whitelist (The "I-Paradigm")
# We remove standard English filler words, but STRICTLY KEEP these psychological markers
CLINICAL_WHITELIST = {"i", "me", "my", "myself", "mine", "we", "us", "our", "ours", "you", "your"}

# 2. Create the "Safe Stopwords" list
# This contains 'is', 'are', 'the', 'any', 'a', etc., but NOT the whitelisted pronouns
SAFE_STOPWORDS = set(stopwords.words('english')) - CLINICAL_WHITELIST

def clean_text(text: str) -> str:
    """
    Cleans raw social media text, preserves clinical pronouns, translates emojis,
    and destroys Mojibake/Punctuation noise to optimize Graph Node creation.
    """
    if not isinstance(text, str):
        return ""

    # 1. Strip URLs FIRST so they don't count as "real words"
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'pic\.twitter\.com\S+', '', text) # Explicitly catch twitter image links
    
    # 2. THE ALPHABET FILTER
    # If there are no English letters left after removing URLs, this is a garbage tweet.
    if not re.search(r'[a-zA-Z]', text):
        return "" 

    # 3. Preserve & Translate Emojis (Crucial for GoEmotions)
    # e.g., 💔 becomes " broken_heart "
    #text = emoji.demojize(text, delimiters=(" ", " "))    
    text = text.lower()
    
    # 4. Standard Social Media Cleaning
    text = re.sub(r'\@\w+', '', text)  # Remove @ mentions
    text = re.sub(r'\#\w+', '', text)  # Remove # hashtags
    text = re.sub(r'<.*?>', '', text)  # Remove HTML tags
    
    # 5. THE NUMBER CRUSHER
    # Removes all standalone digits to prevent matrix bloat and noisy graph edges
    text = re.sub(r'\b\d+\b', '', text)

    # 6. AGGRESSIVE PUNCTUATION STRIPPER
    # Because emojis are now text, we can safely destroy all punctuation 
    # to prevent node duplication (e.g., merging "approach." and "approach")
    text = re.sub(r'[^\w\s]', '', text)

    # 7. SMART STOP-WORD REMOVAL
    # Remove safe stopwords to prevent GCN over-smoothing, but keep clinical pronouns
    words = text.split()
    filtered_words = [w for w in words if w not in SAFE_STOPWORDS]
    text = " ".join(filtered_words)

    # 8. Clean up white spaces
    text = re.sub(r'\s+', ' ', text).strip()
    
    # 9. Final Safety Check (Must be at least 2 characters long)
    if len(text) < 2:
        return ""
        
    return text 

def load_and_clean_data(file_path: str) -> pd.DataFrame:
    """
    Loads the raw CSV dataset, applies advanced text cleaning, 
    and drops ghost nodes from the final graph architecture.
    """
    print(f"Loading data from {file_path}...")
    df = pd.read_csv(file_path)
    
    # Auto-detect the Text Column
    for col in ['text', 'tweet', 'post', 'content', 'message']:
        if col in df.columns:
            df.rename(columns={col: 'text'}, inplace=True)
            break
            
    # Auto-detect the Label Column
    for col in ['label', 'lable', 'target', 'class', 'sentiment', 'depression']:
        if col in df.columns:
            df.rename(columns={col: 'label'}, inplace=True)
            break
            
    print("Cleaning social media posts (Applying Smart Stop-Word Filter & Demojization)...")
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
