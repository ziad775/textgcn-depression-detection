import pandas as pd
from groq import Groq
import time
from tqdm import tqdm
import os

print("=== STEP 0: Label-Aware Context Augmentation ===")

GROQ_API_KEY = os.getenv("GROQ_API_KEY") 
client = Groq(api_key=GROQ_API_KEY)

def get_clinical_context(text, label, max_retries=3):
    """
    Asks LLaMA-3 for context, but gives it STRICTLY DIFFERENT instructions 
    based on whether the tweet is actually depressed or healthy.
    """
    
    if label == 1:
        # PROMPT FOR DEPRESSED TWEETS
        prompt = f"""This social media post is from a user diagnosed with depression. 
        Your task is to act as a clinical psychologist and expand on the emotional subtext. 
        Describe the specific depressive symptoms, emotional pain, isolation, or cognitive distortions implied here. 
        Use rich, professional clinical vocabulary. 
        Keep your analysis between 150 and 200 words.
        
        Post: "{text}"
        """
    else:
        # PROMPT FOR HEALTHY / CONTROL TWEETS
        prompt = f"""This social media post is from a healthy control group. 
        Your task is to explain the literal context, humor, pop culture reference, or everyday situation occurring in this post. 
        Focus on joy, mundane daily complaints, standard human interaction, or internet memes. 
        CRITICAL INSTRUCTION: You MUST NOT use words like depression, anxiety, trauma, cognitive distortion, mental health, or hopelessness. 
        Keep your analysis between 150 and 200 words.
        
        Post: "{text}"
        """
    
    for attempt in range(max_retries):
        try:
            chat_completion = client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model="llama-3.1-8b-instant",
                max_tokens=300, # Safe limit for ~200 words
            )
            return chat_completion.choices[0].message.content.replace('\n', ' ').strip()
            
        except Exception as e:
            error_msg = str(e)
            if "429" in error_msg:
                wait_time = 5 
                print(f"\n[Warning] API busy. Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
            else:
                print(f"\n[Error] Unhandled Exception: {e}")
                return ""
                
    return "" 

def main():
    input_path = "../data/dataset1_tweets_combined.csv"
    output_path = "../data/dataset1_tweets_combined_augmented.csv"
    
    print(f"Loading raw data from {input_path}...")
    df = pd.read_csv(input_path)
    
    # Clean the column headers just in case there are invisible spaces
    df.columns = [str(col).lower().strip() for col in df.columns]
    
    # EXPLICITLY HARDCODE YOUR COLUMNS
    text_col = 'tweet'
    label_col = 'target'
    
    augmented_texts = []
    
    print(f"Beginning Label-Aware Augmentation for {len(df)} rows.")
    
    for index, row in tqdm(df.iterrows(), total=len(df), desc="Processing Tweets"):
        original_tweet = str(row[text_col])
        
        # --- CRASH PROTECTION ---
        # If a row is corrupted in the CSV, skip it instead of crashing!
        try:
            actual_label = int(float(row[label_col]))
        except ValueError:
            print(f"\n[WARNING] Broken data at row {index}. Skipping! Target value was: {row[label_col]}")
            # Append the original tweet un-augmented to preserve dataset length
            augmented_texts.append(original_tweet) 
            continue
        # ------------------------
        
        context = get_clinical_context(original_tweet, actual_label)
        
        if context == "":
            combined_text = original_tweet
        else:
            combined_text = f"{original_tweet} | Context: {context}"
            
        augmented_texts.append(combined_text)
        
        time.sleep(2.5) # Protect against API rate limits
        
        # SAFETY CHECKPOINT
        if (index + 1) % 100 == 0:
            df_checkpoint = df.iloc[:index+1].copy()
            df_checkpoint[text_col] = augmented_texts
            df_checkpoint.to_csv(output_path, index=False)
            
    # Final Save
    df[text_col] = augmented_texts
    df.to_csv(output_path, index=False)
    print(f"\n[SUCCESS] Safely augmented dataset saved to {output_path}!")

if __name__ == "__main__":
    main()