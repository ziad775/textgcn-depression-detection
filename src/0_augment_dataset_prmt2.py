import os
import pandas as pd
import time
from tqdm import tqdm
from dotenv import load_dotenv
from groq import Groq

# 1. Secure API Key Loading
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise ValueError("API Key not found! Please check your .env file.")

client = Groq(api_key=GROQ_API_KEY)

print("=== STEP 0: Neutral Semantic Expansion (Scientifically Valid) ===")

def get_neutral_context(text, max_retries=3):
    """
    Asks LLaMA-3 for a safe, neutral semantic expansion.
    Notice that the label is NOT passed to this function. The LLM cannot cheat.
    """
    
    prompt = f"""You are an expert descriptive writer. Your task is to expand the following short social media post by vividly describing the literal scene, physical actions, environment, and explicit feelings mentioned by the user. 

    Write naturally and organically. Only provide as much detail as the original text justifies. Do not pad the text with repetitive fluff.

    CRITICAL RULES:
    1. Do NOT act as a psychologist or therapist.
    2. Do NOT diagnose the user.
    3. You are strictly forbidden from using clinical mental health terminology (e.g., do not use words like depression, anxiety, trauma, cognitive distortion, or ideation).
    4. Do NOT invent events, backstory, or deep meanings that are not explicitly stated in the text. Keep it grounded in the literal words.

    Social Media Post: "{text}"
    """
    
    for attempt in range(max_retries):
        try:
            chat_completion = client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model="llama-3.1-8b-instant",
                max_tokens=250, # Generates organic length based on input
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
    input_path = "../data/dataset2_twitter_English.csv"
    # Saved with a new name so you don't overwrite your previous experiments!
    output_path = "../data/dataset2_twitter_English_neutral_expansion.csv" 
    
    print(f"Loading raw data from {input_path}...")
    df = pd.read_csv(input_path)
    
    # Clean headers
    df.columns = [str(col).lower().strip() for col in df.columns]
    
    # Hardcoded columns based on our previous fix
    text_col = 'tweet'
    label_col = 'label'
    
    augmented_texts = []
    
    print(f"Beginning Neutral Expansion for {len(df)} rows.")
    
    for index, row in tqdm(df.iterrows(), total=len(df), desc="Processing Tweets"):
        original_tweet = str(row[text_col])
        
        # --- CRASH PROTECTION ---
        try:
            _ = int(float(row[label_col]))
        except ValueError:
            print(f"\n[WARNING] Broken data at row {index}. Skipping! Target value was: {row[label_col]}")
            augmented_texts.append(original_tweet) 
            continue
        # ------------------------
        
        # WE ONLY PASS THE TEXT. The LLM is blind to the label.
        context = get_neutral_context(original_tweet)
        
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
    print(f"\n[SUCCESS] Scientifically valid augmented dataset saved to {output_path}!")

if __name__ == "__main__":
    main()