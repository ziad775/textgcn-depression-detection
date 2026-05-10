import pandas as pd
import numpy as np

def calculate_character_statistics(csv_path, text_column='text'):
    """
    Reads a dataset and calculates the exact character-count statistics.
    """
    print(f"Loading dataset from: {csv_path}...")
    df = pd.read_csv(csv_path)
    
    # SAFETY CHECK: Verify the column exists before doing anything
    if text_column not in df.columns:
        print(f"\n[ERROR] Could not find a column named '{text_column}'.")
        print(f"Here are the columns that actually exist in your CSV:")
        print(f"-> {list(df.columns)}")
        print(f"\nPlease update the 'text_column' variable at the bottom of this script to match one of the names above!\n")
        return None, None
        
    # Drop any NaN values just to be safe
    df = df.dropna(subset=[text_column])
    
    # Calculate the CHARACTER count for every single post
    # Using len(x) instead of len(x.split())
    char_counts = df[text_column].astype(str).apply(lambda x: len(x))
    
    # Extract the mathematical statistics
    max_len = char_counts.max()
    min_len = char_counts.min()
    mean_len = char_counts.mean()
    std_dev = char_counts.std()
    
    # Print the results
    print("\n==========================================")
    print(f" CHARACTER LENGTH STATISTICS ")
    print("==========================================")
    print(f"Total Posts:        {len(df)}")
    print(f"Minimum Characters: {min_len}")
    print(f"Maximum Characters: {max_len}")
    print(f"Mean (Average):     {mean_len:.0f}")
    print(f"Std Deviation:      {std_dev:.0f}")
    print("==========================================\n")
    
    return min_len, max_len

if __name__ == "__main__":
    dataset_path = "../data/dataset4_reddit.csv" 
    
    # Remember to match this with the actual column name found in the error list!
    calculate_character_statistics(dataset_path, text_column='text')