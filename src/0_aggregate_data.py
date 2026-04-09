import os
import pandas as pd

def main():
    print("=== STEP 0: Data Aggregation (Context Restoration Phase) ===")
    
    # Define where your folders are located
    base_dir = "../data"
    
    # Map the specific folders to the 3 distinct datasets from the research paper
    datasets_config = {
        "dataset3_webforums.csv": {
            "blogs_depression": 1,
            "blogs_non_depression": 0
        },
        "dataset4_reddit.csv": {
            "reddit_depression": 1,
            "reddit_non_depression": 0,
            "reddit_breastcancer": 0  # Used as medical control
        },
        "dataset5_mixed.csv": {
            "mixed_depression": 1,
            "mixed_non_depression": 0
        }
    }
    
    for output_filename, folder_mapping in datasets_config.items():
        dataset = []
        total_files = 0
        print(f"\n--- Constructing {output_filename} ---")
        
        for folder_name, label in folder_mapping.items():
            folder_path = os.path.join(base_dir, folder_name)
            
            # Skip if the folder isn't found
            if not os.path.exists(folder_path):
                print(f"[WARNING] Could not find folder: {folder_path}. Skipping...")
                continue
                
            print(f"Reading folder: {folder_name} (Label: {label})")
            
            for filename in os.listdir(folder_path):
                if filename.endswith(".txt"):
                    file_path = os.path.join(folder_path, filename)
                    
                    # Safely read the text file
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
                        # Strip newlines and carriage returns to make it a single continuous string
                        text = file.read().replace('\n', ' ').replace('\r', ' ').strip()
                        
                        if text:
                            dataset.append({"text": text, "label": label})
                            total_files += 1
                            
        # Convert to a Pandas DataFrame and save
        if dataset:
            df = pd.DataFrame(dataset)
            save_path = os.path.join(base_dir, output_filename)
            # THE FIX: Tell pandas to escape weird Reddit characters with a backslash
            df.to_csv(save_path, index=False, escapechar='\\')
            print(f"[SUCCESS] Saved {total_files} full-length posts to {save_path}")

if __name__ == "__main__":
    main()