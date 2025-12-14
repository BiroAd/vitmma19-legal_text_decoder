import os
import json
import pandas as pd
import requests
import zipfile
from collections import Counter
from sklearn.model_selection import train_test_split

from utils import setup_logger

logger = setup_logger()

def download_and_extract_zip(url, extract_to):
    """
    Downloads a zip file from a URL and extracts it to the specified directory.
    Skips download if the directory already contains files.
    
    Args:
        url (str): URL to download the zip file from
        extract_to (str): Directory path where files will be extracted
    """
    # Ensure extraction directory exists
    os.makedirs(extract_to, exist_ok=True)
    print(f"Created directory {extract_to} for data extraction.")
    
    # Skip if directory already has files (excluding zip file)
    existing_files = [f for f in os.listdir(extract_to) if f != 'legaltextdecoder.zip']
    if existing_files:
        print(f"Data directory {extract_to} is not empty. Skipping download.")
        return
    
    zip_path = os.path.join(extract_to, 'legaltextdecoder.zip')
    
    try:
        # Download zip file
        print(f"Downloading data from {url}...")
        session = requests.Session()
        response = session.get(url, stream=True, timeout=30)
        response.raise_for_status()
        
        with open(zip_path, 'wb') as out_file:
            for chunk in response.iter_content(chunk_size=8192):
                out_file.write(chunk)
        
        # Extract zip file
        print("\nExtracting zip file...")
        print(f"Extracting to '{extract_to}'...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
        
        # Display extracted contents
        print("\nExtracted files:")
        for root, dirs, files in os.walk(extract_to):
            level = root.replace(extract_to, '').count(os.sep)
            indent = ' ' * 2 * level
            print(f"{indent}{os.path.basename(root)}/")
            subindent = ' ' * 2 * (level + 1)
            for file in files[:5]:
                print(f"{subindent}{file}")
            if len(files) > 5:
                print(f"{subindent}... and {len(files) - 5} more files")
        
        print("Download and extraction complete.")
        
    except requests.RequestException as e:
        print(f"Error downloading file: {e}")
        raise
    except zipfile.BadZipFile:
        print("Error: Invalid zip file.")
        raise
    finally:
        if os.path.exists(zip_path):
            os.remove(zip_path)

def preprocess_data(data_dir='data'):
    """
    Reads JSON annotation files from a directory, extracts text and sentiment,
    and returns two pandas DataFrames: one for consensus data and one for regular data.

    Args:
        data_dir (str): The path to the directory containing the JSON files.

    Returns:
        tuple: (consensus_df, regular_df) - Two DataFrames
    """
    consensus_data = []
    regular_data = []
    consensus_dir = os.path.join(data_dir, 'legaltextdecoder', 'consensus')

    # Check if the data directory exists
    if not os.path.exists(data_dir):
        print(f"Error: Directory not found at '{data_dir}'")
        print("Please ensure you are running this script from the project's root directory.")
        return pd.DataFrame(), pd.DataFrame()

    print(f"Starting preprocessing of files in '{data_dir}'...")
    
    # Search recursively for JSON files
    json_files = []
    for root, dirs, files in os.walk(data_dir):
        for filename in files:
            if filename.endswith('.json'):
                json_files.append(os.path.join(root, filename))
    
    if not json_files:
        print(f"Error: No JSON files found in '{data_dir}' or subdirectories.")
        return pd.DataFrame(), pd.DataFrame()
    
    print(f"Found {len(json_files)} JSON files.")
    
    # Separate consensus and regular files
    consensus_files = [f for f in json_files if consensus_dir in f]
    regular_files = [f for f in json_files if consensus_dir not in f]
    
    print(f"Consensus files: {len(consensus_files)}")
    print(f"Regular files: {len(regular_files)}")

    # Process consensus files with majority voting
    print("\nProcessing consensus files...")
    consensus_annotations = {}  # {text: [labels]}
    
    for file_path in consensus_files:
        filename = os.path.basename(file_path)
        print(f"Processing consensus file: {filename}...")
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                loaded_json = json.load(f)

            if not isinstance(loaded_json, list):
                print(f"Warning: Expected a list in {filename}. Skipping.")
                continue

            for record in loaded_json:
                text_content = extract_text(record, filename)
                if not text_content:
                    continue
                
                annotation_value = extract_annotation(record, filename)
                if annotation_value is None:
                    continue
                
                # Store multiple annotations for the same text
                if text_content not in consensus_annotations:
                    consensus_annotations[text_content] = []
                consensus_annotations[text_content].append(annotation_value)

        except Exception as e:
            print(f"Error processing {filename}: {e}")
    
    # Apply majority voting for consensus
    print(f"\nApplying majority voting for {len(consensus_annotations)} unique texts...")
    consensus_texts = set()  # Track consensus texts for deduplication
    
    for text, labels in consensus_annotations.items():
        # Count label occurrences
        label_counts = Counter(labels)
        majority_label = label_counts.most_common(1)[0][0]
        
        consensus_data.append({
            'filename': 'consensus',
            'text': text,
            'sentiment_choice': majority_label,
        })
        consensus_texts.add(text)  # Add to set for filtering

    # Process regular files
    print("\nProcessing regular files...")
    skipped_duplicates = 0
    
    for file_path in regular_files:
        filename = os.path.basename(file_path)
        print(f"Processing {filename}...")
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                loaded_json = json.load(f)

            if not isinstance(loaded_json, list):
                print(f"Warning: Expected a list in {filename}. Skipping.")
                continue

            if not loaded_json:
                print(f"Warning: {filename} is empty. Skipping.")
                continue

            for record in loaded_json:
                text_content = extract_text(record, filename)
                if not text_content:
                    continue
                
                # Skip if text already in consensus data
                if text_content in consensus_texts:
                    skipped_duplicates += 1
                    continue

                annotation_value = extract_annotation(record, filename)
                if annotation_value is None:
                    continue

                regular_data.append({
                    'filename': filename,
                    'text': text_content,
                    'sentiment_choice': annotation_value
                })

        except Exception as e:
            print(f"Error processing {filename}: {e}")

    print(f"\nPreprocessing complete.")
    print(f"Consensus samples: {len(consensus_data)}")
    print(f"Regular samples: {len(regular_data)}")
    print(f"Skipped duplicates (already in consensus): {skipped_duplicates}")
    
    consensus_df = pd.DataFrame(consensus_data)
    regular_df = pd.DataFrame(regular_data)
    
    return consensus_df, regular_df


def extract_text(record, filename):
    """Extract text content from a record."""
    if not isinstance(record, dict):
        return None
    
    # Try common patterns
    if 'data' in record and isinstance(record['data'], dict) and 'text' in record['data']:
        return record['data']['text']
    elif 'text' in record:
        return record['text']
    elif 'content' in record:
        return record['content']
    
    # Fallback: find first string field
    for v in record.values():
        if isinstance(v, str) and v.strip():
            return v
    
    return None


def extract_annotation(record, filename):
    """Extract annotation value from a record."""
    try:
        if (record.get('annotations') and 
            len(record['annotations']) > 0 and 
            record['annotations'][0].get('result') and
            len(record['annotations'][0]['result']) > 0 and
            record['annotations'][0]['result'][0].get('value') and
            record['annotations'][0]['result'][0]['value'].get('choices') and
            len(record['annotations'][0]['result'][0]['value']['choices']) > 0):
            
            annotation_value = record['annotations'][0]['result'][0]['value']['choices'][0]
            return annotation_value[0] if annotation_value else None
    except (KeyError, IndexError, TypeError):
        pass
    
    return None


if __name__ == "__main__":
    # This block runs when the script is executed directly.
    # We assume the script is run from the project root directory 
    # (e.g., by running: python src/preprocessing.py)

    DATA_DIRECTORY = './data'  # Relative path
    OUTPUT_DIRECTORY = './output'
    
    DOWNLOAD_URL = "https://bmeedu-my.sharepoint.com/:u:/g/personal/gyires-toth_balint_vik_bme_hu/IQDYwXUJcB_jQYr0bDfNT5RKARYgfKoH97zho3rxZ46KA1I?e=iFp3iz&download=1"
    download_and_extract_zip(DOWNLOAD_URL, DATA_DIRECTORY)
    
    consensus_df, regular_df = preprocess_data(DATA_DIRECTORY)
    
    # Ensure output directory exists
    if not os.path.exists(OUTPUT_DIRECTORY):
        os.makedirs(OUTPUT_DIRECTORY)
    
    # Save consensus data (test set)
    if not consensus_df.empty:
        print("\n" + "="*60)
        print("TEST SET (Consensus Data)")
        print("="*60)
        print(f"Total examples: {len(consensus_df)}")
        print("\nClass distribution:")
        class_counts = consensus_df['sentiment_choice'].value_counts().sort_index()
        for label, count in class_counts.items():
            percentage = (count / len(consensus_df)) * 100
            print(f"  Class {label}: {count:4d} ({percentage:5.2f}%)")
        
        print("\n--- Sample Data (Head) ---")
        print(consensus_df.head())
        
        consensus_path = os.path.join(DATA_DIRECTORY, 'test_sentiments.csv')
        try:
            consensus_df.to_csv(consensus_path, index=False, encoding='utf-8')
            print(f"\nSuccessfully saved test data to '{consensus_path}'")
        except Exception as e:
            print(f"\nError saving consensus data: {e}")
    else:
        print("No consensus data was processed.")
    
    if not regular_df.empty:
        print("\n" + "="*60)
        print("REGULAR DATA (Before Split)")
        print("="*60)
        print(f"Total examples: {len(regular_df)}")
        print("\nClass distribution:")
        class_counts = regular_df['sentiment_choice'].value_counts().sort_index()
        for label, count in class_counts.items():
            percentage = (count / len(regular_df)) * 100
            print(f"  Class {label}: {count:4d} ({percentage:5.2f}%)")
        
        # Stratified train/val split (85/15)
        try:
            train_df, val_df = train_test_split(
                regular_df,
                test_size=0.15,
                stratify=regular_df['sentiment_choice'],
                random_state=42
            )
            
            # Print train set statistics
            print("\n" + "="*60)
            print("TRAIN SET")
            print("="*60)
            print(f"Total examples: {len(train_df)}")
            print("\nClass distribution:")
            train_counts = train_df['sentiment_choice'].value_counts().sort_index()
            for label, count in train_counts.items():
                percentage = (count / len(train_df)) * 100
                print(f"  Class {label}: {count:4d} ({percentage:5.2f}%)")
            
            # Print validation set statistics
            print("\n" + "="*60)
            print("VALIDATION SET")
            print("="*60)
            print(f"Total examples: {len(val_df)}")
            print("\nClass distribution:")
            val_counts = val_df['sentiment_choice'].value_counts().sort_index()
            for label, count in val_counts.items():
                percentage = (count / len(val_df)) * 100
                print(f"  Class {label}: {count:4d} ({percentage:5.2f}%)")
            
            # Print overall summary
            print("\n" + "="*60)
            print("OVERALL SUMMARY")
            print("="*60)
            print(f"Train:      {len(train_df):5d} examples ({len(train_df)/(len(train_df)+len(val_df))*100:5.2f}%)")
            print(f"Validation: {len(val_df):5d} examples ({len(val_df)/(len(train_df)+len(val_df))*100:5.2f}%)")
            if not consensus_df.empty:
                print(f"Test:       {len(consensus_df):5d} examples")
                total = len(train_df) + len(val_df) + len(consensus_df)
                print(f"Total:      {total:5d} examples")
            
            # Save train and val datasets
            train_path = os.path.join(DATA_DIRECTORY, 'train_sentiments.csv')
            val_path = os.path.join(DATA_DIRECTORY, 'val_sentiments.csv')
            
            train_df.to_csv(train_path, index=False, encoding='utf-8')
            print(f"\nSuccessfully saved training data to '{train_path}'")
            
            val_df.to_csv(val_path, index=False, encoding='utf-8')
            print(f"Successfully saved validation data to '{val_path}'")
            
            # Save complete dataset
            complete_df = pd.concat([train_df, val_df, consensus_df], ignore_index=True)
            complete_path = os.path.join(DATA_DIRECTORY, 'complete_sentiments.csv')
            complete_df.to_csv(complete_path, index=False, encoding='utf-8')
            print(f"Successfully saved complete dataset to '{complete_path}'")
            
        except ValueError as e:
            print(f"\nError during stratified split: {e}")
            print("This may happen if some classes have too few samples.")
            print("Falling back to saving the full dataset without splitting.")
            
            regular_path = os.path.join(OUTPUT_DIRECTORY, 'preprocessed_sentiments.csv')
            regular_df.to_csv(regular_path, index=False, encoding='utf-8')
            print(f"Saved full dataset to '{regular_path}'")
            
    else:
        print("No regular data was processed.")