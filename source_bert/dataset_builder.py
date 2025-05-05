"""this class is responsible to build input and output text. We will be using wikipedia to source data
Reads text files, splits into sentences, and creates training data
Format: lowercase_sentence</> original_sentence
Preserves all punctuation including question marks and exclamation points
"""

import re
from source.initial_setup import config
from torch.utils.data import Dataset, DataLoader


def clean_whitespace(text):
    """
    Clean up whitespace in text.
    Removes line breaks and extra spaces.
    """
    # Remove line breaks and replace with space
    text = text.replace('\n', ' ').replace('\r', ' ')
    
    # Remove multiple spaces, tabs, and other whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Remove spaces before punctuation
    text = re.sub(r'\s+([.,!?;:])', r'\1', text)
    
    # Ensure space after punctuation
    text = re.sub(r'([.,!?])([A-Za-z])', r'\1 \2', text)
    
    return text.strip()


def create_training_dataset(key):
    """
    Create a dataset from book files for case correction.
    Preserves all sentence-ending punctuation (., !, ?)
    
    Args:
        book_files (list): List of paths to text files
        output_file (str): Path where the dataset will be saved
    """
    # Get file paths
    book_files = config['source_files']
    output_file = config[key]['training_data']
    format_string = config[key]['format']
    sentence_count = 0
    
    # Open output file for writing
    with open(output_file, 'w', encoding='utf-8') as outfile:
        
        # Process each book file
        for book_file in book_files:
            print(f"Processing {book_file}...")
            
            # Read the book
            with open(book_file, 'r', encoding='utf-8') as infile:
                text = infile.read()
            
            # Clean whitespace first
            text = clean_whitespace(text)
            
            # Split text by sentences
            sentences = re.split(r'(?<=[.!?])\s+', text)
            
            # Process each sentence
            for sentence in sentences:
                sentence = sentence.strip()
                
                # Skip empty or very short sentences
                if len(sentence) < 10:
                    continue
                
                # Additional cleaning for individual sentences
                sentence = clean_whitespace(sentence)
                
                # Create lowercase version for input
                lowercase_sentence = sentence.lower()
                
                # Apply the format
                output_line = format_string.replace('<lowercase>', lowercase_sentence).replace('<original>', sentence)
                
                # Write to output file
                outfile.write(f"{output_line}\n")
                
                sentence_count += 1
                
                # Print progress
                if sentence_count % 1000 == 0:
                    print(f"Processed {sentence_count} sentences...")
    
    print(f"Done! Created dataset with {sentence_count} sentences in {output_file}")
    
    
    


class TrainingData(Dataset):
    """Dataset class that loads from a file with format: lowercase</> original"""
    
    def __init__(self, file_path, tokenizer, max_length=128):
        self.data = []
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Load data from file
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if '</>' in line:
                    lowercase, original = line.split('</>')
                    self.data.append((lowercase.strip(), original.strip()))
        
        print(f"Loaded {len(self.data)} examples from {file_path}")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        source_text, target_text = self.data[idx]
        
        # Tokenize inputs
        source_encodings = self.tokenizer(
            source_text,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        
        # Tokenize targets
        target_encodings = self.tokenizer(
            target_text,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        
        input_ids = source_encodings.input_ids.squeeze()
        attention_mask = source_encodings.attention_mask.squeeze()
        labels = target_encodings.input_ids.squeeze()
        
        # Replace padding token id's in the labels with -100
        labels[labels == self.tokenizer.pad_token_id] = -100
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }
        