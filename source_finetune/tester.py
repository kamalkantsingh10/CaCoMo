# model_tester.py
import pandas as pd
import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer
from peft import PeftModel
import os
import time
from tqdm import tqdm
from collections import defaultdict


class ModelTester:
    def __init__(self, model_name="t5-small", adapter_path="./lora_adapter"):
        self.model_name = model_name
        self.adapter_path = adapter_path
        self.model = None
        self.tokenizer = None
        self.device = torch.device("cpu")  # Use CPU as per your setup
        
    def load_base_model(self):
        """Load the base model and tokenizer"""
        self.model = T5ForConditionalGeneration.from_pretrained(self.model_name)
        self.tokenizer = T5Tokenizer.from_pretrained(self.model_name, legacy=True)
        
        # Add padding token if it doesn't exist
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Move model to device
        self.model = self.model.to(self.device)
        
    def load_finetuned_model(self):
        """Load the fine-tuned model with LoRA adapter"""
        if not os.path.exists(self.adapter_path):
            raise FileNotFoundError(f"LoRA adapter not found at {self.adapter_path}")
        
        # First load base model
        self.load_base_model()
        
        # Then load LoRA adapter
        self.model = PeftModel.from_pretrained(self.model, self.adapter_path)
        print(f"LoRA adapter loaded from {self.adapter_path}")
        
    def load_test_data(self, file_path):
        """Load test data from file with custom format"""
        data = []
        with open(file_path, 'r') as f:
            for line in f:
                line = line.strip()
                if '</>' in line:
                    try:
                        input_part, output_part = line.split('</>')
                        # Add the required prompt prefix
                        data.append({
                            "input_text": f"correct case for sentence: {input_part}",
                            "target_text": output_part.strip()
                        })
                    except ValueError:
                        continue
        
        print(f"Loaded {len(data)} test samples")
        return pd.DataFrame(data)
    
    def predict(self, text):
        """Generate prediction for a single text"""
        # Tokenize input
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        
        # Generate output
        outputs = self.model.generate(**inputs, max_length=512)
        
        # Decode the output
        result = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return result
    
    def batch_predict(self, texts, batch_size=8):
        """Generate predictions for multiple texts"""
        predictions = []
        
        # Process in batches
        for i in tqdm(range(0, len(texts), batch_size), desc="Predicting"):
            batch_texts = texts[i:i+batch_size]
            
            # Tokenize batch
            inputs = self.tokenizer(batch_texts, 
                                  padding=True, 
                                  truncation=True, 
                                  return_tensors="pt").to(self.device)
            
            # Generate outputs
            outputs = self.model.generate(**inputs, max_length=512)
            
            # Decode outputs
            batch_predictions = [self.tokenizer.decode(output, skip_special_tokens=True) 
                               for output in outputs]
            predictions.extend(batch_predictions)
        
        return predictions
    
    def evaluate(self, test_file, output_file=None, batch_size=8):
        """Evaluate model on test data"""
        if self.model is None or self.tokenizer is None:
            raise ValueError("Must load model first")
        
        # Load test data
        test_df = self.load_test_data(test_file)
        
        # Get predictions
        start_time = time.time()
        predictions = self.batch_predict(test_df['input_text'].tolist(), batch_size)
        prediction_time = time.time() - start_time
        
        # Add predictions to dataframe
        test_df['predicted'] = predictions
        
        # Compute metrics
        metrics = self.compute_metrics(test_df)
        
        # Print results
        print("\n" + "="*50)
        print("EVALUATION RESULTS")
        print("="*50)
        print(f"Total samples: {len(test_df)}")
        print(f"Total prediction time: {prediction_time:.2f} seconds")
        print(f"Average time per prediction: {prediction_time/len(test_df):.3f} seconds")
        print(f"Exact match accuracy: {metrics['exact_match']:.2%}")
        print(f"Character accuracy: {metrics['char_accuracy']:.2%}")
        print(f"Word accuracy: {metrics['word_accuracy']:.2%}")
        
        # Print some examples
        print("\n" + "="*50)
        print("SAMPLE PREDICTIONS")
        print("="*50)
        for i in range(min(10, len(test_df))):
            print(f"\nExample {i+1}:")
            print(f"Input: {test_df.iloc[i]['input_text']}")
            print(f"Target: {test_df.iloc[i]['target_text']}")
            print(f"Predicted: {test_df.iloc[i]['predicted']}")
            print("-" * 30)
        
        # Save results in readable format if output file specified
        if output_file:
            with open(output_file, 'w') as f:
                # Write header
                f.write("MODEL EVALUATION RESULTS\n")
                f.write("=" * 50 + "\n\n")
                
                # Write evaluation metrics
                f.write("METRICS:\n")
                f.write(f"Total samples: {len(test_df)}\n")
                f.write(f"Total prediction time: {prediction_time:.2f} seconds\n")
                f.write(f"Average time per prediction: {prediction_time/len(test_df):.3f} seconds\n")
                f.write(f"Exact match accuracy: {metrics['exact_match']:.2%}\n")
                f.write(f"Character accuracy: {metrics['char_accuracy']:.2%}\n")
                f.write(f"Word accuracy: {metrics['word_accuracy']:.2%}\n")
                f.write("\n" + "=" * 50 + "\n\n")
                
                # Write individual test results
                f.write("TEST RESULTS:\n\n")
                for i, row in test_df.iterrows():
                    f.write(f"Sample {i+1}:\n")
                    f.write(f"Input: {row['input_text']}\n")
                    f.write(f"Target: {row['target_text']}\n")
                    f.write(f"Predicted: {row['predicted']}\n")
                    f.write(f"Correct: {'Yes' if row['target_text'] == row['predicted'] else 'No'}\n")
                    f.write("-" * 80 + "\n")
                
            print(f"\nResults saved to {output_file}")
        
        return metrics, test_df
    
    def compute_metrics(self, df):
        """Compute various metrics for evaluation"""
        metrics = {
            'exact_match': 0,
            'char_accuracy': 0,
            'word_accuracy': 0
        }
        
        exact_matches = 0
        total_chars = 0
        correct_chars = 0
        total_words = 0
        correct_words = 0
        
        for _, row in df.iterrows():
            target = row['target_text']
            predicted = row['predicted']
            
            # Exact match
            if target == predicted:
                exact_matches += 1
            
            # Character accuracy
            for t_char, p_char in zip(target, predicted):
                total_chars += 1
                if t_char == p_char:
                    correct_chars += 1
            
            # Handle cases where predicted is shorter
            if len(predicted) < len(target):
                total_chars += len(target) - len(predicted)
            
            # Word accuracy
            target_words = target.split()
            predicted_words = predicted.split()
            
            for t_word, p_word in zip(target_words, predicted_words):
                total_words += 1
                if t_word == p_word:
                    correct_words += 1
            
            # Handle cases where prediction has fewer words
            if len(predicted_words) < len(target_words):
                total_words += len(target_words) - len(predicted_words)
        
        metrics['exact_match'] = exact_matches / len(df)
        metrics['char_accuracy'] = correct_chars / total_chars if total_chars > 0 else 0
        metrics['word_accuracy'] = correct_words / total_words if total_words > 0 else 0
        
        return metrics
    
    def interactive_test(self):
        """Interactive testing interface"""
        if self.model is None or self.tokenizer is None:
            raise ValueError("Must load model first")
            
        print("\nINTERACTIVE TESTING MODE")
        print("Enter text to correct (or 'quit' to exit)")
        print("-" * 50)
        
        while True:
            user_input = input("\nInput: ")
            if user_input.lower() in ['quit', 'exit', 'q']:
                break
            
            # Add the prompt prefix
            full_input = f"correct case for sentence: {user_input}"
            prediction = self.predict(full_input)
            print(f"Correction: {prediction}")
    
    def error_analysis(self, test_file, output_file=None):
        """Analyze common errors made by the model"""
        test_df = self.load_test_data(test_file)
        predictions = self.batch_predict(test_df['input_text'].tolist())
        test_df['predicted'] = predictions
        
        # Find mismatches
        mismatches = test_df[test_df['target_text'] != test_df['predicted']]
        
        # Categorize errors
        error_types = defaultdict(list)
        
        for _, row in mismatches.iterrows():
            target = row['target_text']
            predicted = row['predicted']
            
            # Categorize error
            if len(predicted) < len(target):
                error_types['undergeneration'].append((target, predicted))
            elif len(predicted) > len(target):
                error_types['overgeneration'].append((target, predicted))
            elif predicted.lower() == target.lower():
                error_types['case_error'].append((target, predicted))
            else:
                error_types['other'].append((target, predicted))
        
        # Print error analysis
        print("\n" + "="*50)
        print("ERROR ANALYSIS")
        print("="*50)
        print(f"Total errors: {len(mismatches)}")
        
        for error_type, examples in error_types.items():
            print(f"\n{error_type.upper()}: {len(examples)} occurrences")
            # Show up to 5 examples
            for i, (target, predicted) in enumerate(examples[:5]):
                print(f"  Example {i+1}:")
                print(f"    Target:    {target}")
                print(f"    Predicted: {predicted}")
        
        # Save error analysis if output file specified
        if output_file:
            error_df = pd.DataFrame({
                'error_type': [],
                'target': [],
                'predicted': []
            })
            
            for error_type, examples in error_types.items():
                for target, predicted in examples:
                    error_df = pd.concat([error_df, pd.DataFrame({
                        'error_type': [error_type],
                        'target': [target],
                        'predicted': [predicted]
                    })], ignore_index=True)
            
            error_df.to_csv(output_file, index=False)
            print(f"\nError analysis saved to {output_file}")
        
        return error_types


# Usage example:
if __name__ == "__main__":
    # Create tester instance
    tester = ModelTester(model_name="t5-small", 
                         adapter_path="models/lora_adapter")
    
    # Load the fine-tuned model
    tester.load_finetuned_model()
    
    # Optional: Compare with base model
    # base_tester = ModelTester()
    # base_tester.load_base_model()
    
    # Evaluate on test data
    metrics, results_df = tester.evaluate("files/dataset/testing.txt", "test_results.csv")
    
    # Perform error analysis
    error_types = tester.error_analysis("files/dataset/testing.txt", "error_analysis.csv")
    
    # Interactive testing
    # tester.interactive_test()