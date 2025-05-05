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
        self.is_finetuned = False  # Track whether model is fine-tuned
        
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
        self.is_finetuned = True
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
        print(f"EVALUATION RESULTS ({'Fine-tuned' if self.is_finetuned else 'Base'} Model)")
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
        
        # Save results if output file specified
        if output_file:
            test_df.to_csv(output_file, index=False)
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
    
    def compare_trained_vs_untrained(self, test_file, output_file=None):
        """Compare outputs from trained vs untrained model"""
        if not self.is_finetuned:
            print("Error: Current model is not fine-tuned. Load base model first to compare.")
            return
        
        # Save current model (fine-tuned)
        finetuned_model = self.model
        
        # Create and load base model for comparison
        base_tester = ModelTester(model_name=self.model_name)
        base_tester.load_base_model()
        
        # Load test data
        test_df = self.load_test_data(test_file)
        
        print("\nGenerating predictions with both models...")
        
        # Get predictions from fine-tuned model
        finetuned_predictions = self.batch_predict(test_df['input_text'].tolist())
        
        # Get predictions from base model
        self.model = base_tester.model  # Temporarily switch to base model
        self.tokenizer = base_tester.tokenizer
        self.is_finetuned = False
        
        base_predictions = self.batch_predict(test_df['input_text'].tolist())
        
        # Restore fine-tuned model
        self.model = finetuned_model
        self.is_finetuned = True
        
        # Add both predictions to dataframe
        test_df['base_predicted'] = base_predictions
        test_df['finetuned_predicted'] = finetuned_predictions
        
        # Compute metrics for both models
        base_metrics = self.compute_metrics(test_df[['target_text', 'base_predicted']].copy().rename(columns={'base_predicted': 'predicted'}))
        finetuned_metrics = self.compute_metrics(test_df[['target_text', 'finetuned_predicted']].copy().rename(columns={'finetuned_predicted': 'predicted'}))
        
        # Print comparison
        print("\n" + "="*50)
        print("MODEL COMPARISON")
        print("="*50)
        print(f"{'Metric':<25} {'Base Model':<15} {'Fine-tuned Model':<15}")
        print("-" * 55)
        print(f"{'Exact match accuracy':<25} {base_metrics['exact_match']:<15.2%} {finetuned_metrics['exact_match']:<15.2%}")
        print(f"{'Character accuracy':<25} {base_metrics['char_accuracy']:<15.2%} {finetuned_metrics['char_accuracy']:<15.2%}")
        print(f"{'Word accuracy':<25} {base_metrics['word_accuracy']:<15.2%} {finetuned_metrics['word_accuracy']:<15.2%}")
        
        # Print sample comparisons
        print("\n" + "="*50)
        print("SAMPLE COMPARISONS")
        print("="*50)
        for i in range(min(10, len(test_df))):
            print(f"\nExample {i+1}:")
            print(f"Input:      {test_df.iloc[i]['input_text']}")
            print(f"Target:     {test_df.iloc[i]['target_text']}")
            print(f"Base:       {test_df.iloc[i]['base_predicted']}")
            print(f"Fine-tuned: {test_df.iloc[i]['finetuned_predicted']}")
            
            # Highlight differences
            if test_df.iloc[i]['base_predicted'] != test_df.iloc[i]['finetuned_predicted']:
                print("           ↑ Outputs differ")
            elif test_df.iloc[i]['target_text'] == test_df.iloc[i]['finetuned_predicted']:
                print("           ↑ Fine-tuned correct!")
            print("-" * 30)
        
        # Analyze improvements
        improvements = 0
        deteriorations = 0
        for _, row in test_df.iterrows():
            base_correct = row['base_predicted'] == row['target_text']
            finetuned_correct = row['finetuned_predicted'] == row['target_text']
            
            if not base_correct and finetuned_correct:
                improvements += 1
            elif base_correct and not finetuned_correct:
                deteriorations += 1
        
        print("\n" + "="*50)
        print("IMPROVEMENT ANALYSIS")
        print("="*50)
        print(f"Cases improved by fine-tuning: {improvements}")
        print(f"Cases worsened by fine-tuning: {deteriorations}")
        print(f"Net improvement: {improvements - deteriorations}")
        
        # Save results if output file specified
        if output_file:
            comparison_df = test_df.copy()
            comparison_df.to_csv(output_file, index=False)
            print(f"\nComparison results saved to {output_file}")
        
        return test_df, base_metrics, finetuned_metrics



if __name__ == "__main__":
    # Create tester instance
    tester = ModelTester(model_name="t5-small", adapter_path="models/lora_adapter")
    
    # Load the fine-tuned model
    tester.load_finetuned_model()
    
    # Evaluate fine-tuned model on test data
    metrics, results_df = tester.evaluate("files/dataset/testing.txt", "files/testing/finetuned/test_results.csv")
    
    # Compare trained vs untrained model
    comparison_df, base_metrics, finetuned_metrics = tester.compare_trained_vs_untrained(
        "files/dataset/testing.txt", 
        "files/testing/finetuned/comparison_results.csv"
    )
    
    # Perform error analysis
    error_types = tester.error_analysis("files/dataset/testing.txt", "error_analysis.csv")
    
    # Interactive testing
    # tester.interactive_test()
    
    # For testing just the base model:
    # base_tester = ModelTester()
    # base_tester.load_base_model()
    # base_metrics, base_results_df = base_tester.evaluate("test_data.txt")