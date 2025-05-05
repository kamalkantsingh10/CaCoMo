# Check if the model understands the case correction task

def analyze_model_behavior(model_path="models/full_model"):
    """Analyze what the model is actually learning"""
    
    import torch
    from transformers import BertTokenizer, EncoderDecoderModel
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = BertTokenizer.from_pretrained(model_path)
    model = EncoderDecoderModel.from_pretrained(model_path)
    model.to(device)
    model.eval()
    
    print("ANALYZING MODEL BEHAVIOR")
    print("=" * 50)
    
    # Test if the model is producing capitalized output at all
    test_cases = [
        "this is a test",
        "hello world",
        "mary had a little lamb",
        "the quick brown fox",
        "i am a lowercase sentence"
    ]
    
    all_lowercase = True
    
    for test in test_cases:
        inputs = tokenizer(test, return_tensors="pt", padding=True, max_length=64).to(device)
        
        with torch.no_grad():
            outputs = model.generate(**inputs, max_length=64)
        
        result = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"Input:  '{test}'")
        print(f"Output: '{result}'")
        
        # Check if output has any capital letters
        has_capitals = any(c.isupper() for c in result)
        if not has_capitals:
            print("⚠️ Output has NO capital letters!")
        else:
            print("✓ Output contains capital letters")
            all_lowercase = False
        
        print()
    
    if all_lowercase:
        print("\n🚨 CRITICAL ISSUE: Model is not producing ANY capital letters!")
        print("   The model may not have learned the case correction task at all.")
    
    # Test with variation
    print("\nTesting with forced generation parameters:")
    print("-" * 50)
    
    test_case = "hello world this is a test"
    inputs = tokenizer(test_case, return_tensors="pt", padding=True, max_length=64).to(device)
    
    # Test different generation strategies
    strategies = [
        {"name": "Default", "params": {}},
        {"name": "No sampling", "params": {"do_sample": False}},
        {"name": "With temperature", "params": {"do_sample": True, "temperature": 1.0}},
        {"name": "Low temperature", "params": {"do_sample": True, "temperature": 0.3}},
    ]
    
    for strategy in strategies:
        with torch.no_grad():
            outputs = model.generate(**inputs, max_length=64, **strategy["params"])
        
        result = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"{strategy['name']}: '{result}'")
        print(f"  Has capitals: {any(c.isupper() for c in result)}")
    
    return True


def check_training_data_distribution(train_file):
    """Check if training data is properly balanced"""
    
    lowercase_count = 0
    uppercase_count = 0
    sample_outputs = []
    
    with open(train_file, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if '</>' in line:
                input_text, output_text = line.split('</>')
                input_text = input_text.strip()
                output_text = output_text.strip()
                
                # Check if input is fully lowercase
                if input_text.islower():
                    lowercase_count += 1
                
                # Check if output has capitals
                if any(c.isupper() for c in output_text):
                    uppercase_count += 1
                
                # Save some samples
                if i < 5:
                    sample_outputs.append(output_text)
    
    print("\nTRAINING DATA ANALYSIS")
    print("=" * 50)
    print(f"Lines with lowercase input: {lowercase_count}")
    print(f"Lines with capitalized output: {uppercase_count}")
    print(f"Percentage with capitals: {uppercase_count/lowercase_count*100:.1f}%")
    
    print("\nSample expected outputs:")
    for i, sample in enumerate(sample_outputs):
        print(f"{i+1}. '{sample}'")
        print(f"   Has capitals: {any(c.isupper() for c in sample)}")
    
    if uppercase_count < lowercase_count * 0.8:
        print("\n⚠️ WARNING: Not enough examples with capital letters in training data!")


def minimal_retraining_test():
    """Suggest minimal retraining with cleaner data"""
    
    print("\nMINIMAL RETRAINING RECOMMENDATION")
    print("=" * 50)
    print("1. Create a small clean dataset (100-200 examples)")
    print("2. Ensure each example has proper case conversion")
    print("3. Train for 20-30 epochs with learning rate 1e-5")
    print("4. Test frequently during training")
    
    # Sample clean training data format
    print("\nExample clean training data format:")
    print("-" * 30)
    clean_examples = [
        "hello world</> Hello world",
        "this is a test</> This is a test",
        "new york city</> New York City",
        "mary had a little lamb</> Mary had a little lamb",
        "the quick brown fox</> The quick brown fox"
    ]
    
    for example in clean_examples:
        print(example)
    
    print("\nThis format ensures the model learns simple case conversion first.")


# Main execution
if __name__ == "__main__":
    model_path = "models/full_model"
    train_file = "train.txt"
    
    # Analyze model behavior
    analyze_model_behavior(model_path)
    
    # Check training data
    check_training_data_distribution(train_file)
    
    # Give recommendations
    minimal_retraining_test()