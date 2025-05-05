import torch
from source.dataset_builder import create_training_dataset
from source.model_trainer import Trainer
from source.initial_setup import config
from source_finetune.finetuner import FineTuner
#from source.tester import tester
from source_pos_tagging.tester import Tester

# Set device


def train_full_model():
    #create_training_dataset_from_books()
    trainer = Trainer(
        train_file=config["training_full_model"]["training_data"],
        val_file=config["training_full_model"]["validation_data"],        
        output_dir=config["training_full_model"]["model_location"]+"\\tiny_model",
        epochs=3,
        batch_size=32,  # Can use larger batch size with smaller model
        learning_rate=5e-5,
        tokenizer_name=config["training_full_model"]["tokenizer"]
    )
    
    # Train the model
    model = trainer.train()
    
 
 
def finetune():
    fine_tuner = FineTuner()
    
    # Setup with config (automatically uses config values)
    tokenized_dataset = fine_tuner.setup()
    
    # Train
    fine_tuner.train()
    
    # Save LoRA adapter
    fine_tuner.save()
    
    # Test inference
    test_text = "correct case for sentence: hello world"
    result = fine_tuner.test_inference(test_text)
    print(f"Result: {result}")
    


def test_full_model():
    test_file = "files/dataset/testing.txt"  # Your file with 2000 lines
    
    # Option 1: Full evaluation with output file
    results = test_from_formatted_file(
        input_file=test_file,
        output_file="test_results.txt",
        verbose=True,  # Set to False for quiet mode,
        model_path="models/full_model"
    )
    
    print(f"\nFinal accuracy: {results['accuracy']:.2%}")
    
    # Option 2: Show only errors (faster)
    # test_from_file_errors_only(test_file)



def test_pos():
    # Path to your test file
    test_file_path = config["testing"]["file"]
    
    
    # Create tester
    tester = Tester(test_file_path)
        
    # Run tests
    results = tester.run_tests()
        
    # Analyze results
    tester.analyze_results(results)
        
    # Analyze error patterns
    tester.analyze_errors(results)
        
    # Save error report
    tester.save_error_report(results,output_file=f'{config["pos_tagging"]["testing_results"]}/error_report_pos.txt')
        
    print("\nTest completed. Error report saved to 'error_report.txt'")
        



#finetune()
test_pos()
