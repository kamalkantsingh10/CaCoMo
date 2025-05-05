# finetuner_complete.py
import pandas as pd
import torch
from transformers import (
    T5ForConditionalGeneration, 
    T5Tokenizer, 
    DataCollatorForSeq2Seq, 
    Seq2SeqTrainingArguments, 
    Seq2SeqTrainer
)
from peft import LoraConfig, get_peft_model, TaskType
from datasets import Dataset
import os
import yaml
from source.initial_setup import config,device


class FineTuner:
    def __init__(self, model_name="t5-small", output_dir="./lora_adapter"):
        self.model_name = model_name
        self.output_dir = output_dir
        self.model = None
        self.tokenizer = None
        self.trainer = None
        
        # Always use CPU
        self.device= device
        
    def load_data_from_file(self, file_path):
        """Load data from file with custom format"""
        data = []
        with open(file_path, 'r') as f:
            for line in f:
                line = line.strip()
                if '</>' in line:
                    try:
                        input_part, output_part = line.split('</>')
                        if input_part.startswith("correct case for sentence: "):
                            actual_input = input_part[27:]
                            data.append({
                                "input_text": f"correct case for sentence: {actual_input}",
                                "target_text": output_part.strip()
                            })
                    except ValueError:
                        continue
                    
        print(f"Loaded {len(data)} samples")
        df = pd.DataFrame(data)
        return Dataset.from_pandas(df)
    
    def preprocess_function(self, examples):
        """Preprocess data for T5"""
        # This is the fix - make sure inputs are lists
        inputs = examples["input_text"] if isinstance(examples["input_text"], list) else [examples["input_text"]]
        targets = examples["target_text"] if isinstance(examples["target_text"], list) else [examples["target_text"]]
        
        model_inputs = self.tokenizer(inputs, max_length=512, padding="max_length", truncation=True)
        
        with self.tokenizer.as_target_tokenizer():
            labels = self.tokenizer(targets, max_length=512, padding="max_length", truncation=True)
        
        # Optionally replace padding token labels with -100 to ignore them during training
        labels["input_ids"] = [[(l if l != self.tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]]
        
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs
    
    def setup(self, data_file=None, lora_config=None):
        """Setup model, tokenizer, and data"""
        # Get finetune config from global config
        finetune_config = config.get('finetune', {})
        
        # Use config values if available
        if data_file is None:
            data_file = finetune_config.get('training_data', '')
        
        # Load model and tokenizer
        self.model = T5ForConditionalGeneration.from_pretrained(self.model_name)
        self.tokenizer = T5Tokenizer.from_pretrained(self.model_name, legacy=True)
        
        # Add padding token if it doesn't exist
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load and preprocess data
        dataset = self.load_data_from_file(data_file)
        tokenized_dataset = dataset.map(
            self.preprocess_function, 
            batched=True,
            remove_columns=dataset.column_names  # Remove original columns to avoid conflicts
        )
        
        # Configure LoRA from config
        if lora_config is None:
            lora_settings = finetune_config.get('lora', {})
            lora_config = LoraConfig(
                r=lora_settings.get('r', 8),
                lora_alpha=lora_settings.get('alpha', 16),
                target_modules=["q", "v"],
                lora_dropout=lora_settings.get('dropout', 0.1),
                bias="none",
                task_type=TaskType.SEQ_2_SEQ_LM
            )
        
        # Apply LoRA to model
        self.model = get_peft_model(self.model, lora_config)
        
        # Move model to CPU
        self.model = self.model.to(self.device)
        
        # Get training settings from config
        training_args = Seq2SeqTrainingArguments(
            output_dir="./results",
            learning_rate=float(finetune_config.get('learning_rate', 1e-3)),
            per_device_train_batch_size=8,
            per_device_eval_batch_size=8,
            num_train_epochs=finetune_config.get('epochs', 3),
            weight_decay=0.01,
            eval_strategy="no",
            save_strategy="epoch",
            predict_with_generate=True,
            fp16=False,  # Disable fp16 for CPU
            dataloader_pin_memory=False,
            dataloader_num_workers=8,
            report_to=None
        )
        
        # Data collator
        data_collator = DataCollatorForSeq2Seq(
            tokenizer=self.tokenizer, 
            model=self.model,
            padding=True
        )
        
        # Create trainer
        self.trainer = Seq2SeqTrainer(
            model=self.model,
            args=training_args,
            train_dataset=tokenized_dataset,
            data_collator=data_collator,
        )
        
        return tokenized_dataset
    
    def train(self):
        """Train the model"""
        if self.trainer is None:
            raise ValueError("Must call setup() first")
        
        print("Training on CPU...")
        self.trainer.train()
        
    def save(self):
        """Save LoRA adapter"""
        if self.model is None:
            raise ValueError("Must train model first")
        
        self.model.save_pretrained(self.output_dir)
        print(f"LoRA adapter saved to {self.output_dir}")
    
    def test_inference(self, test_text):
        """Test the trained model"""
        if self.model is None or self.tokenizer is None:
            raise ValueError("Must train model first")
        
        # Move inputs to CPU (device is already CPU)
        inputs = self.tokenizer(test_text, return_tensors="pt").to(self.device)
        outputs = self.model.generate(**inputs, max_length=512)
        result = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return result


# Usage example:
if __name__ == "__main__":
    # Create fine-tuner instance
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