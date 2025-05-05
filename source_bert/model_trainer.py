import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertConfig, EncoderDecoderModel
from tqdm import tqdm
import matplotlib.pyplot as plt

from source.initial_setup import config
from source.dataset_builder import TrainingData


class Trainer:
    """Trainer class for Case Correction Model"""
    
    def __init__(self, train_file,val_file, output_dir="final_model", 
                 epochs=3, batch_size=16, learning_rate=5e-5,
                 max_length=128, tokenizer_name='bert-base-uncased',
                 model_config=None):
        self.train_file = train_file
        self.output_dir = output_dir
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.max_length = max_length
        self.tokenizer_name = tokenizer_name
        self.model_config = model_config or {}
        
        # step 1: Set device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # step 2: initialize tokenizer
        self.tokenizer = BertTokenizer.from_pretrained(self.tokenizer_name)
        # Add special tokens if not already present
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # step 3: Create dataset +loader
        train_dataset = TrainingData(self.train_file, self.tokenizer, self.max_length)
        self.train_dataloader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        
        val_dataset = TrainingData(val_file, self.tokenizer, self.max_length)
        self.val_dataloader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)
        # step 4: set up model -config.. lets stick with the config for now. We  
        encoder_config = BertConfig.from_pretrained(self.tokenizer_name)
        decoder_config = BertConfig.from_pretrained(self.tokenizer_name)
        
        # Set decoder-specific attributes
        decoder_config.is_decoder = True
        decoder_config.add_cross_attention = True
        
        config_updated = {
            # Model size parameters
            'hidden_size': 768,              # Size of hidden layers (default: 768)
            'num_hidden_layers': 12,         # Number of transformer layers (default: 12)
            'num_attention_heads': 12,       # Number of attention heads (default: 12)
            'intermediate_size': 3072,       # Size of "intermediate" (i.e., FeedForward) layer
            
            # Regularization
            'hidden_dropout_prob': 0.1,      # Dropout probability for all fully connected layers
            'attention_probs_dropout_prob': 0.1,  # Dropout probability for attention scores
            
            # Training behavior
            'initializer_range': 0.02,       # Standard deviation for weight initialization
            'layer_norm_eps': 1e-12,         # Epsilon for layer normalization
            
            # Vocabulary and sequence
            'vocab_size': encoder_config.vocab_size, # Keep the original vocab size
            'max_position_embeddings': 512,  # Maximum sequence length (default: 512)
            'type_vocab_size': 2,            # Number of token type embeddings
            
            # Activation function
            'hidden_act': "gelu",            # Activation function (default: "gelu")
        }
        
        for key, value in self.model_config.items():
            if key in config_updated:
                config_updated[key] = value
        
        # Apply updates to both encoder and decoder configs
        for key, value in config_updated.items():
            setattr(encoder_config, key, value)
            setattr(decoder_config, key, value)
        
        # Set special token ids for the decoder config
        decoder_config.decoder_start_token_id = self.tokenizer.cls_token_id
        decoder_config.eos_token_id = self.tokenizer.sep_token_id
        decoder_config.pad_token_id = self.tokenizer.pad_token_id
        
        # step 5 -lets build a model
        self.model = EncoderDecoderModel.from_encoder_decoder_pretrained(
            self.tokenizer_name, 
            self.tokenizer_name,
            encoder_config=encoder_config,
            decoder_config=decoder_config
        )
        
        # Set the special token ids in the model config as well
        self.model.config.decoder_start_token_id = self.tokenizer.cls_token_id
        self.model.config.eos_token_id = self.tokenizer.sep_token_id
        self.model.config.pad_token_id = self.tokenizer.pad_token_id
        
        self.model.to(self.device)
        
        # Initialize optimizer
        self.optimizer = optim.AdamW(self.model.parameters(), lr=self.learning_rate)
        
        self.train_losses = []
        self.val_losses = []
        print("Initialization and setup complete.")
    
    def train_epoch(self, epoch):
        """Train for one epoch"""
        
        # step 1: setting up tracking mechanism
        total_loss = 0
        progress_bar = tqdm(self.train_dataloader, desc=f"Epoch {epoch+1}/{self.epochs}")
        
        # step 2: put the model in training mode
        self.model.train()
        
        # step 3: 
        for batch in progress_bar:
            # step 3.1: Move the batch to device
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            labels = batch["labels"].to(self.device)
            
            # step 3.2: Forward pass
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                decoder_input_ids=None,
                labels=labels
            )
            
            loss = outputs.loss
            
            # step 3.3: Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            # step 3.4: update loss and track
            total_loss += loss.item()
            progress_bar.set_postfix({"loss": loss.item()})
        
        # step 4: findout average loss 
        avg_loss = total_loss / len(self.train_dataloader)
        self.train_losses.append(avg_loss)
        return avg_loss
    
    def validate_epoch(self):
        """Validate for one epoch"""
        if self.val_dataloader is None:
            return None
        
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for batch in self.val_dataloader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)
                
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                
                total_loss += outputs.loss.item()
        
        avg_loss = total_loss / len(self.val_dataloader)
        self.val_losses.append(avg_loss)
        return avg_loss
    
      
    
    def save_checkpoint(self, epoch, loss):
        """Save training checkpoint"""
        checkpoint_path = f"checkpoint_epoch_{epoch+1}.pt"
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': loss,
        }, checkpoint_path)
        print(f"Saved checkpoint to {checkpoint_path}")
    
    def save_final_model(self):
        """Save the final trained model"""
        os.makedirs(self.output_dir, exist_ok=True)
        self.model.save_pretrained(self.output_dir)
        self.tokenizer.save_pretrained(self.output_dir)
        print(f"Saved final model to '{self.output_dir}' directory")
    
    def plot_training_curve(self):
        """Plot and save training loss curve"""
        plt.figure(figsize=(10, 6))
        plt.plot(self.train_losses, label="Training Loss", marker='o')
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.title("Training Loss Over Time")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig("training_loss_curve.png", dpi=300, bbox_inches='tight')
        plt.close()
        print("Saved training loss curve to 'training_loss_curve.png'")
    
    def train(self):
        """Main training method with validation"""
        for epoch in range(self.epochs):
            # Training
            train_loss = self.train_epoch(epoch)
            
            # Validation
            val_loss = self.validate_epoch()
            
            print(f"Epoch {epoch+1}/{self.epochs}")
            print(f"  Training Loss:   {train_loss:.4f}")
            if val_loss is not None:
                print(f"  Validation Loss: {val_loss:.4f}")
            
            # Save checkpoint
            self.save_checkpoint(epoch, train_loss)
            print()
        
        # Save final model and plot
        self.save_final_model()
        self.plot_training_curve()
    
    def quick_test(self, test_sentences):
        """Run quick test on the trained model"""
        self.model.eval()
        
        print("\nQuick test results:")
        for sentence in test_sentences:
            # Tokenize
            inputs = self.tokenizer(sentence, return_tensors="pt", padding=True).to(self.device)
            
            # Generate
            with torch.no_grad():
                outputs = self.model.generate(**inputs)
            
            # Decode
            result = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            print(f"Input:  {sentence}")
            print(f"Output: {result}")
            print()