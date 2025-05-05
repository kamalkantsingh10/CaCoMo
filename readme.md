# **Testing 3 Ways to Fix Lowercase Text- Rule Based, Transformer, and Fine-Tuning**

---

## Problem Definition

Restore proper capitalization to text that has been converted to all lowercase letters.

`Input: "hello world. this is john smith from apple inc. he lives in new york."`   
`Output:"Hello world. This is John Smith from Apple Inc. He lives in New York."`

## Three Approaches

### 1\. Rule-Based POS Tagging (Baseline)

Traditional approach using linguistic rules and part-of-speech patterns. Serves as a performance baseline with minimal computational requirements.  
Implementation Steps:

1. Load spaCy model and define rule sets (abbreviations, special cases)  
2. Process text with POS tagger to identify word types  
3. Apply capitalization rules based on word position and POS tags  
4. Handle special formatting for punctuation and contractions

2\. BERT Encoder-Decoder (Strong Contender)  
Transformer-based approach leveraging BERT's contextual understanding. Expected to outperform rule-based methods through learned patterns.  
Implementation Steps:

1. Configure BERT encoder-decoder architecture  
2. Prepare dataset with lowercase/correct-case pairs  
3. Train the model with cross-entropy loss  
4. Fine-tune until convergence  
5. Validate performance on test set

3\. T5 with LoRA (Expected Best Performance)  
State-of-the-art language model adapted for case correction. Parameter-efficient training expected to achieve optimal results with reasonable compute requirements.  
Implementation Steps:

1. Load pre-trained T5 model ("t5-small")  
2. Configure LoRA adapter with rank 8 for efficient training  
3. Preprocess data with "correct case for sentence:" prefix  
4. Train adapter layers while freezing base model  
5. Save LoRA adapter for inference
