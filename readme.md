# **Testing 3 Ways to Fix Lowercase Text- Rule Based, Transformer, and Fine-Tuning**

---

## Problem Definition

Restore proper capitalization to text that has been converted to all lowercase letters.

`Input: "hello world. this is john smith from apple inc. he lives in new york."`   
`Output:"Hello world. This is John Smith from Apple Inc. He lives in New York."`

## **Approaches**

| Approach 1: Rule-Based POS Tagging (Baseline) We leverage spaCy's part-of-speech tagger to classify words and apply grammatical rules for capitalization. This traditional approach serves as our performance baseline while maintaining fast, deterministic processing. Key Benefits Zero training required – deploy immediately Fully transparent decisions – debug easily Lightning fast inference – no GPU needed Consistent results every time Implementation Steps Load spaCy model and define rule sets Process text with POS tagger to identify word types Apply capitalization rules based on word position and POS tags Handle special formatting for punctuation and contractions | Approach 2: BERT Encoder-Decoder (Strong Contender) We construct a sequence-to-sequence model by combining two BERT models: one as the encoder and one as the decoder. This architecture learns to transform lowercase text into correctly capitalized text through end-to-end training. Key Benefits Learns complex patterns automatically Understands full sentence context Adapts to different text domains Leverages BERT's language knowledge for both encoding and decoding Implementation Steps Configure BERT-to-BERT encoder-decoder architecture Prepare dataset with lowercase/correct-case pairs Train model with cross-entropy loss Fine-tune until convergence Validate on test set  | Approach 3: T5 with LoRA (Expected Best Performance) We adapt the state-of-the-art T5 model for case correction using Low-Rank Adaptation, an efficient fine-tuning technique. This method promises optimal results while minimizing computational requirements. Key Benefits Inherits T5's powerful language knowledge Trains only 0.1% of model parameters Reduces training costs by \~80% Easily adapts to new domains Superior generalization to novel patterns Implementation Steps Load pre-trained T5-small model Configure LoRA adapter with rank 8 Preprocess data with "correct case for sentence:" prefix Train adapter while freezing base model Save LoRA adapter for inference  |
| :---- | :---- | :---- |
