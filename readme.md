# Comparing Methods for Automated Text Capitalization Restoration

## Project Overview

This project evaluates three different approaches to restoring proper capitalization to lowercase text. The goal is to convert text like:

```
hello world. this is john smith from apple inc. he lives in new york.
```

Into properly capitalized text:

```
Hello world. This is John Smith from Apple Inc. He lives in New York.
```

All approaches were tested and implemented within a 2-day timeframe without using large LLMs directly, focusing on practical learning outcomes.

## Approaches Tested
![Alt text](https://raw.githubusercontent.com/kamalkantsingh10/CaCoMo/refs/heads/main/CaCoMo.png)
### 1. Rule-Based POS Tagging

This approach uses spaCy's part-of-speech tagger to classify words and applies grammatical rules for capitalization. It serves as our performance baseline.

**Advantages:**
- No training required - deploy immediately
- Fully transparent decisions - easy debugging
- Lightning fast inference - no GPU needed
- Consistent results every time

**Disadvantages:**
- Cannot adapt to new patterns without manual rule updates
- Struggles with ambiguous cases and context-dependent decisions
- Requires extensive manual effort to handle edge cases

### 2. BERT-to-BERT Encoder-Decoder

This approach combines two BERT models (one as encoder, one as decoder) to create a sequence-to-sequence model that learns to transform lowercase text into correctly capitalized text.

**Advantages:**
- Learns complex patterns automatically
- Understands full sentence context
- Adapts to different text domains
- Leverages BERT's language knowledge for both encoding and decoding

**Disadvantages:**
- Computationally expensive - requires resources for training
- Needs large labeled datasets for effective training
- Slower inference compared to rule-based approach
- Memory intensive - two full BERT models in one architecture

### 3. Fine-Tuning T5 with LoRA

This approach adapts the T5 model for case correction using Low-Rank Adaptation (LoRA), an efficient fine-tuning technique that modifies only a small portion of the model parameters.

**Advantages:**
- Inherits T5's powerful language knowledge
- Trains only 0.1% of model parameters
- Reduces training costs by ~80%
- Easily adapts to new domains
- Superior generalization to novel patterns

**Disadvantages:**
- Still requires labeled training data
- Moderate inference latency compared to rules-based approach

## Training Data

Text from Project Gutenberg books was used to create the case correction dataset:

1. Downloaded plain text books from Project Gutenberg
2. Parsed text into individual sentences
3. Converted to lowercase for training inputs
4. Used original sentences as correction targets
5. Split into 12,500 training, 2,500 validation, and 3,000 testing sentences

**Example:**
- **Input:** `she went with some trepidation, and was not unpleasantly surprised, and more than a little nervous, when she found that he was not so inaccessible as his name and position seemed to indicate.`
- **Target:** `She went with some trepidation, and was not unpleasantly surprised, and more than a little nervous, when she found that he was not so inaccessible as his name and position seemed to indicate.`

The same training and testing data was used across all approaches to ensure fair comparison.

## Implementation Process

Each approach followed a specific implementation path:

1. **Rule-Based POS Tagging:**
   - Loaded spaCy model and defined rule sets
   - Processed text with POS tagger to identify word types
   - Applied capitalization rules based on word position and POS tags
   - Handled special formatting for punctuation and contractions

2. **BERT-to-BERT Encoder-Decoder:**
   - Configured BERT-to-BERT encoder-decoder architecture
   - Prepared dataset with lowercase/correct-case pairs
   - Trained model with cross-entropy loss
   - Fine-tuned until convergence
   - Validated on test set

3. **T5 with LoRA:**
   - Loaded pre-trained T5-small model
   - Configured LoRA adapter with rank 8
   - Preprocessed data with "correct case for sentence:" prefix
   - Trained adapter while freezing base model
   - Saved LoRA adapter for inference

## Training Comparison

| Metric | Rule-Based POS Tagging | BERT-to-BERT Encoder-Decoder | T5 with LoRA |
|--------|------------------------|------------------------------|--------------|
| Training dataset | N/A - No training required | 7,367 sentences | 7,867 sentences |
| Validation dataset | N/A | 2,500 sentences | 2,500 sentences |
| Training time | N/A (but handling additional rules took 4 hours) | ~4 hours | ~8.5 hours |
| Epochs | N/A | 3 | 3 |
| Hardware used | CPU only | CPU only | CPU only |
| CPU load while training | N/A | Medium | Medium to heavy |

## Results

After 20 hours of development and testing over a weekend:

| Metric | Rule-Based POS Tagging | BERT-to-BERT Encoder-Decoder | Fine Tuning T5 with LoRA |
|--------|------------------------|------------------------------|---------------------------|
| Model size (final) | ~15MB (spaCy model) | ~985 MB | ~10MB (adapter only) |
| Time to process 3000 tests | 1 second | 8 minutes | 10 minutes |
| CPU usage while testing | Almost nil | Medium | Medium to heavy |
| Sentences failed to correct (out of 3,001) | 1,544 | 3,001 | 836 |
| Accuracy | 48.5% | 0% | 72.14% |

### Key Findings

1. **Rule-Based POS Tagging:**
   - Easy to implement initially but limited
   - Started at 10% accuracy, improved to 48.5% after adding more rules
   - Further improvements possible but would be increasingly hardcoded

2. **BERT-to-BERT:**
   - Completely failed (0% accuracy), unexpectedly
   - Likely needed more training data and epochs (50-60)
   - Capitalized very few characters and produced altered text

3. **T5 with LoRA:**
   - Best performer with 72.14% accuracy on first attempt
   - Original T5-small model could not correct any sentences
   - Promising results that could improve with more training data and tuning

### Example Comparison

**Input:**
```
chapter i introduction to storm probably susan hawthorne got a lot of her courage and independence from her father, old smiler hawthorne, who in his time had been nearly everything and nearly everywhere--a tall, grizzled man who was generally broke but always unbeatable.
```

**Expected Output:**
```
CHAPTER I INTRODUCTION TO STORM Probably Susan Hawthorne got a lot of her courage and independence from her father, old Smiler Hawthorne, who in his time had been nearly everything and nearly everywhere--a tall, grizzled man who was generally broke but always unbeatable.
```

**T5 with LoRA Output:**
```
Chapter I Introduction to Storm probably Susan Hawthorne got a lot of her courage and independence from her father, old Smiler Hawthorne, who in his time had been nearly everything and nearly everywhere--a tall, grizzled man who was generally broke but always unbeatable.
```

## Future Improvements

1. Debug the BERT-to-BERT approach to understand the complete failure (0% accuracy)
2. Perform more iterations for the T5 with LoRA approach to potentially reach >90% accuracy
3. Create a Streamlit frontend to demonstrate the models in action
4. Test with more diverse text types beyond literary sources
5. Explore hybrid approaches that combine rule-based efficiency with neural model accuracy

## Conclusion

T5 with LoRA emerged as the clear winner, achieving 72.14% accuracy while maintaining a compact model size of just 10MB (adapter only). This lightweight approach outperformed both the traditional rule-based system and the larger neural network architecture, demonstrating the effectiveness of efficient fine-tuning techniques for NLP tasks with limited computational resources.
