# **Testing 3 Ways to Fix Lowercase Text- Rule Based, Transformer, and Fine-Tuning**

---

## Problem Definition

Restore proper capitalization to text that has been converted to all lowercase letters.

`Input: hello world. this is john smith from apple inc. he lives in new york.`  
`Output: Hello world. This is John Smith from Apple Inc. He lives in New York.`

## Approaches tested
![Cover](https://github.com/kamalkantsingh10/CaCoMo/blob/main/CaCoMo.png?raw=true)
**Approach 1: Rule-Based POS Tagging**  
We leverage spaCy's part-of-speech tagger to classify words and apply grammatical rules for capitalization. This traditional approach serves as our performance baseline while maintaining fast, deterministic processing.

**Pro:-** Zero training required – deploy immediately; Fully transparent decisions – debug easily; Lightning fast inference – no GPU needed; Consistent results every time  
**Con:-** Cannot adapt to new patterns without manual rule updates; Struggles with ambiguous cases and context-dependent decisions; Requires extensive manual effort to handle edge cases

### **Approach 2: BERT-to-BERT Encoder-Decoder** 

We construct a sequence-to-sequence model by combining two BERT models: one as the encoder and one as the decoder. This architecture learns to transform lowercase text into correctly capitalized text through end-to-end training.

**Pro:-** earns complex patterns automatically; Understands full sentence context; Adapts to different text domains; Leverages BERT's language knowledge for both encoding and decoding  
**Con:-**  Computationally expensive – requires resources for training; Needs large labeled datasets for effective training; Slower inference compared to rule-based approach; Memory intensive – two full BERT models in one architecture

### **Approach 3: T5 with LoRA (Expected Best Performance)**

We adapt the state-of-the-art T5 model for case correction using Low-Rank Adaptation, an efficient fine-tuning technique. This method promises optimal results while minimizing computational requirem  
ents.

**Pro:-** Inherits T5's powerful language knowledge;  Trains only 0.1% of model parameters; Reduces training costs by \~80%; Easily adapts to new domains; Superior generalization to novel patterns  
**Con:-** Still requires labeled training data; Moderate inference latency compared to rule**s**

## Training Data

We sourced text from Project Gutenberg books to create a case correction dataset.

**Data Preparation Steps**

1. Download plain text books from [Project Gutenberg](https://gutenberg.org/)  
2. Parse text into individual sentences  
3. Convert to lowercase for training inputs  
4. Use original sentences as correction targets  
5. Split: 12,500 training, 2,500 validation, 3,000 testing

**Example:**

**`Input`**`:she went with some trepidation, and was not unpleasantly surprised, and more than a little nervous, when she found that he was not so inaccessible as his name and position seemed to indicate.`  
**`Target`**`:She went with some trepidation, and was not unpleasantly surprised, and more than a little nervous, when she found that he was not so inaccessible as his name and position seemed to indicate.`

**Key Points**

* Shared training/testing data across all approaches   
* Created custom Python class for data processing  
* Ensures fair comparison between models as same testing data will be used


`#code:`  
`from source.dataset_builder import create_training_dataset`  
`create_training_dataset(“training_full_model”) #for approach 2`  
`create_training_dataset(“finetune”) #for approach 3`

* Training dataset for [approach 2](https://github.com/kamalkantsingh10/CaCoMo/blob/main/files/dataset/full_model/training.txt) and [approach 3](https://github.com/kamalkantsingh10/CaCoMo/blob/main/files/dataset/finetune/training.txt)  
* [Validation dataset](https://github.com/kamalkantsingh10/CaCoMo/blob/main/files/dataset/full_model/validation.txt)  
* [Testing dataset](https://github.com/kamalkantsingh10/CaCoMo/blob/main/files/dataset/testing.txt)

