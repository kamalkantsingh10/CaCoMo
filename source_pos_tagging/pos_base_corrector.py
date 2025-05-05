import string
try:
    import spacy
    nlp = spacy.load("en_core_web_sm")
except ModuleNotFoundError:
    print("Please install spacy and download the model:")
    print("pip install spacy")
    print("python -m spacy download en_core_web_sm")
    raise
except OSError:
    print("Please download the spacy model:")
    print("python -m spacy download en_core_web_sm")
    raise


class POS_Tag_CaseCorrecter:
    """
    A class to format sentences using POS tagging.
    Converts lowercase text to properly formatted sentences.
    Uses spaCy for improved accuracy.
    """
    
    def __init__(self):
        """Initialize the formatter with common abbreviations."""
        self.abbreviations = {
            'dr', 'mr', 'mrs', 'ms', 'prof', 'sr', 'jr', 
            'phd', 'md', 'ba', 'ma', 'dds', 'llb', 'etc',
            'vs', 'inc', 'ltd', 'corp', 'co'
        }
        
        # Special cases that should stay lowercase
        self.keep_lowercase = {
            'iphone', 'ipad', 'ebay', 'icloud', 'ios', 'android',
            'a', 'an', 'the', 'and', 'or', 'but', 'for', 'nor', 'so', 'yet'
        }
        
        # Words that should always be capitalized
        self.always_capitalize = {'i'}
    
    def format_text(self, text):
        """
        Format the input text into properly formatted sentences.
        
        Args:
            text (str): Input text in lowercase
            
        Returns:
            str: Properly formatted text
        """
        if not text:
            return ""
        
        # Process the text with spaCy
        doc = nlp(text.strip())
        
        formatted_sentences = []
        for sent in doc.sents:
            formatted_sentence = self._format_sentence(sent)
            formatted_sentences.append(formatted_sentence)
        
        return ' '.join(formatted_sentences)
    
    def _format_sentence(self, sentence):
        """
        Format a single sentence using POS tagging.
        
        Args:
            sentence (spacy.tokens.span.Span): A spaCy sentence span
            
        Returns:
            str: Formatted sentence
        """
        formatted_words = []
        
        for i, token in enumerate(sentence):
            formatted_word = self._format_word(token, i, sentence)
            formatted_words.append(formatted_word)
        
        # Join words and handle punctuation spacing
        result = self._join_words(formatted_words)
        
        # Ensure sentence starts with capital
        if result:
            result = result[0].upper() + result[1:]
        
        return result
    
    def _format_word(self, token, position, sentence):
        """
        Format a single word based on its POS tag and context.
        
        Args:
            token (spacy.tokens.token.Token): The spaCy token
            position (int): Position in the sentence
            sentence (spacy.tokens.span.Span): The sentence span
            
        Returns:
            str: Formatted word
        """
        word = token.text
        
        # Handle contractions - check if this is a contraction part
        if position > 0 and sentence[position - 1].text + token.text in ["don't", "can't", "won't", "it's", "he's", "she's", "isn't", "aren't"]:
            # This is a contraction part, directly append to previous
            return token.text.lower()
        
        # Only convert to lowercase if not a contraction part
        if token.text not in ["n't", "'s", "'ll", "'re", "'ve", "'d"]:
            word = word.lower()
        
        # Handle specific cases
        if position == 0:
            # First word, capitalize unless it's a special case
            if word in self.keep_lowercase:
                return word
            else:
                return word.capitalize()
        
        # Check if this word follows an opening quote
        if position > 0:
            prev_token = sentence[position - 1]
            if prev_token.text in ['"', "'"]:
                # Capitalize first word after quote
                return word.capitalize()
        
        # Proper nouns
        if token.pos_ == 'PROPN':
            return word.capitalize()
        
        # Check for abbreviations
        if self._is_abbreviation(word):
            return word.capitalize()
        
        # Check for words that should always be capitalized
        if word in self.always_capitalize:
            return word.capitalize()
        
        # Keep special cases lowercase
        if word in self.keep_lowercase:
            return word
        
        return word
    
    def _is_abbreviation(self, word):
        """Check if a word is an abbreviation."""
        word_lower = word.lower().rstrip('.')
        return word_lower in self.abbreviations
    
    def _join_words(self, words):
        """
        Join words with proper spacing around punctuation.
        
        Args:
            words (list): List of formatted words
            
        Returns:
            str: Joined sentence with proper spacing
        """
        if not words:
            return ""
        
        result = []
        i = 0
        
        while i < len(words):
            word = words[i]
            
            # Handle different punctuation cases
            if word == '-':
                # Handle hyphenated words (no spaces)
                if result and i < len(words) - 1:
                    # Attach hyphen to previous and next word
                    result[-1] += word
                    # Continue to next word without adding a space
                    i += 1
                    if i < len(words):
                        result[-1] += words[i]
                else:
                    result.append(word)
            elif word == '--':
                # Handle em dashes: attach to previous word
                if result:
                    result[-1] += word
                else:
                    result.append(word)
            elif word in ',.!?:;':
                # Punctuation that attaches to previous word
                if result:
                    result[-1] += word
                else:
                    result.append(word)
            elif word in '()[]{}':
                # Brackets and parentheses
                if word in '([{':
                    result.append(word)
                else:
                    if result:
                        result[-1] += word
                    else:
                        result.append(word)
            elif word in ['"', "'"]:
                # Handle quotes - no space before the quote
                result.append(word)
            elif word in ['``', "''"]:
                # Convert NLTK-style quotes to standard quotes
                std_quote = '"'
                result.append(std_quote)
            elif word in ["n't", "'s", "'ll", "'re", "'ve", "'d"]:
                # Handle contraction parts - attach to previous word
                if result:
                    result[-1] += word
                else:
                    result.append(word)
            else:
                # Regular word - add with space if not first element and previous wasn't em dash or quote
                if result and not result[-1].endswith('--') and result[-1] not in ['"', "'"]:
                    result.append(' ')
                result.append(word)
            
            i += 1
        
        return ''.join(result)