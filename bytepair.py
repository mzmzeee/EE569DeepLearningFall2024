from collections import Counter, defaultdict
import re

def split_text_into_words(text):
    """Splits text into words, treating punctuation as separate words.

    Args:
        text: The input text string.

    Returns:
        A list of words, including punctuation marks as separate elements.
        Returns an empty list if the input is None or an empty string.
    """
    if not text:  # Check for None or empty string
        return []

    # Use a regular expression to split the text
    # \w+ matches one or more word characters (letters, numbers, underscore)
    # [^\w\s]+ matches one or more characters that are NOT word characters or whitespace
    # | is the "or" operator
    words = re.findall(r"\w+|[^\w\s]+|\s+", text)

    # Filter out empty strings that may result from multiple spaces
    words = [word for word in words if word.strip()]
    return words

class BytePairTokenizer:
    def __init__(self, vocab_size=50):
        """
        Initialize the Byte Pair Encoding tokenizer.

        Args:
            vocab_size (int): The maximum size of the vocabulary.
        """
        self.vocab_size = vocab_size
        self.vocab = {}
        self.bpe_codes = {}

    def get_vocab(self, corpus):
        """
        Create the initial vocabulary (character-level with word boundaries).

        Args:
            corpus (list of str): The input text corpus.

        Returns:
            dict: Token frequencies in the corpus.
        """
        vocab = Counter()
        for word in corpus:
            word = " ".join(list(word)) + " </w>"  # Add word boundary marker
            vocab[word] += 1
        return vocab

    def get_stats(self, vocab):
        """
        Count the frequency of each pair of symbols in the vocabulary.

        Args:
            vocab (dict): The current vocabulary.

        Returns:
            dict: Pair frequencies.
        """
        pairs = defaultdict(int)
        for word, freq in vocab.items():
            symbols = word.split()
            for i in range(len(symbols) - 1):
                pairs[symbols[i], symbols[i + 1]] += freq
        return pairs

    def merge_vocab(self, pair, vocab):
        """
        Merge the most frequent pair in the vocabulary.

        Args:
            pair (tuple): Pair of symbols to merge.
            vocab (dict): The current vocabulary.

        Returns:
            dict: Updated vocabulary.
        """
        bigram = re.escape(" ".join(pair))
        pattern = re.compile(r"(?<!\S)" + bigram + r"(?!\S)")
        new_vocab = {}
        for word in vocab:
            new_word = pattern.sub("".join(pair), word)
            new_vocab[new_word] = vocab[word]
        return new_vocab

    def fit(self, corpus):
        """
        Train the BPE tokenizer on the given corpus.

        Args:
            corpus (list of str): The input text corpus.
        """
        self.vocab = self.get_vocab(corpus)
        for _ in range(self.vocab_size - len(self.vocab)):
            pairs = self.get_stats(self.vocab)
            if not pairs:
                break
            best_pair = max(pairs, key=pairs.get)
            self.bpe_codes[best_pair] = len(self.bpe_codes)
            self.vocab = self.merge_vocab(best_pair, self.vocab)

    def encode(self, word):
        """
        Tokenize a word using the trained BPE codes.

        Args:
            word (str): The input word.

        Returns:
            list of str: Tokenized word.
        """
        word = " ".join(list(word)) + " </w>"
        symbols = word.split()
        while True:
            pairs = [(symbols[i], symbols[i + 1]) for i in range(len(symbols) - 1)]
            if not pairs:
                break
            pair_to_merge = None
            for pair in pairs:
                if pair in self.bpe_codes:
                    pair_to_merge = pair
                    break
            if not pair_to_merge:
                break
            bigram = re.escape(" ".join(pair_to_merge))
            pattern = re.compile(r"(?<!\S)" + bigram + r"(?!\S)")
            word = pattern.sub("".join(pair_to_merge), " ".join(symbols))
            symbols = word.split()
        return symbols

    def decode(self, tokens):
        """
        Decode a list of tokens back into a word.

        Args:
            tokens (list of str): The BPE tokens.

        Returns:
            str: Decoded word.
        """
        return "".join(tokens).replace("</w>", "")

    def get_bpe_codes(self):
        """
        Get the BPE codes generated during training.

        Returns:
            dict: The BPE codes.
        """
        return self.bpe_codes
# Example usage
text = """
The cat sleeps in the house 
The big cat sees the small bird 
The dog runs all day
The dog eats food in the house 
A bird flies in the sky
The bird sleeps at night
The sun shines on the trees
The sun shines on the house
"""
print(split_text_into_words(text))

# Example corpus
corpus = split_text_into_words(text.lower())

# Initialize the BPE tokenizer with a vocabulary size of 20
tokenizer = BytePairTokenizer(vocab_size=36)

# Train the tokenizer on the corpus
tokenizer.fit(corpus)

# Print the vocabulary after training
print("Vocabulary:")
for token, freq in tokenizer.vocab.items():
    print(f"{token}: {freq}")

# Print the BPE codes
print("\nBPE Codes:")
for pair, index in tokenizer.get_bpe_codes().items():
    print(f"{pair}: {index}")

# Tokenize words using the trained tokenizer
word = "birds"
tokens = tokenizer.encode(word)
print(f"\nTokenized '{word}': {tokens}")

# Decode tokens back into the original word
decoded_word = tokenizer.decode(tokens)
print(f"Decoded tokens: {decoded_word}")

# Tokenize another word
word = "dog"
tokens = tokenizer.encode(word)
print(f"\nTokenized '{word}': {tokens}")

# Decode tokens back into the original word
decoded_word = tokenizer.decode(tokens)
print(f"Decoded tokens: {decoded_word}")





