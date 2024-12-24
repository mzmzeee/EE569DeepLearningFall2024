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
The Prophet Muhammad (peace be upon him), the final prophet in Islam, was born in Mecca around 570 CE into the respected Quraysh tribe. His father, Abdullah, died before his birth, and his mother, Amina, passed away when he was six. He was then raised by his grandfather, Abd al-Muttalib, and after his death, by his uncle, Abu Talib.
Known for his honesty and integrity, Muhammad earned the title "al-Amin" (the trustworthy one). At the age of 25, he married Khadija, a respected women, and they had several children.
Around 610 CE, at the age of 40, while meditating in a cave on Mount Hira, Muhammad received his first revelation from God through the angel Gabriel. These revelations continued for the rest of his life and were memorized and written down by his followers, forming the Quran, the holy book of Islam.
Muhammad began preaching his message of monotheism (the belief in one God, Allah) in Mecca, calling people to abandon idolatry and submit to God. His message was met with resistance and persecution from the Meccan elite, who feared the social and economic changes his teachings threatened.
In 622 CE, facing increasing hostility, Muhammad and his followers migrated to Medina, an event known as the Hijra (migration). In Medina, he established a Muslim community and became a political and religious leader. The Muslim community grew, and conflicts arose with the Meccans.
Over several years, battles were fought between the Muslims of Medina and the Meccans. Eventually, in 630 CE, Muhammad returned to Mecca with a large army, and the city surrendered peacefully. He forgave his former persecutors and cleansed the Kaaba, the central sanctuary in Mecca, of idols, dedicating it to the worship of one God.
In the final years of his life, Muhammad consolidated Islam's position in Arabia, uniting most of the peninsula under the new faith. He established principles of governance, justice, and social welfare based on divine guidance.
In 632 CE, Muhammad performed his Farewell Pilgrimage to Mecca, where he delivered his final sermon, emphasizing the importance of unity, equality, and adherence to the Quran and his teachings. Shortly after returning to Medina, he fell ill and passed away. He is buried in Medina, in the Prophet's Mosque, which remains a site of pilgrimage for Muslims worldwide.
Muhammad's life and teachings have had a profound impact on the world, shaping the lives of billions of Muslims across centuries and continents. His example as a prophet, leader, and moral guide continues to inspire and influence people today."""
print(split_text_into_words(text))

# Example corpus
corpus = split_text_into_words(text.lower())

# Initialize the BPE tokenizer with a vocabulary size of 20
tokenizer = BytePairTokenizer(vocab_size=500)

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
word = "abumuhammad"
tokens = tokenizer.encode(word)
print(f"\nTokenized '{word}': {tokens}")

# Decode tokens back into the original word
decoded_word = tokenizer.decode(tokens)
print(f"Decoded tokens: {decoded_word}")

# Tokenize another word
word = "following"
tokens = tokenizer.encode(word)
print(f"\nTokenized '{word}': {tokens}")

# Decode tokens back into the original word
decoded_word = tokenizer.decode(tokens)
print(f"Decoded tokens: {decoded_word}")





