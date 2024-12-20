from nltk.corpus import brown

# Loading the corpus
corpus = brown.words()

# Case folding and getting vocab
lower_case_corpus = [w.lower() for w in corpus]
vocab = set(lower_case_corpus)

print('CORPUS EXAMPLE: ' + str(lower_case_corpus[:30]) + '\n\n')
print('VOCAB EXAMPLE: ' + str(list(vocab)[:10]))


print('Total words in Corpus: ' + str(len(lower_case_corpus)))
print('Vocab of the Corpus: ' + str(len(vocab)))

bigram_counts = {}
trigram_counts = {}

# Sliding through corpus to get bigram and trigram counts
for i in range(len(lower_case_corpus) - 2):
    # Getting bigram and trigram at each slide
    bigram = (lower_case_corpus[i], lower_case_corpus[i + 1])
    trigram = (lower_case_corpus[i], lower_case_corpus[i + 1], lower_case_corpus[i + 2])

    # Keeping track of the bigram counts
    if bigram in bigram_counts.keys():
        bigram_counts[bigram] += 1
    else:
        bigram_counts[bigram] = 1

    # Keeping track of trigram counts
    if trigram in trigram_counts.keys():
        trigram_counts[trigram] += 1
    else:
        trigram_counts[trigram] = 1

print("Example, count for bigram ('the', 'king') is: " + str(bigram_counts[('the', 'king')]))


# Function takes sentence as input and suggests possible words that comes after the sentence
def suggest_next_word(input_, bigram_counts, trigram_counts, vocab, num=3):
    # Consider the last bigram of sentence
    tokenized_input = input_.lower().split()
    last_bigram = tokenized_input[-2:]

    # Calculating probability for each word in vocab
    vocab_probabilities = {}
    for vocab_word in vocab:
        test_trigram = (last_bigram[0], last_bigram[1], vocab_word)
        test_bigram = (last_bigram[0], last_bigram[1])

        test_trigram_count = trigram_counts.get(test_trigram, 0)
        test_bigram_count = bigram_counts.get(test_bigram, 0)

        probability = test_trigram_count / test_bigram_count
        vocab_probabilities[vocab_word] = probability

    # Sorting the vocab probability in descending order to get top probable words
    top_suggestions = sorted(vocab_probabilities.items(), key=lambda x: x[1], reverse=True)[:num]
    return top_suggestions

num = 5
print(suggest_next_word('I am the king', bigram_counts, trigram_counts, vocab, num))
print(suggest_next_word('I am the king of', bigram_counts, trigram_counts, vocab, num))
print(suggest_next_word('I am the king of france', bigram_counts, trigram_counts, vocab, num))
print(suggest_next_word('I am the king of france and', bigram_counts, trigram_counts, vocab, num))
