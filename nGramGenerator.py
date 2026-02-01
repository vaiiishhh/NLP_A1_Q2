import unicodedata

ZERO_WIDTH_CHARS = {
    '\u200b',  # zero width space
    '\u200c',  # ZWNJ
    '\u200d',  # ZWJ
    '\ufeff'   # BOM
}

def preprocess(line):
    line = unicodedata.normalize("NFKC", line.strip())
    line = "".join(ch for ch in line if ch not in ZERO_WIDTH_CHARS)
    tokens = []
    current = ""
    for ch in line:
        if unicodedata.category(ch).startswith("P"):
            if current:
                tokens.append(current)
                current = ""
            tokens.append(ch)
        elif ch.isspace():
            if current:
                tokens.append(current)
                current = ""
        else:
            current += ch
    if current:
        tokens.append(current)
    return tokens

# Generate FastText n-grams (3–5 + full word)
def fasttext_ngrams(word, min_n=3, max_n=5):
    word = f"<{word}>"
    L = len(word)
    ngrams = []
    for n in range(min_n, max_n + 1):
        for i in range(L - n + 1):
            ngrams.append(word[i:i + n])
    ngrams.append(word)
    return ngrams


# Build global n-gram vocabulary
def build_vocab(lines):
    vocab = set()
    for line in lines:
        for token in preprocess(line):
            vocab.update(fasttext_ngrams(token))
    return {ng: i for i, ng in enumerate(sorted(vocab))}

# Convert corpus to n-gram ID bags
def to_ngram_bags(lines, vocab):
    output = []
    for line in lines:
        bags = []
        for token in preprocess(line):
            ids = sorted(vocab[ng] for ng in fasttext_ngrams(token))
            bags.append(",".join(map(str, ids)))
        output.append(" ".join(bags))
    return output

# Example
# if __name__ == "__main__":
#     corpus = ["लेकिन मुश्किल तब होती है, जब आपका पार्टनर या भाई-बहन अच्छा नहीं कर रहे होते हैं।"]

#     vocab = build_vocab(corpus)
#     bags = to_ngram_bags(corpus, vocab)

#     print("Vocabulary:", (vocab))
#     print("Output:")
#     print(bags[0])

def main():
    with open("input.txt", "r", encoding="utf-8") as f:
        lines = f.readlines()

    vocab = build_vocab(lines)
    output = to_ngram_bags(lines, vocab)
    print("Vocabulary size:", len(vocab))

    with open("output.txt", "w", encoding="utf-8") as f:
        for line in output:
            f.write(line + "\n")


if __name__ == "__main__":
    main()