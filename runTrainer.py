import torch
import argparse
import json
import sys
from nGramGenerator import preprocess, fasttext_ngrams, build_vocab
from trainer import train_fasttext, save_model


def get_device(gpu_id=None):
    """Automatically detect and return the best available device"""
    if gpu_id is not None:
        device = f'cuda:{gpu_id}'
        if not torch.cuda.is_available():
            print(f"Warning: GPU {gpu_id} requested but CUDA not available. Using CPU.")
            return 'cpu'
    elif torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'

    if device.startswith('cuda'):
        gpu_name = torch.cuda.get_device_name(device)
        gpu_memory = torch.cuda.get_device_properties(device).total_memory / 1e9
        print(f"\n{'='*60}")
        print(f"GPU Detected: {gpu_name}")
        print(f"GPU Memory: {gpu_memory:.2f} GB")
        print(f"CUDA Version: {torch.version.cuda}")
        print(f"{'='*60}\n")
    else:
        print(f"\n{'='*60}")
        print("Running on CPU")
        print(f"{'='*60}\n")

    return device


def main():
    parser = argparse.ArgumentParser(
        description='Train FastText model with GPU support',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # File paths
    parser.add_argument('--input', type=str, default='input.txt',
                        help='Input corpus file')
    parser.add_argument('--output', type=str, default='fasttext_model.pt',
                        help='Output model file')
    parser.add_argument('--vocab-output', type=str, default='ngram_vocab.txt',
                        help='Save n-gram vocabulary to file')

    # Model hyperparameters
    parser.add_argument('--dim', type=int, default=100,
                        help='Embedding dimension (50-300 recommended)')
    parser.add_argument('--epochs', type=int, default=5,
                        help='Number of training epochs')
    parser.add_argument('--window', type=int, default=2,
                        help='Context window size')
    parser.add_argument('--neg', type=int, default=5,
                        help='Number of negative samples')
    parser.add_argument('--lr', type=float, default=0.025,
                        help='Learning rate')

    # GPU settings
    parser.add_argument('--batch-size', type=int, default=None,
                        help='Batch size (auto-set based on device if not specified)')
    parser.add_argument('--gpu', type=int, default=None,
                        help='GPU ID to use (default: auto-detect)')
    parser.add_argument('--no-amp', action='store_true',
                        help='Disable automatic mixed precision (FP16)')
    parser.add_argument('--cpu', action='store_true',
                        help='Force CPU even if GPU is available')

    # Data processing
    parser.add_argument('--subsample', type=float, default=1e-5,
                        help='Subsampling threshold for frequent words')

    args = parser.parse_args([]) # Pass an empty list to parse_args() to avoid kernel arguments

    # Determine device
    if args.cpu:
        device = 'cpu'
        print("\nForced CPU mode\n")
    else:
        device = get_device(args.gpu)

    # Auto-set batch size based on device
    if args.batch_size is None:
        if device == 'cpu':
            args.batch_size = 512
            print(f"Auto-set batch size for CPU: {args.batch_size}")
        else:
            # Check GPU memory and set appropriate batch size
            gpu_memory_gb = torch.cuda.get_device_properties(device).total_memory / 1e9
            if gpu_memory_gb >= 16:
                args.batch_size = 4096
            elif gpu_memory_gb >= 8:
                args.batch_size = 2048
            else:
                args.batch_size = 1024
            print(f"Auto-set batch size for GPU ({gpu_memory_gb:.1f}GB): {args.batch_size}")

    use_amp = (device.startswith('cuda') and not args.no_amp)

    print(f"\nConfiguration Summary:")
    print(f"  Input file: {args.input}")
    print(f"  Output file: {args.output}")
    print(f"  Device: {device}")
    print(f"  Mixed Precision: {use_amp}")
    print(f"  Batch Size: {args.batch_size}")
    print(f"  Embedding Dim: {args.dim}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Window Size: {args.window}")
    print(f"  Negative Samples: {args.neg}")
    print(f"  Learning Rate: {args.lr}\n")

    # -----------------------------
    # 1. Read corpus
    # -----------------------------
    print(f"{'='*60}")
    print("Step 1/6: Reading corpus")
    print(f"{'='*60}")
    try:
        with open(args.input, "r", encoding="utf-8") as f:
            raw_lines = f.readlines()
    except FileNotFoundError:
        print(f"Error: {args.input} not found!")
        sys.exit(1)

    print(f"Read {len(raw_lines):,} lines from {args.input}")

    # -----------------------------
    # 2. Tokenize sentences
    # -----------------------------
    print(f"\n{'='*60}")
    print("Step 2/6: Tokenizing sentences")
    print(f"{'='*60}")
    sentences = []
    for i, line in enumerate(raw_lines):
        tokens = preprocess(line)
        if len(tokens) > 0:
            sentences.append(tokens)
        if (i + 1) % 10000 == 0:
            print(f"Processed {i + 1:,}/{len(raw_lines):,} lines")

    print(f"Tokenized into {len(sentences):,} non-empty sentences")

    # Calculate vocabulary statistics
    all_words = [w for sent in sentences for w in sent]
    unique_words = set(all_words)
    print(f"Total tokens: {len(all_words):,}")
    print(f"Unique words: {len(unique_words):,}")

    # -----------------------------
    # 3. Build n-gram vocabulary
    # -----------------------------
    print(f"\n{'='*60}")
    print("Step 3/6: Building n-gram vocabulary")
    print(f"{'='*60}")
    ngram_to_id = build_vocab(raw_lines)
    print(f"N-gram vocabulary size: {len(ngram_to_id):,}")

    # Save n-gram vocabulary
    print(f"Saving n-gram vocabulary to {args.vocab_output}...")
    with open(args.vocab_output, "w", encoding="utf-8") as f:
        for ng, idx in sorted(ngram_to_id.items(), key=lambda x: x[1]):
            f.write(f"{idx}\t{ng}\n")

    # -----------------------------
    # 4. Build word vocabulary
    # -----------------------------
    print(f"\n{'='*60}")
    print("Step 4/6: Building word vocabulary")
    print(f"{'='*60}")
    words = sorted(unique_words)
    word_to_id = {w: i for i, w in enumerate(words)}
    id_to_word = {i: w for w, i in word_to_id.items()}
    print(f"Word vocabulary size: {len(words):,}")

    # -----------------------------
    # 5. Map words to n-grams
    # -----------------------------
    print(f"\n{'='*60}")
    print("Step 5/6: Mapping words to n-grams")
    print(f"{'='*60}")
    word_to_ngrams = {}
    for i, word in enumerate(words):
        ngrams = fasttext_ngrams(word)
        word_to_ngrams[word] = [ngram_to_id[ng] for ng in ngrams]

        if (i + 1) % 5000 == 0 or (i + 1) == len(words):
            print(f"Processed {i + 1:,}/{len(words):,} words")

    # -----------------------------
    # 6. Train FastText
    # -----------------------------
    print(f"\n{'='*60}")
    print("Step 6/6: Training FastText model")
    print(f"{'='*60}")

    # Clear GPU cache if using CUDA
    if device.startswith('cuda'):
        torch.cuda.empty_cache()
        print("Cleared GPU cache")

    model, loss_history = train_fasttext(
        sentences=sentences,
        word_to_id=word_to_id,
        id_to_word=id_to_word,
        word_to_ngrams=word_to_ngrams,
        ngram_vocab_size=len(ngram_to_id),
        embedding_dim=args.dim,
        window_size=args.window,
        num_negatives=args.neg,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        device=device,
        use_amp=use_amp
    )

    # -----------------------------
    # 7. Save model and metadata
    # -----------------------------
    print(f"\n{'='*60}")
    print("Saving model and metadata")
    print(f"{'='*60}")

    save_model(model, word_to_id, ngram_to_id, loss_history, args.output)

    # Save loss history for plotting
    loss_file = args.output.replace('.pt', '_loss.json')
    with open(loss_file, "w") as f:
        json.dump({
            "epochs": list(range(1, len(loss_history) + 1)),
            "loss": loss_history,
            "config": {
                "embedding_dim": args.dim,
                "window_size": args.window,
                "num_negatives": args.neg,
                "batch_size": args.batch_size,
                "learning_rate": args.lr,
                "device": device,
                "mixed_precision": use_amp
            }
        }, f, indent=2)
    print(f"Loss history saved to {loss_file}")

    # Save training statistics
    stats_file = args.output.replace('.pt', '_stats.txt')
    with open(stats_file, "w", encoding="utf-8") as f:
        f.write("FastText Training Statistics\n")
        f.write("="*60 + "\n\n")
        f.write(f"Corpus:\n")
        f.write(f"  Lines: {len(raw_lines):,}\n")
        f.write(f"  Sentences: {len(sentences):,}\n")
        f.write(f"  Total tokens: {len(all_words):,}\n")
        f.write(f"  Unique words: {len(unique_words):,}\n\n")
        f.write(f"Vocabularies:\n")
        f.write(f"  N-gram vocabulary: {len(ngram_to_id):,}\n")
        f.write(f"  Word vocabulary: {len(words):,}\n\n")
        f.write(f"Model Configuration:\n")
        f.write(f"  Embedding dimension: {args.dim}\n")
        f.write(f"  Context window: {args.window}\n")
        f.write(f"  Negative samples: {args.neg}\n")
        f.write(f"  Epochs: {args.epochs}\n")
        f.write(f"  Learning rate: {args.lr}\n")
        f.write(f"  Batch size: {args.batch_size}\n\n")
        f.write(f"Training:\n")
        f.write(f"  Device: {device}\n")
        f.write(f"  Mixed precision: {use_amp}\n")
        f.write(f"  Final loss: {loss_history[-1]:.4f}\n")
    print(f"Training statistics saved to {stats_file}")

    # Print final summary
    print(f"\n{'='*60}")
    print("Training Complete!")
    print(f"{'='*60}")
    print(f"Model saved to: {args.output}")
    print(f"Loss history: {loss_file}")
    print(f"Statistics: {stats_file}")
    print(f"N-gram vocab: {args.vocab_output}")
    print(f"\nFinal training loss: {loss_history[-1]:.4f}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()