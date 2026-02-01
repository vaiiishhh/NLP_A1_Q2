import random
import math
from collections import Counter
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
from nGramGenerator import preprocess, fasttext_ngrams
import numpy as np
from tqdm import tqdm


# -----------------------------
# FastText Skip-gram Model
# -----------------------------
class FastTextSkipGram(nn.Module):
    def __init__(self, num_ngrams, num_words, dim):
        super().__init__()
        self.ngram_emb = nn.Embedding(num_ngrams, dim)
        self.word_emb = nn.Embedding(num_words, dim)

        nn.init.uniform_(self.ngram_emb.weight, -0.5 / dim, 0.5 / dim)
        nn.init.zeros_(self.word_emb.weight)

    def forward(self, center_ngram_ids, target_word_ids, negative_word_ids=None):
        """
        GPU-optimized batched forward pass
        center_ngram_ids: list of lists (variable length n-grams per word)
        target_word_ids: (batch_size,) tensor on device
        negative_word_ids: (batch_size, num_negatives) tensor on device
        """
        device = target_word_ids.device
        
        # Handle variable-length n-gram lists efficiently
        if isinstance(center_ngram_ids, list):
            center_vecs = []
            for ngram_list in center_ngram_ids:
                if len(ngram_list) == 0:
                    # Empty n-gram list - use zero vector
                    center_vecs.append(torch.zeros(self.ngram_emb.embedding_dim, device=device))
                else:
                    # Convert to tensor and move to device
                    ngram_tensor = torch.tensor(ngram_list, dtype=torch.long, device=device)
                    center_vecs.append(self.ngram_emb(ngram_tensor).sum(dim=0))
            center_vec = torch.stack(center_vecs)  # (batch_size, dim)
        else:
            center_vec = center_ngram_ids
        
        # Positive scores
        target_vec = self.word_emb(target_word_ids)  # (batch_size, dim)
        pos_scores = (center_vec * target_vec).sum(dim=1)  # (batch_size,)
        
        if negative_word_ids is not None:
            # Negative scores
            neg_vec = self.word_emb(negative_word_ids)  # (batch_size, num_neg, dim)
            # Efficient batch matrix multiplication
            neg_scores = torch.bmm(
                neg_vec, 
                center_vec.unsqueeze(2)
            ).squeeze(2)  # (batch_size, num_neg)
            return pos_scores, neg_scores
        
        return pos_scores


# -----------------------------
# Subsampling (Mikolov et al.)
# -----------------------------
def subsample_sentence(sentence, word_counts, total_words, t=1e-5):
    """Vectorized subsampling"""
    if not sentence:
        return []
    
    subsampled = []
    for w in sentence:
        f = word_counts.get(w, 1) / total_words
        p_discard = max(0.0, 1.0 - math.sqrt(t / f))
        if random.random() > p_discard:
            subsampled.append(w)
    return subsampled


# -----------------------------
# Negative sampling (GPU-OPTIMIZED)
# -----------------------------
class NegativeSampler:
    def __init__(self, word_freqs, power=0.75, device='cpu'):
        """
        Pre-compute sampling distribution
        word_freqs: dict mapping word_id -> frequency
        """
        ids = list(word_freqs.keys())
        freqs = np.array([word_freqs[i] for i in ids]) ** power
        self.probs = freqs / freqs.sum()
        self.ids = np.array(ids)
        self.device = device
    
    def sample(self, batch_size, num_negatives, forbidden_ids):
        """
        Sample negatives in batch (GPU-ready)
        forbidden_ids: (batch_size,) numpy array or tensor of context word IDs to avoid
        Returns: (batch_size, num_negatives) tensor on device
        """
        # Convert to numpy if tensor
        if isinstance(forbidden_ids, torch.Tensor):
            forbidden_ids = forbidden_ids.cpu().numpy()
        
        # Oversample to account for rejections
        samples = np.random.choice(
            self.ids, 
            size=(batch_size, num_negatives * 3),
            p=self.probs
        )
        
        # Filter out forbidden IDs
        result = np.zeros((batch_size, num_negatives), dtype=np.int64)
        for i in range(batch_size):
            valid = samples[i][samples[i] != forbidden_ids[i]]
            if len(valid) >= num_negatives:
                result[i] = valid[:num_negatives]
            else:
                # Need to resample (rare case)
                result[i][:len(valid)] = valid
                remaining = num_negatives - len(valid)
                extra_samples = np.random.choice(self.ids, size=remaining * 2, p=self.probs)
                extra_valid = extra_samples[extra_samples != forbidden_ids[i]]
                result[i][len(valid):] = extra_valid[:remaining]
        
        # Convert to tensor on device
        return torch.tensor(result, dtype=torch.long, device=self.device)


# -----------------------------
# Training Data Generator (BATCHED & GPU-OPTIMIZED)
# -----------------------------
def generate_training_pairs(sentences, word_to_id, word_to_ngrams, word_counts, 
                           total_words, window_size=2, t=1e-5):
    """
    Generate all training pairs at once for efficient batching
    Returns: list of (center_ngram_ids, context_id) tuples
    """
    pairs = []
    
    for sent in sentences:
        # Subsample
        sent = subsample_sentence(sent, word_counts, total_words, t)
        sent_ids = [word_to_id[w] for w in sent if w in word_to_id]
        
        if len(sent_ids) < 2:
            continue
        
        for i, center_id in enumerate(sent_ids):
            # Get center word and its n-grams
            center_word = None
            for w, wid in word_to_id.items():
                if wid == center_id:
                    center_word = w
                    break
            
            if center_word is None:
                continue
                
            center_ngrams = word_to_ngrams.get(center_word, [])
            
            # Dynamic window size (FastText paper recommendation)
            actual_window = random.randint(1, window_size)
            
            context_range = range(
                max(0, i - actual_window),
                min(len(sent_ids), i + actual_window + 1)
            )
            
            for j in context_range:
                if i != j:
                    context_id = sent_ids[j]
                    pairs.append((center_ngrams, context_id))
    
    return pairs


# -----------------------------
# Trainer (GPU-OPTIMIZED with Mixed Precision)
# -----------------------------
def train_fasttext(
    sentences,
    word_to_id,
    id_to_word,
    word_to_ngrams,
    ngram_vocab_size,
    embedding_dim=100,
    window_size=2,
    num_negatives=5,
    epochs=5,
    lr=0.025,
    batch_size=2048,  # Larger batch size for GPU
    device='cuda',
    use_amp=True  # Automatic Mixed Precision
):
    """
    Train FastText model with GPU acceleration
    
    Args:
        device: 'cuda', 'cpu', or specific GPU like 'cuda:0'
        use_amp: Use automatic mixed precision (FP16) for faster training
    """
    print(f"\n{'='*60}")
    print(f"Training Configuration:")
    print(f"  Device: {device}")
    print(f"  Mixed Precision: {use_amp}")
    print(f"  Batch Size: {batch_size}")
    print(f"  Embedding Dim: {embedding_dim}")
    print(f"  Window Size: {window_size}")
    print(f"  Negative Samples: {num_negatives}")
    print(f"  Learning Rate: {lr}")
    print(f"  Epochs: {epochs}")
    print(f"{'='*60}\n")
    
    # Word frequencies
    print("Computing word frequencies...")
    word_counts = Counter(w for sent in sentences for w in sent)
    total_words = sum(word_counts.values())
    
    word_freqs = {
        word_to_id[w]: word_counts[w]
        for w in word_counts if w in word_to_id
    }
    
    # Initialize model and move to device
    print("Initializing model...")
    model = FastTextSkipGram(
        num_ngrams=ngram_vocab_size,
        num_words=len(word_to_id),
        dim=embedding_dim
    ).to(device)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Use AdamW with weight decay for better convergence
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=lr * 0.1
    )
    
    # Initialize negative sampler on device
    neg_sampler = NegativeSampler(word_freqs, device=device)
    
    # Mixed precision scaler
    scaler = GradScaler() if use_amp else None
    
    loss_history = []
    
    for epoch in range(epochs):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch + 1}/{epochs}")
        print(f"{'='*60}")
        
        # Generate training pairs
        print("Generating training pairs...")
        pairs = generate_training_pairs(
            sentences, word_to_id, word_to_ngrams, 
            word_counts, total_words, window_size
        )
        
        print(f"Total training pairs: {len(pairs):,}")
        
        # Shuffle pairs
        random.shuffle(pairs)
        
        total_loss = 0.0
        num_batches = (len(pairs) + batch_size - 1) // batch_size
        
        # Batch training with progress bar
        model.train()
        pbar = tqdm(range(num_batches), desc=f"Epoch {epoch+1}/{epochs}")
        
        for batch_idx in pbar:
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, len(pairs))
            batch_pairs = pairs[start_idx:end_idx]
            
            # Prepare batch
            center_ngrams_batch = [p[0] for p in batch_pairs]
            context_ids = torch.tensor(
                [p[1] for p in batch_pairs], 
                dtype=torch.long, 
                device=device
            )
            
            # Sample negatives on device
            neg_ids = neg_sampler.sample(
                len(batch_pairs), 
                num_negatives, 
                context_ids
            )
            
            # Forward pass with mixed precision
            if use_amp:
                with autocast():
                    pos_scores, neg_scores = model(
                        center_ngrams_batch, 
                        context_ids, 
                        neg_ids
                    )
                    
                    # Loss computation
                    pos_loss = -F.logsigmoid(pos_scores).mean()
                    neg_loss = -F.logsigmoid(-neg_scores).mean()
                    loss = pos_loss + neg_loss
                
                # Backward pass with scaling
                optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                pos_scores, neg_scores = model(
                    center_ngrams_batch, 
                    context_ids, 
                    neg_ids
                )
                
                # Loss computation
                pos_loss = -F.logsigmoid(pos_scores).mean()
                neg_loss = -F.logsigmoid(-neg_scores).mean()
                loss = pos_loss + neg_loss
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            
            total_loss += loss.item()
            
            # Update progress bar
            if batch_idx % 10 == 0:
                pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        # Step scheduler
        scheduler.step()
        
        avg_loss = total_loss / num_batches
        loss_history.append(avg_loss)
        
        print(f"\nEpoch {epoch + 1} Summary:")
        print(f"  Average Loss: {avg_loss:.4f}")
        print(f"  Learning Rate: {scheduler.get_last_lr()[0]:.6f}")
    
    print(f"\n{'='*60}")
    print("Training Completed!")
    print(f"{'='*60}\n")
    
    return model, loss_history


# -----------------------------
# Save model
# -----------------------------
def save_model(model, word_to_id, ngram_to_id, loss_history=None, 
               path="fasttext_model.pt"):
    """Save model to CPU for compatibility"""
    checkpoint = {
        "ngram_embeddings": model.ngram_emb.weight.data.cpu(),
        "word_embeddings": model.word_emb.weight.data.cpu(),
        "word_to_id": word_to_id,
        "ngram_to_id": ngram_to_id,
    }
    
    if loss_history is not None:
        checkpoint["loss_history"] = loss_history
    
    torch.save(checkpoint, path)
    print(f"Model saved to {path}")


# -----------------------------
# Load model for inference
# -----------------------------
def load_model(path="fasttext_model.pt", device='cpu'):
    """Load model and move to specified device"""
    checkpoint = torch.load(path, map_location=device)
    
    ngram_embeddings = checkpoint["ngram_embeddings"].to(device)
    word_embeddings = checkpoint["word_embeddings"].to(device)
    word_to_id = checkpoint["word_to_id"]
    ngram_to_id = checkpoint["ngram_to_id"]
    
    # Reconstruct model
    model = FastTextSkipGram(
        num_ngrams=len(ngram_to_id),
        num_words=len(word_to_id),
        dim=ngram_embeddings.shape[1]
    ).to(device)
    
    model.ngram_emb.weight.data = ngram_embeddings
    model.word_emb.weight.data = word_embeddings
    model.eval()
    
    return model, word_to_id, ngram_to_id, checkpoint.get("loss_history", [])