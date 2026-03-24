# Train Word2Vec models — FROM SCRATCH using NumPy + compare with Gensim
import os
import sys
import time
import pickle
from collections import Counter
from itertools import product

import numpy as np
import pandas as pd

# CONFIGURATION
# Directory paths
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
MODELS_DIR = os.path.join(BASE_DIR, 'models')
OUTPUT_DIR = os.path.join(BASE_DIR, 'outputs')

# Path to cleaned corpus
CORPUS_PATH = os.path.join(DATA_DIR, 'cleaned_corpus.txt')

# Hyperparameter search space
# We use 3 dimensions × 3 windows = 9 configurations per model type
# × 2 model types (CBOW, Skip-gram) × 2 implementations (Scratch, Gensim) = 36 total
EMBEDDING_DIMS = [50, 100]
WINDOW_SIZES = [5]
NEGATIVE_SAMPLES = [10]

# Training settings
LEARNING_RATE = 0.025        # Initial learning rate
MIN_LEARNING_RATE = 0.0001   # Minimum learning rate (linear decay)
EPOCHS = 10                  # Number of passes through the corpus
MIN_COUNT = 2                # Ignore words with frequency < this
SEED = 42                    # Random seed for reproducibility


# VOCABULARY CLASS

class Vocabulary:
    # Vocabulary class to handle word-to-index mapping and word frequencies.
    def __init__(self, corpus, min_count=2):
        # Build vocabulary from a list of sentences.
        # Step 1: Count word frequencies across all sentences
        self.word_counts = Counter()
        for sentence in corpus:
            self.word_counts.update(sentence)
        
        # Step 2: Filter out infrequent words (below min_count)
        # Infrequent words don't have enough context for good embeddings
        self.word_counts = Counter({
            word: count for word, count in self.word_counts.items()
            if count >= min_count
        })
        
        # Step 3: Build word <-> index mappings
        # Sorting ensures deterministic assignment of indices
        sorted_words = sorted(self.word_counts.keys())
        self.word2idx = {word: idx for idx, word in enumerate(sorted_words)}
        self.idx2word = {idx: word for word, idx in self.word2idx.items()}
        
        # Step 4: Record vocabulary size
        self.vocab_size = len(self.word2idx)
        
        # Step 5: Build the negative sampling distribution
        # Words are sampled proportional to freq^(3/4) as suggested by
        # Mikolov et al. (2013). The 3/4 power flattens the distribution,
        # giving rare words a slightly higher chance of being sampled.
        self._build_sampling_table()
        
        print(f"  Vocabulary built: {self.vocab_size:,} words")
    
    def _build_sampling_table(self, table_size=100_000_000):
        # Build a unigram table for efficient negative sampling.
        # Reduce table size for smaller corpora to save memory
        table_size = min(table_size, self.vocab_size * 1000)
        
        # Compute freq^(3/4) for each word
        freqs = np.array([
            self.word_counts[self.idx2word[i]] for i in range(self.vocab_size)
        ], dtype=np.float64)
        
        # Apply the 3/4 power as per the original Word2Vec paper
        powered_freqs = np.power(freqs, 0.75)
        
        # Normalize to probabilities
        probs = powered_freqs / powered_freqs.sum()
        
        # Store probabilities for use in sampling
        self.neg_sampling_probs = probs
    
    def get_negative_samples(self, target_idx, num_samples):
        # Sample negative word indices for training.
        neg_samples = []
        while len(neg_samples) < num_samples:
            # Sample from the distribution
            sampled = np.random.choice(
                self.vocab_size, 
                size=num_samples * 2,  # Oversample to account for rejections
                p=self.neg_sampling_probs
            )
            # Filter out the target word
            valid = sampled[sampled != target_idx]
            neg_samples.extend(valid[:num_samples - len(neg_samples)])
        
        return np.array(neg_samples[:num_samples])
    
    def encode_corpus(self, corpus):
        # Convert text corpus to integer-encoded corpus.
        encoded = []
        for sentence in corpus:
            encoded_sent = [
                self.word2idx[w] for w in sentence if w in self.word2idx
            ]
            if len(encoded_sent) >= 3:  # Need enough words for context
                encoded.append(encoded_sent)
        return encoded


# WORD2VEC FROM SCRATCH — CORE IMPLEMENTATION
def sigmoid(x):
    # Compute the sigmoid activation function.
    x = np.clip(x, -10, 10)  # Prevent overflow
    return 1.0 / (1.0 + np.exp(-x))


class Word2VecScratch:
    # Word2Vec implementation from scratch using NumPy.
    # Supports both CBOW and Skip-gram with Negative Sampling.
    def __init__(self, vocab, embedding_dim=100, window_size=5,
                 num_neg_samples=5, sg=0, learning_rate=0.025,
                 min_lr=0.0001, epochs=20, seed=42):
        # Initialize the Word2Vec model.
        self.vocab = vocab
        self.embedding_dim = embedding_dim
        self.window_size = window_size
        self.num_neg_samples = num_neg_samples
        self.sg = sg  # 0 = CBOW, 1 = Skip-gram
        self.initial_lr = learning_rate
        self.min_lr = min_lr
        self.epochs = epochs
        self.loss_history = []
        self.training_time = 0.0
        
        np.random.seed(seed)
        
        # Initialize weight matrices with small random values
        # W_in: Input embeddings — these become the final word vectors
        # W_out: Output embeddings — used during training only
        # Uniform initialization in [-0.5/dim, 0.5/dim] as in original Word2Vec
        scale = 0.5 / embedding_dim
        self.W_in = np.random.uniform(
            -scale, scale, (vocab.vocab_size, embedding_dim)
        ).astype(np.float32)
        self.W_out = np.zeros(
            (vocab.vocab_size, embedding_dim), dtype=np.float32
        )
    
    def _train_cbow_pair(self, context_indices, target_idx, lr):
        # Train a single CBOW example: predict target from context.
        # Forward pass: compute the hidden layer as average of context vectors
        # h = (1/|C|) * sum(W_in[c] for c in context)
        context_vectors = self.W_in[context_indices]  # shape: (|C|, dim)
        h = np.mean(context_vectors, axis=0)          # shape: (dim,)
        
        # Accumulate gradient for the hidden layer
        grad_h = np.zeros_like(h)
        loss = 0.0
        
        # Positive sample (target word, label = 1)
        # Compute P(target | context) = sigmoid(W_out[target] · h)
        score = np.dot(self.W_out[target_idx], h)
        prob = sigmoid(score)
        
        # Gradient: (prob - 1) because label = 1
        # dL/d(W_out[target]) = (sigmoid - 1) * h
        # dL/dh += (sigmoid - 1) * W_out[target]
        g = (prob - 1.0)
        grad_h += g * self.W_out[target_idx]
        self.W_out[target_idx] -= lr * g * h
        
        # Binary cross-entropy loss: -log(sigmoid(score))
        loss -= np.log(prob + 1e-10)
        
        # Negative samples (random words, label = 0)
        neg_indices = self.vocab.get_negative_samples(target_idx, self.num_neg_samples)
        for neg_idx in neg_indices:
            score = np.dot(self.W_out[neg_idx], h)
            prob = sigmoid(score)
            
            # Gradient: (prob - 0) = prob because label = 0
            g = prob
            grad_h += g * self.W_out[neg_idx]
            self.W_out[neg_idx] -= lr * g * h
            
            # Binary cross-entropy loss: -log(1 - sigmoid(score))
            loss -= np.log(1.0 - prob + 1e-10)
        
        # Update input embeddings for all context words
        # Each context word gets the same gradient (divided by context size)
        grad_per_word = grad_h / len(context_indices)
        for c_idx in context_indices:
            self.W_in[c_idx] -= lr * grad_per_word
        
        return loss
    
    def _train_skipgram_pair(self, center_idx, context_idx, lr):
        # Train a single Skip-gram example: predict context from center word.
        # Forward pass: look up center word embedding
        h = self.W_in[center_idx].copy()  # shape: (dim,)
        
        # Accumulate gradient for the input word
        grad_h = np.zeros_like(h)
        loss = 0.0
        
        # Positive sample (context word, label = 1)
        score = np.dot(self.W_out[context_idx], h)
        prob = sigmoid(score)
        
        g = (prob - 1.0)  # label = 1
        grad_h += g * self.W_out[context_idx]
        self.W_out[context_idx] -= lr * g * h
        
        loss -= np.log(prob + 1e-10)
        
        # Negative samples (random words, label = 0)
        neg_indices = self.vocab.get_negative_samples(context_idx, self.num_neg_samples)
        for neg_idx in neg_indices:
            score = np.dot(self.W_out[neg_idx], h)
            prob = sigmoid(score)
            
            g = prob  # label = 0
            grad_h += g * self.W_out[neg_idx]
            self.W_out[neg_idx] -= lr * g * h
            
            loss -= np.log(1.0 - prob + 1e-10)
        
        # Update input embedding for center word
        self.W_in[center_idx] -= lr * grad_h
        
        return loss
    
    def train(self, encoded_corpus):
        # Train the Word2Vec model on the encoded corpus.
        model_type = "CBOW" if self.sg == 0 else "Skip-gram"
        print(f"\n  Training {model_type} from scratch...")
        print(f"    Vocab: {self.vocab.vocab_size:,}, Dim: {self.embedding_dim}, "
              f"Window: {self.window_size}, Neg: {self.num_neg_samples}")
        
        # Calculate total number of training words for LR scheduling
        total_words = sum(len(s) for s in encoded_corpus)
        total_training_words = total_words * self.epochs
        words_processed = 0
        
        start_time = time.time()
        
        for epoch in range(self.epochs):
            epoch_loss = 0.0
            epoch_examples = 0
            
            # Shuffle sentences for each epoch (different training order)
            indices = np.random.permutation(len(encoded_corpus))
            
            for sent_idx in indices:
                sentence = encoded_corpus[sent_idx]
                sent_len = len(sentence)
                
                for pos in range(sent_len):
                    # Linear learning rate decay
                    # LR decreases linearly from initial_lr to min_lr
                    progress = words_processed / total_training_words
                    lr = max(
                        self.min_lr,
                        self.initial_lr * (1.0 - progress)
                    )
                    
                    # Dynamic window: randomly shrink window for variety
                    actual_window = np.random.randint(1, self.window_size + 1)
                    
                    # Get context word indices within [pos-window, pos+window]
                    start = max(0, pos - actual_window)
                    end = min(sent_len, pos + actual_window + 1)
                    
                    context_indices = [
                        sentence[i] for i in range(start, end) if i != pos
                    ]
                    
                    if not context_indices:
                        continue
                    
                    if self.sg == 0:
                        # CBOW: predict center word from all context words
                        loss = self._train_cbow_pair(
                            context_indices, sentence[pos], lr
                        )
                        epoch_loss += loss
                        epoch_examples += 1
                    else:
                        # Skip-gram: predict each context word from center
                        for ctx_idx in context_indices:
                            loss = self._train_skipgram_pair(
                                sentence[pos], ctx_idx, lr
                            )
                            epoch_loss += loss
                            epoch_examples += 1
                    
                    words_processed += 1
            
            # Print epoch progress
            elapsed = time.time() - start_time
            avg_loss = epoch_loss / max(epoch_examples, 1)
            self.loss_history.append(float(avg_loss))
            print(f"    Epoch {epoch+1:2d}/{self.epochs}: "
                  f"loss={avg_loss:.4f}, lr={lr:.6f}, "
                  f"time={elapsed:.1f}s")
        
        self.training_time = time.time() - start_time
        print(f"  Training complete in {self.training_time:.1f}s")
    
    def get_embedding(self, word):
        # Get the embedding vector for a word.
        if word in self.vocab.word2idx:
            idx = self.vocab.word2idx[word]
            return self.W_in[idx]
        return None
    
    def most_similar(self, word, topn=5):
        # Find the top-N most similar words using cosine similarity.
        if word not in self.vocab.word2idx:
            return []
        
        word_idx = self.vocab.word2idx[word]
        word_vec = self.W_in[word_idx]
        
        # Compute cosine similarity with ALL words in vocabulary
        # Efficient vectorized computation using matrix multiplication
        norms = np.linalg.norm(self.W_in, axis=1)  # L2 norm of each word
        word_norm = np.linalg.norm(word_vec)
        
        # Avoid division by zero
        norms = np.maximum(norms, 1e-10)
        word_norm = max(word_norm, 1e-10)
        
        # Cosine similarity = dot product / (norm_a * norm_b)
        similarities = np.dot(self.W_in, word_vec) / (norms * word_norm)
        
        # Get top-N indices (exclude the query word itself)
        # argsort returns ascending order, so we negate for descending
        top_indices = np.argsort(-similarities)
        
        results = []
        for idx in top_indices:
            if idx != word_idx:
                results.append((self.vocab.idx2word[idx], float(similarities[idx])))
            if len(results) >= topn:
                break
        
        return results
    
    def analogy(self, word_a, word_b, word_c, topn=5):
        # Solve word analogies: word_a : word_b :: word_c : ?
        # Check all words exist in vocabulary
        for w in [word_a, word_b, word_c]:
            if w not in self.vocab.word2idx:
                return []
        
        # Compute the analogy vector
        vec_a = self.W_in[self.vocab.word2idx[word_a]]
        vec_b = self.W_in[self.vocab.word2idx[word_b]]
        vec_c = self.W_in[self.vocab.word2idx[word_c]]
        
        result_vec = vec_b - vec_a + vec_c
        
        # Find most similar words to the result vector
        norms = np.linalg.norm(self.W_in, axis=1)
        result_norm = np.linalg.norm(result_vec)
        norms = np.maximum(norms, 1e-10)
        result_norm = max(result_norm, 1e-10)
        
        similarities = np.dot(self.W_in, result_vec) / (norms * result_norm)
        
        # Exclude the input words from results
        exclude = {
            self.vocab.word2idx[word_a],
            self.vocab.word2idx[word_b],
            self.vocab.word2idx[word_c]
        }
        
        top_indices = np.argsort(-similarities)
        results = []
        for idx in top_indices:
            if idx not in exclude:
                results.append((self.vocab.idx2word[idx], float(similarities[idx])))
            if len(results) >= topn:
                break
        
        return results
    
    def save(self, filepath):
        # Save the model to disk using pickle.  
        # Saves the embedding matrices, vocabulary, and hyperparameters.
        model_data = {
            'W_in': self.W_in,
            'W_out': self.W_out,
            'word2idx': self.vocab.word2idx,
            'idx2word': self.vocab.idx2word,
            'embedding_dim': self.embedding_dim,
            'window_size': self.window_size,
            'num_neg_samples': self.num_neg_samples,
            'sg': self.sg,
            'loss_history': self.loss_history,
            'training_time': self.training_time
        }
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
    
    @staticmethod
    def load(filepath):
        # Load a saved model from disk.
        with open(filepath, 'rb') as f:
            return pickle.load(f)


# GENSIM TRAINING (for comparison)
from gensim.models.callbacks import CallbackAny2Vec

class LossCallback(CallbackAny2Vec):
    """Callback to print loss after each epoch."""
    def __init__(self):
        self.epoch = 0
        self.loss_to_display = []
        self.last_loss = 0

    def on_epoch_end(self, model):
        # Gensim's get_latest_training_loss() returns cumulative loss
        cumulative_loss = model.get_latest_training_loss()
        loss_now = cumulative_loss - self.last_loss
        self.last_loss = cumulative_loss
        self.loss_to_display.append(loss_now)
        # print(f"    Epoch {self.epoch+1:2d} loss: {loss_now}")
        self.epoch += 1


def train_gensim_model(corpus, sg, vector_size, window, negative, epochs=20):
    """ Train a Word2Vec model using Gensim for comparison. """
    from gensim.models import Word2Vec
    
    loss_callback = LossCallback()
    
    model = Word2Vec(
        sentences=corpus,
        vector_size=vector_size,
        window=window,
        min_count=MIN_COUNT,
        sg=sg,
        negative=negative,
        workers=4,
        epochs=epochs,
        seed=SEED,
        compute_loss=True,  # Enable loss computation for comparison
        callbacks=[loss_callback]
    )
    return model, loss_callback


# EVALUATION HELPERS
def evaluate_scratch_model(model, test_words, topn=5):
    """ Evaluate a from-scratch Word2Vec model using average cosine similarity. """
    similarities = []
    for word in test_words:
        neighbors = model.most_similar(word, topn=topn)
        if neighbors:
            avg_sim = np.mean([sim for _, sim in neighbors])
            similarities.append(avg_sim)
    
    return np.mean(similarities) if similarities else 0.0


def evaluate_gensim_model(model, test_words, topn=5):
    """ Evaluate a Gensim Word2Vec model using average cosine similarity. """
    similarities = []
    for word in test_words:
        if word in model.wv:
            neighbors = model.wv.most_similar(word, topn=topn)
            avg_sim = np.mean([sim for _, sim in neighbors])
            similarities.append(avg_sim)
    
    return np.mean(similarities) if similarities else 0.0


# MAIN EXECUTION
def main():
    """ Main function for model training and comparison."""
    print("=" * 70)
    print("WORD2VEC TRAINING — FROM SCRATCH + GENSIM COMPARISON")
    print("=" * 70)
    
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Step 1: Load corpus
    print("\n[Step 1] Loading cleaned corpus...")
    if not os.path.exists(CORPUS_PATH):
        print(f"  ERROR: Corpus not found at {CORPUS_PATH}")
        print("  Please run 02_preprocess.py first.")
        sys.exit(1)
    
    corpus = []
    with open(CORPUS_PATH, 'r', encoding='utf-8') as f:
        for line in f:
            tokens = line.strip().split()
            if tokens:
                corpus.append(tokens)
    
    print(f"  Loaded: {len(corpus):,} sentences")
    
    # Step 2: Build vocabulary and encode corpus
    print("\n[Step 2] Building vocabulary...")
    vocab = Vocabulary(corpus, min_count=MIN_COUNT)
    encoded_corpus = vocab.encode_corpus(corpus)
    print(f"  Encoded corpus: {len(encoded_corpus):,} sentences")
    
    # Test words for evaluation
    test_words = ['research', 'student', 'phd', 'exam']
    
    # Step 3: Hyperparameter experiments
    print("\n[Step 3] Running hyperparameter experiments...")
    print(f"  Dimensions: {EMBEDDING_DIMS}")
    print(f"  Windows: {WINDOW_SIZES}")
    print(f"  Neg samples: {NEGATIVE_SAMPLES}")
    
    all_results = []
    best_scratch = {'CBOW': None, 'Skip-gram': None}
    best_scratch_score = {'CBOW': -1, 'Skip-gram': -1}
    best_gensim = {'CBOW': None, 'Skip-gram': None}
    best_gensim_score = {'CBOW': -1, 'Skip-gram': -1}
    
    experiment_num = 0
    total_experiments = len(EMBEDDING_DIMS) * len(WINDOW_SIZES) * len(NEGATIVE_SAMPLES) * 2
    
    for dim, win, neg in product(EMBEDDING_DIMS, WINDOW_SIZES, NEGATIVE_SAMPLES):
        for sg_flag, model_name in [(0, 'CBOW'), (1, 'Skip-gram')]:
            experiment_num += 1
            print(f"\n{'='*60}")
            print(f"  [{experiment_num}/{total_experiments}] {model_name} — "
                  f"dim={dim}, window={win}, neg={neg}")
            print(f"{'='*60}")
            
            # Train FROM SCRATCH
            start_time = time.time()
            scratch_model = Word2VecScratch(
                vocab=vocab,
                embedding_dim=dim,
                window_size=win,
                num_neg_samples=neg,
                sg=sg_flag,
                learning_rate=LEARNING_RATE,
                min_lr=MIN_LEARNING_RATE,
                epochs=EPOCHS,
                seed=SEED,
            )
            scratch_model.train(encoded_corpus)
            scratch_time = time.time() - start_time
            
            # Evaluate scratch model
            scratch_sim = evaluate_scratch_model(scratch_model, test_words)
            
            # Save scratch model
            scratch_filename = f"scratch_{model_name}_d{dim}_w{win}_n{neg}.pkl"
            scratch_model.save(os.path.join(MODELS_DIR, scratch_filename))
            
            # Track best scratch model
            if scratch_sim > best_scratch_score[model_name]:
                best_scratch_score[model_name] = scratch_sim
                best_scratch[model_name] = scratch_model
            
            # Train with GENSIM (comparison)
            start_time = time.time()
            gensim_model, gensim_callback = train_gensim_model(
                corpus, sg=sg_flag, vector_size=dim,
                window=win, negative=neg, epochs=EPOCHS
            )
            gensim_time = time.time() - start_time
            
            # Store loss in gensim model object for later use
            gensim_model.loss_history = gensim_callback.loss_to_display
            
            # Evaluate gensim model
            gensim_sim = evaluate_gensim_model(gensim_model, test_words)
            
            # Save gensim model
            gensim_filename = f"gensim_{model_name}_d{dim}_w{win}_n{neg}.model"
            gensim_model.save(os.path.join(MODELS_DIR, gensim_filename))
            
            # Track best gensim model
            if gensim_sim > best_gensim_score[model_name]:
                best_gensim_score[model_name] = gensim_sim
                best_gensim[model_name] = gensim_model
            
            # Record results
            print(f"\n  Results:")
            print(f"    Scratch — Avg Sim: {scratch_sim:.4f}, Time: {scratch_time:.1f}s")
            print(f"    Gensim  — Avg Sim: {gensim_sim:.4f}, Time: {gensim_time:.1f}s")
            
            all_results.append({
                'Model': model_name,
                'Implementation': 'Scratch',
                'Dim': dim, 'Window': win, 'Neg': neg,
                'Avg Top-5 Sim': round(scratch_sim, 4),
                'Time (s)': round(scratch_time, 2),
            })
            all_results.append({
                'Model': model_name,
                'Implementation': 'Gensim',
                'Dim': dim, 'Window': win, 'Neg': neg,
                'Avg Top-5 Sim': round(gensim_sim, 4),
                'Time (s)': round(gensim_time, 2),
            })
    
    # Step 4: Save results to CSV and print table
    results_df = pd.DataFrame(all_results)
    csv_path = os.path.join(OUTPUT_DIR, 'hyperparameter_results.csv')
    results_df.to_csv(csv_path, index=False)
    
    print("\n" + "=" * 70)
    print("RESULTS TABLE")
    print("=" * 70)
    print(results_df.to_string(index=False))
    
    # Step 5: Save best models with clean names and print summary
    print("\n" + "-" * 70)
    print("BEST MODELS")
    print("-" * 70)
    
    for model_name in ['CBOW', 'Skip-gram']:
        clean = model_name.replace('-', '').lower()
        
        # Save best scratch model
        if best_scratch[model_name]:
            path = os.path.join(MODELS_DIR, f'{clean}_best_scratch.pkl')
            best_scratch[model_name].save(path)
            print(f"  Best Scratch {model_name}: score={best_scratch_score[model_name]:.4f}")
            print(f"    Saved: {path}")
        
        # Save best gensim model
        if best_gensim[model_name]:
            path = os.path.join(MODELS_DIR, f'{clean}_best.model')
            best_gensim[model_name].save(path)
            print(f"  Best Gensim {model_name}: score={best_gensim_score[model_name]:.4f}")
            print(f"    Saved: {path}")
    
    print("\n" + "=" * 70)
    print("SAVING LOSS CURVES")
    print("=" * 70)
    
    loss_data = {}
    for model_name in ['CBOW', 'Skip-gram']:
        if best_scratch[model_name]:
            loss_data[f'scratch_{model_name}'] = best_scratch[model_name].loss_history
        if best_gensim[model_name]:
            loss_data[f'gensim_{model_name}'] = best_gensim[model_name].loss_history
            
    with open(os.path.join(OUTPUT_DIR, 'loss_curves.pkl'), 'wb') as f:
        pickle.dump(loss_data, f)
    print(f"  Loss curves saved to {os.path.join(OUTPUT_DIR, 'loss_curves.pkl')}")

    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)


if __name__ == '__main__':
    main()
