# Perform semantic analysis — compare scratch vs gensim embeddings
import os
import sys
import pickle

from gensim.models import Word2Vec
import pandas as pd
import numpy as np

# CONFIGURATION
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
MODELS_DIR = os.path.join(BASE_DIR, 'models')
OUTPUT_DIR = os.path.join(BASE_DIR, 'outputs')

# Query words from the assignment
QUERY_WORDS = ['research', 'student', 'phd', 'examination']
TOP_N = 5

# Analogy experiments
# Format: (word_a, word_b, word_c, description)
# Semantics: a : b :: c : ?
ANALOGY_EXPERIMENTS = [
    ('ug', 'btech', 'pg', 'UG : BTech :: PG : ?'),
    ('student', 'study', 'faculty', 'Student : Study :: Faculty : ?'),
    ('department', 'hod', 'institute', 'Department : HOD :: Institute : ?'),
    ('btech', 'undergraduate', 'phd', 'BTech : Undergraduate :: PhD : ?'),
]


# SCRATCH MODEL WRAPPER
class ScratchModelWrapper:
    def __init__(self, filepath):
        # Load a from-scratch model from a pickle file.
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        self.W_in = data['W_in']           # Input embedding matrix
        self.word2idx = data['word2idx']    # Word -> index mapping
        self.idx2word = data['idx2word']    # Index -> word mapping
        self.embedding_dim = data['embedding_dim']
        self.vocab_size = len(self.word2idx)
        
        print(f"  Loaded scratch model: {self.vocab_size:,} words, "
              f"dim={self.embedding_dim}")
    
    def most_similar(self, word, topn=5):
        # Find the top-N most similar words using cosine similarity.  
        if word not in self.word2idx:
            return []
        
        word_idx = self.word2idx[word]
        word_vec = self.W_in[word_idx]
        
        # Compute cosine similarity with all words
        norms = np.linalg.norm(self.W_in, axis=1)
        word_norm = np.linalg.norm(word_vec)
        norms = np.maximum(norms, 1e-10)
        word_norm = max(word_norm, 1e-10)
        
        similarities = np.dot(self.W_in, word_vec) / (norms * word_norm)
        top_indices = np.argsort(-similarities)
        
        results = []
        for idx in top_indices:
            if idx != word_idx:
                results.append((self.idx2word[idx], float(similarities[idx])))
            if len(results) >= topn:
                break
        
        return results
    
    def analogy(self, word_a, word_b, word_c, topn=5):
        # Solve word analogies using vector arithmetic.
        for w in [word_a, word_b, word_c]:
            if w not in self.word2idx:
                return []
        
        vec_a = self.W_in[self.word2idx[word_a]]
        vec_b = self.W_in[self.word2idx[word_b]]
        vec_c = self.W_in[self.word2idx[word_c]]
        
        result_vec = vec_b - vec_a + vec_c
        
        norms = np.linalg.norm(self.W_in, axis=1)
        result_norm = np.linalg.norm(result_vec)
        norms = np.maximum(norms, 1e-10)
        result_norm = max(result_norm, 1e-10)
        
        similarities = np.dot(self.W_in, result_vec) / (norms * result_norm)
        
        exclude = {self.word2idx[word_a], self.word2idx[word_b], self.word2idx[word_c]}
        top_indices = np.argsort(-similarities)
        
        results = []
        for idx in top_indices:
            if idx not in exclude:
                results.append((self.idx2word[idx], float(similarities[idx])))
            if len(results) >= topn:
                break
        
        return results
    
    def __contains__(self, word):
        # Check if a word is in the vocabulary.
        return word in self.word2idx


# GENSIM MODEL WRAPPER
class GensimModelWrapper:
    # Wrapper for Gensim Word2Vec model to provide a consistent interface.    
    def __init__(self, filepath):
        """Load a gensim model from file."""
        self.model = Word2Vec.load(filepath)
        print(f"  Loaded gensim model: {len(self.model.wv):,} words, "
              f"dim={self.model.wv.vector_size}")
    
    def most_similar(self, word, topn=5):
        """Find top-N similar words."""
        if word not in self.model.wv:
            return []
        return self.model.wv.most_similar(word, topn=topn)
    
    def analogy(self, word_a, word_b, word_c, topn=5):
        """Solve analogies using gensim's most_similar with pos/neg."""
        for w in [word_a, word_b, word_c]:
            if w not in self.model.wv:
                return []
        try:
            return self.model.wv.most_similar(
                positive=[word_b, word_c], negative=[word_a], topn=topn
            )
        except KeyError:
            return []
    
    def __contains__(self, word):
        """Check if a word is in the vocabulary."""
        return word in self.model.wv


# ANALYSIS FUNCTIONS
def find_nearest_neighbors(model, words, model_label, topn=5):
    # Find top-N nearest neighbors for each query word.
    results = []
    
    print(f"\n{'=' * 60}")
    print(f"NEAREST NEIGHBORS — {model_label}")
    print(f"{'=' * 60}")
    
    for word in words:
        neighbors = model.most_similar(word, topn=topn)
        
        if neighbors:
            print(f"\n  '{word}':")
            for rank, (neighbor, sim) in enumerate(neighbors, 1):
                print(f"    {rank}. {neighbor:20s} {sim:.4f}")
                results.append({
                    'Model': model_label,
                    'Query': word,
                    'Rank': rank,
                    'Neighbor': neighbor,
                    'Cosine Similarity': round(sim, 4),
                })
        else:
            print(f"\n  '{word}' — NOT IN VOCABULARY")
    
    return results


def run_analogies(model, experiments, model_label, topn=5):
    # Run analogy experiments on a model.
    results = []
    
    print(f"\n{'=' * 60}")
    print(f"ANALOGY EXPERIMENTS — {model_label}")
    print(f"{'=' * 60}")
    
    for word_a, word_b, word_c, description in experiments:
        print(f"\n  {description}")
        
        answers = model.analogy(word_a, word_b, word_c, topn=topn)
        
        if answers:
            for rank, (answer, sim) in enumerate(answers, 1):
                print(f"    {rank}. {answer:20s} {sim:.4f}")
                results.append({
                    'Model': model_label,
                    'Analogy': description,
                    'Rank': rank,
                    'Answer': answer,
                    'Similarity': round(sim, 4),
                })
        else:
            missing = [w for w in [word_a, word_b, word_c] if w not in model]
            print(f"    ⚠ Missing words: {missing}")
            results.append({
                'Model': model_label,
                'Analogy': description,
                'Rank': '-',
                'Answer': f'MISSING: {missing}',
                'Similarity': '-',
            })
    
    return results


# MAIN SECTION

def main():
    """
    Run full semantic analysis on all 4 models:
      - CBOW (Scratch)
      - CBOW (Gensim)
      - Skip-gram (Scratch)
      - Skip-gram (Gensim)
    """
    print("=" * 70)
    print("SEMANTIC ANALYSIS — SCRATCH vs GENSIM COMPARISON")
    print("=" * 70)
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Load all 4 models
    models = {}
    model_files = {
        'CBOW (Scratch)': ('cbow_best_scratch.pkl', ScratchModelWrapper),
        'CBOW (Gensim)': ('cbow_best.model', GensimModelWrapper),
        'Skip-gram (Scratch)': ('skipgram_best_scratch.pkl', ScratchModelWrapper),
        'Skip-gram (Gensim)': ('skipgram_best.model', GensimModelWrapper),
    }
    
    print("\n[Step 1] Loading models...")
    for label, (filename, wrapper_class) in model_files.items():
        filepath = os.path.join(MODELS_DIR, filename)
        if os.path.exists(filepath):
            models[label] = wrapper_class(filepath)
        else:
            print(f"  ⚠ {label} model not found: {filepath}")
    
    if not models:
        print("ERROR: No models found. Run 03_train_models.py first.")
        sys.exit(1)
    
    # Run nearest neighbor analysis on all models
    print("\n[Step 2] Finding nearest neighbors...")
    nn_results = []
    for label, model in models.items():
        nn_results.extend(find_nearest_neighbors(model, QUERY_WORDS, label, TOP_N))
    
    nn_df = pd.DataFrame(nn_results)
    nn_path = os.path.join(OUTPUT_DIR, 'nearest_neighbors.csv')
    nn_df.to_csv(nn_path, index=False)
    print(f"\n  Saved: {nn_path}")
    
    # Run analogy experiments on all models
    print("\n[Step 3] Running analogy experiments...")
    analogy_results = []
    for label, model in models.items():
        analogy_results.extend(run_analogies(model, ANALOGY_EXPERIMENTS, label, TOP_N))
    
    analogy_df = pd.DataFrame(analogy_results)
    analogy_path = os.path.join(OUTPUT_DIR, 'analogy_results.csv')
    analogy_df.to_csv(analogy_path, index=False)
    print(f"\n  Saved: {analogy_path}")
    
    # Discussion framework
    print("\n" + "=" * 60)
    print("DISCUSSION POINTS FOR REPORT")
    print("=" * 60)
    print("""
    1. SCRATCH vs GENSIM:
       - Do the from-scratch models produce similar neighbors as Gensim?
       - Which implementation yields more semantically coherent results?
       - The Gensim implementation uses optimized C code, so training is
         faster, but the semantic quality should be comparable if the
         from-scratch implementation is correct.
    
    2. CBOW vs SKIP-GRAM:
       - CBOW: Better for frequent words, smoother representations
       - Skip-gram: Better for rare words, more distinct relationships
       - Compare which model produces more meaningful neighbors
    
    3. ANALOGIES:
       - If "UG:BTech :: PG:?" returns MTech/MSc, the model has captured
         the academic level hierarchy from the IIT Jodhpur corpus.
       - Domain-specific analogies may work better than general ones
         since the corpus is specialized.
    
    4. LIMITATIONS:
       - Small corpus size limits embedding quality
       - Domain-specific vocabulary may not generalize well
       - From-scratch implementation may have slightly different
         optimization behavior compared to Gensim's C implementation
    """)
    
    print("\n" + "=" * 70)
    print("SEMANTIC ANALYSIS COMPLETE")
    print("=" * 70)


if __name__ == '__main__':
    main()
