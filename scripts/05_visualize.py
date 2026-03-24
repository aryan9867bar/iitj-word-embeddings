# Visualize Word2Vec embeddings using PCA and t-SNE — all 4 models
import os
import sys
import pickle

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for saving plots
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from gensim.models import Word2Vec

# CONFIGURATION
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
MODELS_DIR = os.path.join(BASE_DIR, 'models')
OUTPUT_DIR = os.path.join(BASE_DIR, 'outputs')

# Semantic word groups for clustering visualization
WORD_GROUPS = {
    'Academic Programs': [
        'btech', 'mtech', 'phd', 'msc', 'mba',
        'undergraduate', 'postgraduate', 'doctoral', 'degree',
        'diploma', 'programme', 'program',
    ],
    'Departments': [
        'cse', 'electrical', 'mechanical', 'civil',
        'physics', 'chemistry', 'mathematics',
        'computer', 'engineering', 'science', 'department',
    ],
    'Academic Terms': [
        'semester', 'credits', 'exam', 'examination',
        'grade', 'cgpa', 'thesis', 'dissertation',
        'syllabus', 'course', 'curriculum', 'lecture',
    ],
    'People': [
        'student', 'faculty', 'professor', 'dean',
        'director', 'hod', 'researcher', 'scholar',
        'candidate', 'advisor', 'mentor', 'teacher',
    ],
    'Research': [
        'research', 'publication', 'journal', 'conference',
        'paper', 'project', 'innovation', 'laboratory',
        'analysis', 'study', 'experiment',
    ],
}

# Colors for each word group — distinct, colorblind-friendly palette
GROUP_COLORS = {
    'Academic Programs': '#e63946',  # Red
    'Departments': '#457b9d',        # Steel Blue
    'Academic Terms': '#2a9d8f',     # Teal
    'People': '#e9c46a',            # Gold
    'Research': '#f4a261',           # Orange
}


# MODEL LOADING
def load_scratch_model(filepath):

    # Loading a from-scratch Word2Vec model.
    with open(filepath, 'rb') as f:
        data = pickle.load(f)
    return data['W_in'], data['word2idx'], data['idx2word']


def get_vectors_scratch(filepath, word_groups):
    # Extract word vectors from a scratch model for visualization.
    W_in, word2idx, idx2word = load_scratch_model(filepath)
    
    words, vectors, labels, colors = [], [], [], []
    for group_name, group_words in word_groups.items():
        for word in group_words:
            if word in word2idx:
                words.append(word)
                vectors.append(W_in[word2idx[word]])
                labels.append(group_name)
                colors.append(GROUP_COLORS[group_name])
    
    return words, np.array(vectors) if vectors else np.array([]), labels, colors


def get_vectors_gensim(filepath, word_groups):
    # Extract word vectors from a Gensim model for visualization.
    model = Word2Vec.load(filepath)
    
    words, vectors, labels, colors = [], [], [], []
    for group_name, group_words in word_groups.items():
        for word in group_words:
            if word in model.wv:
                words.append(word)
                vectors.append(model.wv[word])
                labels.append(group_name)
                colors.append(GROUP_COLORS[group_name])
    
    return words, np.array(vectors) if vectors else np.array([]), labels, colors


# PLOTTING
def plot_embeddings(words, coords_2d, labels, title, output_path):
    # Create a 2D scatter plot of word embeddings.
    fig, ax = plt.subplots(1, 1, figsize=(16, 12))
    
    # Plot each category separately for proper legend
    for group_name in WORD_GROUPS.keys():
        mask = [l == group_name for l in labels]
        indices = [i for i, m in enumerate(mask) if m]
        
        if not indices:
            continue
        
        gx = coords_2d[indices, 0]
        gy = coords_2d[indices, 1]
        
        ax.scatter(gx, gy, c=GROUP_COLORS[group_name], label=group_name,
                  s=150, alpha=0.75, edgecolors='white', linewidths=0.5, zorder=5)
        
        gwords = [words[i] for i in indices]
        for x, y, word in zip(gx, gy, gwords):
            ax.annotate(word, (x, y), fontsize=9, fontweight='bold',
                       ha='center', va='bottom', xytext=(0, 8),
                       textcoords='offset points', alpha=0.85)
    
    ax.set_title(title, fontsize=18, fontweight='bold', pad=20)
    ax.set_xlabel('Component 1', fontsize=13)
    ax.set_ylabel('Component 2', fontsize=13)
    ax.legend(fontsize=11, loc='best', framealpha=0.9, edgecolor='gray',
             title='Semantic Categories', title_fontsize=12)
    ax.grid(True, alpha=0.2, linestyle='--')
    ax.set_axisbelow(True)
    
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"    Saved: {output_path}")


def reduce_and_plot(words, vectors, labels, title, output_path, method='pca'):
    # Reduce dimensions and plot word embeddings.
    if len(words) < 5:
        print(f"    ⚠ Only {len(words)} words found. Skipping.")
        return
    
    print(f"    {method.upper()}: {len(words)} words")
    
    if method == 'pca':
        reducer = PCA(n_components=2, random_state=42)
        coords = reducer.fit_transform(vectors)
        explained = reducer.explained_variance_ratio_
        print(f"    Explained variance: {explained[0]:.2%} + {explained[1]:.2%} = {sum(explained):.2%}")
    else:
        perplexity = min(30, len(words) - 1)
        reducer = TSNE(n_components=2, random_state=42, perplexity=perplexity,
                      max_iter=1000, learning_rate='auto')
        coords = reducer.fit_transform(vectors)
    
    plot_embeddings(words, coords, labels, title, output_path)


# NEW VISUALIZATIONS: LOSS & TIME
def plot_loss_curves(loss_data_path):
    # Plot training loss curves for all models.
    if not os.path.exists(loss_data_path):
        print(f"  ⚠ Loss data not found at {loss_data_path}. Skipping.")
        return

    with open(loss_data_path, 'rb') as f:
        loss_data = pickle.load(f)

    if not loss_data:
        print("  ⚠ No loss data found in file. Skipping.")
        return

    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    for label, losses in loss_data.items():
        if not losses: continue
        epochs = range(1, len(losses) + 1)
        ax.plot(epochs, losses, marker='o', linewidth=2, label=label.replace('_', ' ').title())

    ax.set_title('Training Loss Convergence (Log Scale)', fontsize=16, fontweight='bold')
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Loss (Log Scale)', fontsize=12)
    ax.set_yscale('log')  # Use log scale to handle different loss magnitudes
    ax.legend(fontsize=10)
    ax.grid(True, which="both", alpha=0.3)
    
    output_path = os.path.join(OUTPUT_DIR, 'training_loss_curves.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"    Saved: {output_path}")


def plot_training_time_comparison(csv_path):
    # Create a professional bar chart comparing training times of all 4 models.
    import pandas as pd
    if not os.path.exists(csv_path):
        print(f"  ⚠ Results CSV not found at {csv_path}. Skipping.")
        return

    df = pd.read_csv(csv_path)
    
    # Calculate group averages
    # We want 4 specific categories: CBOW (Gensim), CBOW (Scratch), etc.
    summary = df.groupby(['Model', 'Implementation'])['Time (s)'].mean().reset_index()
    summary['Label'] = summary['Model'] + "\n(" + summary['Implementation'] + ")"
    
    # Sort for consistent display: CBOW first, then Skip-gram
    summary['sort_idx'] = summary['Model'].apply(lambda x: 0 if x == 'CBOW' else 1)
    summary = summary.sort_values(['sort_idx', 'Implementation'], ascending=[True, False])
    
    labels = summary['Label'].tolist()
    times = summary['Time (s)'].tolist()
    
    # Professional color palette
    # CBOW: Blues, Skip-gram: Oranges/Reds
    colors = ['#1f77b4', '#3498db', '#e74c3c', '#c0392b'] 
    # Let's use more distinct ones: 
    # Gensim: Cool tones, Scratch: Warm tones
    colors = ['#457b9d', '#1d3557', '#e63946', '#a8dadc'] # Mixed professional palette
    
    # Re-evaluating colors for maximum professionalism:
    # CBOW Gensim, CBOW Scratch, SG Gensim, SG Scratch
    colors = ['#457b9d', '#e63946', '#1d3557', '#c0392b']
    
    fig, ax = plt.subplots(1, 1, figsize=(11, 7))
    bars = ax.bar(labels, times, color=colors, width=0.6, alpha=0.9, edgecolor='black', linewidth=0.5)
    
    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.1f}s', 
                    (bar.get_x() + bar.get_width() / 2., height), 
                    ha='center', va='center', 
                    xytext=(0, 10), 
                    textcoords='offset points',
                    fontsize=11, fontweight='bold', color='#2c3e50')

    ax.set_title('Training Speed: Custom NumPy vs. Optimized Gensim', fontsize=18, fontweight='bold', pad=25)
    ax.set_ylabel('Total Training Time (seconds)', fontsize=13, labelpad=10)
    ax.set_xlabel('Model Configuration', fontsize=13, labelpad=10)
    
    # Clean up grid and spines
    ax.grid(axis='y', alpha=0.2, linestyle='--', zorder=0)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Add a subtitle/caption via text if needed, or just keep it simple
    plt.xticks(fontsize=11, fontweight='medium')
    
    output_path = os.path.join(OUTPUT_DIR, 'training_time_comparison.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"    Saved: {output_path}")


# MAIN SECTION
def main():
    # Generate all visualizations for scratch and gensim models.
    print("=" * 70)
    print("WORD EMBEDDING VISUALIZATION")
    print("=" * 70)
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Define all model configurations
    model_configs = [
        ('CBOW (Scratch)', 'cbow_best_scratch.pkl', get_vectors_scratch),
        ('CBOW (Gensim)', 'cbow_best.model', get_vectors_gensim),
        ('Skip-gram (Scratch)', 'skipgram_best_scratch.pkl', get_vectors_scratch),
        ('Skip-gram (Gensim)', 'skipgram_best.model', get_vectors_gensim),
    ]
    
    for model_label, filename, get_vectors_fn in model_configs:
        filepath = os.path.join(MODELS_DIR, filename)
        
        if not os.path.exists(filepath):
            print(f"\n  ⚠ {model_label} not found: {filepath}")
            continue
        
        print(f"\n  Processing: {model_label}")
        
        # Get word vectors
        words, vectors, labels, colors = get_vectors_fn(filepath, WORD_GROUPS)
        
        if len(words) < 5:
            print(f"    ⚠ Not enough words. Skipping.")
            continue
        
        # Create a clean filename prefix
        clean_name = model_label.lower().replace(' ', '_').replace('(', '').replace(')', '')
        
        # PCA visualization
        reduce_and_plot(words, vectors, labels,
                       f'{model_label} — PCA Projection',
                       os.path.join(OUTPUT_DIR, f'pca_{clean_name}.png'),
                       method='pca')
        
        # t-SNE visualization
        reduce_and_plot(words, vectors, labels,
                       f'{model_label} — t-SNE Projection',
                        os.path.join(OUTPUT_DIR, f'tsne_{clean_name}.png'),
                       method='tsne')
    
    # NEW: Plot loss and time comparisons
    print("\n  Generating performance comparisons...")
    
    # Plot loss curves
    plot_loss_curves(os.path.join(OUTPUT_DIR, 'loss_curves.pkl'))
    
    # Plot training time comparison
    plot_training_time_comparison(os.path.join(OUTPUT_DIR, 'hyperparameter_results.csv'))
    
    # Interpretation guidelines
    print("\n" + "=" * 60)
    print("INTERPRETATION GUIDELINES FOR REPORT")
    print("=" * 60)
    print("""
    1. CLUSTERING — Do words from the same category cluster together?
    2. PCA vs t-SNE — PCA shows global structure, t-SNE shows local clusters
    3. CBOW vs Skip-gram — Which produces more coherent clusters?
    4. SCRATCH vs GENSIM — Are the clustering patterns similar?
       If yes, your from-scratch implementation is working correctly.
       If different, discuss possible reasons (optimization differences,
       numerical precision, etc.)
    """)
    
    print("\n" + "=" * 70)
    print(f"VISUALIZATION COMPLETE — All plots saved to: {OUTPUT_DIR}")
    print("=" * 70)


if __name__ == '__main__':
    main()
