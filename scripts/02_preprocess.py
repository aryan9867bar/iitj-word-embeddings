# Preprocess the raw collected text into a clean Word2Vec-ready corpus
import os
import re
import sys
from collections import Counter

import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for saving plots
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import numpy as np

# CONFIGURATION
# Directory paths — computed relative to this script's location
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
RAW_DATA_DIR = os.path.join(BASE_DIR, 'data', 'raw')
OUTPUT_DIR = os.path.join(BASE_DIR, 'outputs')
DATA_DIR = os.path.join(BASE_DIR, 'data')

# Minimum number of tokens required for a sentence to be included
# Sentences shorter than this are likely fragments or noise
MIN_SENTENCE_LENGTH = 3

# Minimum word length — single characters are usually noise
MIN_WORD_LENGTH = 2


# PREPROCESSING FUNCTIONS
def load_raw_texts(raw_dir):
    """ Load all .txt files from the raw data directory. """
    texts = {}
    
    # Iterate through all files in the raw data directory
    for filename in sorted(os.listdir(raw_dir)):
        if filename.endswith('.txt'):
            filepath = os.path.join(raw_dir, filename)
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            texts[filename] = content
            print(f"  Loaded: {filename} ({len(content):,} characters)")
    
    return texts


def remove_non_english(text):
    """ Remove lines that contain predominantly non-English text. """
    cleaned_lines = []
    
    for line in text.splitlines():
        if not line.strip():
            continue
        
        # Calculate the ratio of ASCII characters to total characters
        # ASCII characters include standard English letters, digits, and punctuation
        ascii_count = sum(1 for char in line if ord(char) < 128)
        total_count = len(line)
        
        # Keep the line only if it is predominantly ASCII (English)
        if total_count > 0 and (ascii_count / total_count) > 0.70:
            # Additional check: remove any remaining Devanagari characters
            # Unicode range for Devanagari: U+0900 to U+097F
            line = re.sub(r'[\u0900-\u097F]+', '', line)
            if line.strip():
                cleaned_lines.append(line.strip())
    
    return '\n'.join(cleaned_lines)


def remove_boilerplate(text):
    """ Remove boilerplate content that is not useful for language modeling. """
    # Remove URLs — matches http(s):// and www. patterns
    text = re.sub(r'https?://\S+', '', text)
    text = re.sub(r'www\.\S+', '', text)
    
    # Remove email addresses — matches user@domain.ext patterns
    text = re.sub(r'\S+@\S+\.\S+', '', text)
    
    # Remove IIT Jodhpur specific email patterns like:
    # name[at]iitj[dot]ac[dot]in or name[at]domain[dot]com
    # These appear extensively on faculty profile pages
    text = re.sub(r'\w+\s*\[at\]\s*\w+\s*\[dot\]\s*\w+\s*(?:\[dot\]\s*\w+)*', '', text)
    text = re.sub(r'\w+\s*\(at\)\s*\w+\s*\(dot\)\s*\w+\s*(?:\(dot\)\s*\w+)*', '', text)
    
    # Remove phone numbers (including IIT Jodhpur format: 0291 280 XXXX)
    text = re.sub(r'0291\s*\d{3}\s*\d{4}', '', text)
    text = re.sub(r'\+91[\s\-]*\d{3}[\s\-]*\d{7,}', '', text)
    text = re.sub(r'(?:phone|fax|tel|mob|call)[\s:]*[\+\d\-\(\)\s]+', '', text, flags=re.IGNORECASE)
    
    # Remove copyright notices — matches © YEAR patterns
    text = re.sub(r'©.*?(?:\n|$)', '', text)
    text = re.sub(r'Copyright.*?(?:\n|$)', '', text, flags=re.IGNORECASE)
    
    # Remove control characters (non-printable characters)
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f]', '', text)
    
    # Remove lines that are purely numbers (page numbers, etc.)
    text = re.sub(r'^[\d\s\.\-]+$', '', text, flags=re.MULTILINE)
    
    # Remove common website boilerplate and navigation text
    boilerplate_patterns = [
        r'Home\s*[|>»]\s*', r'Skip to\s+\w+', r'Search\s*\.\.\.', 
        r'Menu\s*$', r'Toggle navigation', r'Read more\s*$',
        r'Click here', r'Back to top', r'Follow us',
        r'Important links', r'CCCD @ IITJ', r'Old Website',
        r'NCCR Portal', r'View all \w+', r'Sitemap',
        r'Intranet Links', r'Web Policy', r'Web Information Manager',
        r'CERT-IN', r'Feedback', r'NIRF', r'Internal Committee',
        r'Tenders', r'Techscape', r'Recruitment', r'Correspondence',
        r'How To Reach.*', r'Institute Repository', r'Donations',
    ]
    for pattern in boilerplate_patterns:
        text = re.sub(pattern, '', text, flags=re.IGNORECASE | re.MULTILINE)
    
    # Remove standalone "email" and "call" labels from faculty listings
    text = re.sub(r'^\s*(email|call|school)\s*$', '', text, flags=re.IGNORECASE | re.MULTILINE)
    
    # Collapse multiple whitespace/newlines into single space
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r' {2,}', ' ', text)
    
    return text.strip()


# Stopwords to remove — these are function words with no semantic content.
# We keep some domain-relevant short words but remove standard English stopwords.
from nltk.corpus import stopwords
STOPWORDS = set(stopwords.words('english'))
# Also remove website-specific noise words that survive boilerplate removal
STOPWORDS.update([
    'dot', 'ac', 'iitj', 'email', 'call', 'school', 'view', 'click',
    'http', 'https', 'www', 'com', 'org', 'html', 'php', 'en', 'hi',
    'rtl', 'ltr', 'javascript', 'void', 'pdf', 'download', 'upload',
])


def tokenize_and_clean(text):
    """ Tokenize text into sentences, then clean each sentence's tokens. """
    corpus = []
    
    # Step 1: Split text into individual sentences
    sentences = sent_tokenize(text.lower())
    
    for sentence in sentences:
        # Step 2: Tokenize the sentence into individual words
        tokens = word_tokenize(sentence)
        
        # Step 3-6: Keep only alphabetic tokens of sufficient length,
        # excluding stopwords and noise tokens.
        # This removes:
        #   - Punctuation tokens (e.g., '.', ',', '!', '(', ')')
        #   - Numeric tokens (e.g., '2023', '42')
        #   - Mixed alphanumeric (e.g., 'b1', '3rd')
        #   - Single characters (e.g., 'a', 'i' — context-poor)
        #   - English stopwords (the, is, at, which, etc.)
        #   - Website artifacts (dot, ac, iitj, email, call, etc.)
        cleaned_tokens = [
            token for token in tokens
            if (token.isalpha() 
                and len(token) >= MIN_WORD_LENGTH
                and token not in STOPWORDS)
        ]
        
        # Step 7: Only keep sentences with enough tokens for meaningful context
        if len(cleaned_tokens) >= MIN_SENTENCE_LENGTH:
            corpus.append(cleaned_tokens)
    
    return corpus


# STATISTICS AND VISUALIZATION
def compute_dataset_statistics(corpus, num_documents):
    """ Compute and display comprehensive statistics about the cleaned corpus. """
    # Flatten the corpus to get all tokens in a single list
    all_tokens = [token for sentence in corpus for token in sentence]
    
    # Build vocabulary (set of unique tokens)
    vocabulary = set(all_tokens)
    
    # Count word frequencies for top-N analysis
    word_freq = Counter(all_tokens)
    
    # Calculate average sentence length
    sentence_lengths = [len(sent) for sent in corpus]
    avg_length = np.mean(sentence_lengths) if sentence_lengths else 0
    
    # Assemble all statistics into a dictionary
    stats = {
        'num_documents': num_documents,
        'num_sentences': len(corpus),
        'total_tokens': len(all_tokens),
        'vocabulary_size': len(vocabulary),
        'avg_sentence_length': round(avg_length, 2),
        'top_20_words': word_freq.most_common(20),
    }
    
    # Print formatted statistics
    print("\n" + "=" * 60)
    print("DATASET STATISTICS")
    print("=" * 60)
    print(f"  Total Documents:       {stats['num_documents']}")
    print(f"  Total Sentences:       {stats['num_sentences']:,}")
    print(f"  Total Tokens:          {stats['total_tokens']:,}")
    print(f"  Vocabulary Size:       {stats['vocabulary_size']:,}")
    print(f"  Avg Sentence Length:   {stats['avg_sentence_length']} tokens")
    print(f"\n  Top 20 Most Frequent Words:")
    for rank, (word, count) in enumerate(stats['top_20_words'], 1):
        print(f"    {rank:3d}. {word:20s} {count:>6,}")
    
    return stats


def save_statistics(stats, output_dir):
    """ Save dataset statistics to a text file for inclusion in the report. """
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, 'dataset_stats.txt')
    
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write("=" * 60 + "\n")
        f.write("DATASET STATISTICS — IIT JODHPUR CORPUS\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Total Documents:       {stats['num_documents']}\n")
        f.write(f"Total Sentences:       {stats['num_sentences']:,}\n")
        f.write(f"Total Tokens:          {stats['total_tokens']:,}\n")
        f.write(f"Vocabulary Size:       {stats['vocabulary_size']:,}\n")
        f.write(f"Avg Sentence Length:   {stats['avg_sentence_length']} tokens\n\n")
        f.write("Top 20 Most Frequent Words:\n")
        f.write("-" * 35 + "\n")
        for rank, (word, count) in enumerate(stats['top_20_words'], 1):
            f.write(f"  {rank:3d}. {word:20s} {count:>6,}\n")
    
    print(f"\n  Statistics saved to: {filepath}")


def generate_wordcloud(corpus, output_dir):
    """ Generate and save a word cloud visualization of the corpus."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Flatten corpus and join into a single string for WordCloud
    all_tokens = [token for sentence in corpus for token in sentence]
    text = ' '.join(all_tokens)
    
    # Configure the word cloud with visually appealing settings
    wc = WordCloud(
        width=1600,                # High resolution width
        height=800,                # High resolution height
        background_color='white',  # Clean white background
        colormap='viridis',        # Modern, colorblind-friendly colormap
        max_words=200,             # Show top 200 words
        min_font_size=8,           # Minimum font size for readability
        max_font_size=120,         # Maximum font size for prominent words
        random_state=42,           # Fixed seed for reproducibility
        collocations=False,        # Don't combine words into phrases
    )
    
    # Generate the word cloud from the text
    wc.generate(text)
    
    # Create the figure and display the word cloud
    fig, ax = plt.subplots(1, 1, figsize=(16, 8))
    ax.imshow(wc, interpolation='bilinear')  # Smooth rendering
    ax.axis('off')  # Hide axes for cleaner appearance
    ax.set_title('Word Cloud — IIT Jodhpur Corpus', fontsize=20, fontweight='bold', pad=15)
    
    # Save the word cloud image
    output_path = os.path.join(output_dir, 'wordcloud.png')
    fig.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    
    print(f"  Word cloud saved to: {output_path}")


# CORPUS SAVE FUNCTION
def save_corpus(corpus, output_path):
    """ Save the cleaned corpus to a text file. """
    with open(output_path, 'w', encoding='utf-8') as f:
        for sentence in corpus:
            # Write each sentence as a space-separated line
            f.write(' '.join(sentence) + '\n')
    
    print(f"\n  Cleaned corpus saved to: {output_path}")
    print(f"  Total sentences written: {len(corpus):,}")


# MAIN EXECUTION
def main():
    """ Main preprocessing pipeline. """
    print("=" * 70)
    print("TEXT PREPROCESSING PIPELINE")
    print("=" * 70)
    
    # Step 1: Load all raw text files from the data/raw/ directory
    print("\n[Step 1] Loading raw text files...")
    raw_texts = load_raw_texts(RAW_DATA_DIR)
    
    if not raw_texts:
        print("ERROR: No .txt files found in data/raw/")
        print("Please run 01_scrape_data.py first to collect the data.")
        sys.exit(1)
    
    num_documents = len(raw_texts)
    print(f"\n  Loaded {num_documents} documents")
    
    # Step 2-3: Preprocess each document
    # Apply the full cleaning pipeline to each source file
    print("\n[Step 2-3] Preprocessing documents...")
    full_corpus = []  # Will hold all cleaned sentences from all documents
    
    for filename, text in raw_texts.items():
        print(f"\n  Processing: {filename}")
        
        # Step 2a: Remove non-English text (Hindi, Devanagari, etc.)
        text = remove_non_english(text)
        print(f"    After non-English removal: {len(text):,} characters")
        
        # Step 2b: Remove boilerplate (URLs, emails, navigation text, etc.)
        text = remove_boilerplate(text)
        print(f"    After boilerplate removal: {len(text):,} characters")
        
        # Step 3: Tokenize into sentences and clean tokens
        doc_corpus = tokenize_and_clean(text)
        print(f"    Sentences extracted: {len(doc_corpus):,}")
        
        # Add this document's sentences to the full corpus
        full_corpus.extend(doc_corpus)
    
    print(f"\n  Total sentences in corpus: {len(full_corpus):,}")
    
    # Step 4: Compute and save statistics about the cleaned corpus
    print("\n[Step 4] Computing dataset statistics...")
    stats = compute_dataset_statistics(full_corpus, num_documents)
    save_statistics(stats, OUTPUT_DIR)
    
    # Step 5: Generate word cloud visualization of the cleaned corpus
    print("\n[Step 5] Generating word cloud...")
    generate_wordcloud(full_corpus, OUTPUT_DIR)
    
    # Step 6: Save the cleaned corpus to a text file for use in model training
    print("\n[Step 6] Saving cleaned corpus...")
    corpus_path = os.path.join(DATA_DIR, 'cleaned_corpus.txt')
    save_corpus(full_corpus, corpus_path)
    
    print("\n" + "=" * 70)
    print("PREPROCESSING COMPLETE")
    print("=" * 70)


if __name__ == '__main__':
    main()
