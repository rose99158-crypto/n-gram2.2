# First, install necessary libraries
!pip install nltk
!pip install pandas

import nltk
import pandas as pd
from collections import defaultdict, Counter
from nltk.corpus import brown
import requests
import re

# Download Brown corpus
nltk.download('brown')

# Load Brown corpus
print("Loading Brown corpus...")
words = brown.words()
print(f"Total words in Brown corpus: {len(words):,}")

# Function to create n-gram models
def create_ngram_models(corpus_words, max_n=3):
    """Create unigram, bigram, and trigram models from corpus"""
    
    # Convert to lowercase and get tokens
    tokens = [word.lower() for word in corpus_words]
    
    models = {}
    
    # Unigram model (n=1)
    print("Creating unigram model...")
    models[1] = Counter(tokens)
    
    # Bigram model (n=2)
    print("Creating bigram model...")
    bigrams = []
    for i in range(len(tokens) - 1):
        bigrams.append((tokens[i], tokens[i+1]))
    models[2] = Counter(bigrams)
    
    # Trigram model (n=3)
    print("Creating trigram model...")
    trigrams = []
    for i in range(len(tokens) - 2):
        trigrams.append((tokens[i], tokens[i+1], tokens[i+2]))
    models[3] = Counter(trigrams)
    
    return models, tokens

# Create models
models, all_tokens = create_ngram_models(words)

# Function to get top predictions for a given context
def get_predictions(context, models, top_k=5):
    """Get top predictions for a given context using different models"""
    
    # Tokenize context
    context_tokens = context.lower().split()
    
    predictions = {}
    
    # Unigram predictions (ignore context)
    unigram_preds = sorted(models[1].items(), key=lambda x: x[1], reverse=True)[:top_k]
    predictions['unigram'] = [word for word, count in unigram_preds]
    
    # Bigram predictions
    if len(context_tokens) >= 1:
        last_word = context_tokens[-1]
        # Find all bigrams starting with last_word
        bigram_counts = {}
        for (w1, w2), count in models[2].items():
            if w1 == last_word:
                bigram_counts[w2] = count
        
        if bigram_counts:
            bigram_preds = sorted(bigram_counts.items(), key=lambda x: x[1], reverse=True)[:top_k]
            predictions['bigram'] = [word for word, count in bigram_preds]
        else:
            predictions['bigram'] = predictions['unigram'][:top_k]
    else:
        predictions['bigram'] = predictions['unigram'][:top_k]
    
    # Trigram predictions
    if len(context_tokens) >= 2:
        last_two = tuple(context_tokens[-2:])
        # Find all trigrams starting with last_two words
        trigram_counts = {}
        for (w1, w2, w3), count in models[3].items():
            if (w1, w2) == last_two:
                trigram_counts[w3] = count
        
        if trigram_counts:
            trigram_preds = sorted(trigram_counts.items(), key=lambda x: x[1], reverse=True)[:top_k]
            predictions['trigram'] = [word for word, count in trigram_preds]
        else:
            # Fall back to bigram if no trigrams found
            predictions['trigram'] = predictions['bigram'][:top_k]
    else:
        predictions['trigram'] = predictions['bigram'][:top_k]
    
    return predictions

# ============================================================================
# OPTION 2: Load The Little Prince text from uploaded file
# ============================================================================
def load_text_from_file():
    """Load The Little Prince text from uploaded file in Colab"""
    print("\nPlease upload the 'lpp_en_snt_nopunct.txt' file")
    
    from google.colab import files
    uploaded = files.upload()
    
    for filename in uploaded.keys():
        print(f"Uploaded file: {filename}")
        with open(filename, 'r', encoding='utf-8') as f:
            text = f.read()
        print(f"Loaded {len(text):,} characters from {filename}")
        return text
    
    return None
# ============================================================================
# MAIN: Load The Little Prince text
# ============================================================================
print("\n" + "="*80)
print("LOADING THE LITTLE PRINCE TEXT")
print("="*80)

# Try different methods to load the text
little_prince_text = None

# First try: Load from GitHub
print("\nTrying to load from GitHub...")
little_prince_text = load_text_from_github()

# Second try: If GitHub fails, ask for file upload
if little_prince_text is None:
    print("\nGitHub loading failed. Trying file upload...")
    try:
        little_prince_text = load_text_from_file()
    except:
        print("File upload not available or failed.")

# Third try: Use fallback text
if little_prince_text is None:
    print("\nFile upload failed. Using fallback text...")
    little_prince_text = load_fallback_text()

print(f"\n✓ Successfully loaded The Little Prince text ({len(little_prince_text):,} characters)")

# ============================================================================
# List of contexts from The Little Prince
# ============================================================================
contexts = [
    "it is not my fault",
    "he was such",
    "i was discouraged",
    "the little prince never",
    "but seeds are invisible",
    "he was white with rage",
    "he had",
    "your cigarette has gone out",
    "let us look for",
    "it was"
]

# ============================================================================
# Get predictions for each context
# ============================================================================
print("\n" + "="*80)
print("SHANNON GAME PREDICTIONS FROM BROWN CORPUS")
print("="*80)

results = []

for context in contexts:
    predictions = get_predictions(context, models, top_k=5)
    results.append({
        'Context': context,
        'Unigram': ', '.join(predictions['unigram']),
        'Bigram': ', '.join(predictions['bigram']),
        'Trigram': ', '.join(predictions['trigram'])
    })
    
    # Print results for this context
    print(f"\nContext: '{context}...'")
    print(f"  Unigram predictions: {predictions['unigram']}")
    print(f"  Bigram predictions:  {predictions['bigram']}")
    print(f"  Trigram predictions: {predictions['trigram']}")

# ============================================================================
# Create and display a nice table
# ============================================================================
print("\n" + "="*80)
print("SUMMARY TABLE")
print("="*80)

df = pd.DataFrame(results)
print(df.to_string(index=False))

# ============================================================================
# Function to find actual continuations in The Little Prince text
# ============================================================================
def find_actual_continuations(context, text, max_words=5):
    """Find actual continuations of a context in The Little Prince text"""
    # Clean the text: remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    
    text_lower = text.lower()
    context_lower = context.lower()
    
    # Find all occurrences of the context
    continuations = []
    pos = 0
    
    while True:
        pos = text_lower.find(context_lower, pos)
        if pos == -1:
            break
        
        # Get the position after the context
        after_context = pos + len(context_lower)
        
        # Get the next few words
        remaining_text = text[after_context:].strip()
        # Split by words and take the first few
        words = remaining_text.split()
        if len(words) > 0:
            next_words = ' '.join(words[:max_words])
            # Clean up: remove punctuation at the end if needed
            next_words = next_words.strip('.,;:!?')
            if next_words:
                continuations.append(next_words)
        
        pos += 1  # Move forward to find next occurrence
    
    # Return unique continuations
    unique_continuations = list(set(continuations))
    return unique_continuations[:3]  # Return top 3 unique continuations

# ============================================================================
# Compare predictions with actual text from The Little Prince
# ============================================================================
print("\n" + "="*80)
print("COMPARING PREDICTIONS WITH ACTUAL TEXT FROM 'THE LITTLE PRINCE'")
print("="*80)

comparison_results = []

for context in contexts:
    predictions = get_predictions(context, models, top_k=5)
    actual = find_actual_continuations(context, little_prince_text, max_words=3)
    
    print(f"\nContext: '{context}...'")
    print(f"  Actual continuations in The Little Prince: {actual if actual else ['Not found in text']}")
    print(f"  Trigram predictions: {predictions['trigram']}")
    
    # Check if any trigram predictions match actual continuations
    matches = []
    for pred in predictions['trigram']:
        for actual_cont in actual:
            # Clean the actual continuation for comparison
            actual_first_word = actual_cont.split()[0].lower().strip('.,;:!?') if actual_cont else ""
            if actual_first_word and pred.lower() == actual_first_word:
                matches.append(pred)
    
    if matches:
        print(f"  ✓ Trigram matches actual: {matches}")
    else:
        print(f"  ✗ No trigram predictions match actual continuations")
    
    comparison_results.append({
        'Context': context,
        'Actual Next Word(s)': actual[0][:20] + '...' if actual else 'Not found',
        'Trigram Predictions': ', '.join(predictions['trigram'][:3]),
        'Match?': '✓' if matches else '✗'
    })

# ============================================================================
# Create a final summary table
# ============================================================================
print("\n" + "="*80)
print("FINAL COMPARISON TABLE")
print("="*80)

final_df = pd.DataFrame(comparison_results)
print(final_df.to_string(index=False))

# ============================================================================
# Additional Analysis: Which model performs best?
# ============================================================================
print("\n" + "="*80)
print("MODEL PERFORMANCE ANALYSIS")
print("="*80)

# Count how many times each model's top prediction matches actual text
unigram_matches = 0
bigram_matches = 0
trigram_matches = 0
total_contexts = len(contexts)

for context in contexts:
    predictions = get_predictions(context, models, top_k=1)  # Get only top prediction
    actual = find_actual_continuations(context, little_prince_text, max_words=1)
    
    if actual:
        actual_first_word = actual[0].split()[0].lower().strip('.,;:!?') if actual[0] else ""
        
        # Check unigram
        if predictions['unigram'] and predictions['unigram'][0].lower() == actual_first_word:
            unigram_matches += 1
        
        # Check bigram
        if predictions['bigram'] and predictions['bigram'][0].lower() == actual_first_word:
            bigram_matches += 1
        
        # Check trigram
        if predictions['trigram'] and predictions['trigram'][0].lower() == actual_first_word:
            trigram_matches += 1

print(f"\nTop-1 Prediction Accuracy (compared to actual text):")
print(f"  Unigram model: {unigram_matches}/{total_contexts} ({unigram_matches/total_contexts*100:.1f}%)")
print(f"  Bigram model:  {bigram_matches}/{total_contexts} ({bigram_matches/total_contexts*100:.1f}%)")
print(f"  Trigram model: {trigram_matches}/{total_contexts} ({trigram_matches/total_contexts*100:.1f}%)")

# ============================================================================
# Save results to files for download
# ============================================================================
print("\n" + "="*80)
print("SAVING RESULTS")
print("="*80)

# Save the predictions table
df.to_csv('predictions_table.csv', index=False)
print("✓ Saved predictions table to 'predictions_table.csv'")

# Save the comparison table
final_df.to_csv('comparison_table.csv', index=False)
print("✓ Saved comparison table to 'comparison_table.csv'")

# Save The Little Prince text for reference
with open('little_prince_text.txt', 'w', encoding='utf-8') as f:
    f.write(little_prince_text[:5000] + "\n\n[...text truncated for brevity...]")
print("✓ Saved sample text to 'little_prince_text.txt'")

print("\nFiles available for download:")
print("1. predictions_table.csv - All model predictions")
print("2. comparison_table.csv - Comparison with actual text")
print("3. little_prince_text.txt - Sample of The Little Prince text")

# ============================================================================
# Optional: Display some sample text for verification
# ============================================================================
print("\n" + "="*80)
print("SAMPLE TEXT FROM THE LITTLE PRINCE (for verification)")
print("="*80)
print(little_prince_text[:500] + "...")
