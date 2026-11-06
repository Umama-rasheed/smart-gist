from flask import Flask, render_template, request, jsonify
import nltk
from nltk.tokenize import sent_tokenize
from collections import Counter
import re

# Download NLTK data quietly (runs on startup)
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)

app = Flask(__name__)

def get_topic(text):
    # 1. Clean text and split into words
    clean_text = re.sub(r'[^a-zA-Z\s]', '', text.lower())
    words = clean_text.split()
    
    # 2. Define stop words (added global, human, etc.)
    stop_words = set([
        "the", "and", "is", "in", "to", "of", "that", "it", "for", "on", "with", 
        "as", "by", "are", "this", "an", "be", "from", "or", "at", "which", "but", 
        "not", "have", "has", "can", "will", "their", "its", "they", "our", "we", 
        "more", "these", "such", "other", "than", "into", "only", "also", "been",
        "increase", "make", "used", "many", "good", "help", "global", "human" , "rapid"
    ])
    
    # Filter out stop words and short words
    filtered_words = [w for w in words if w not in stop_words and len(w) > 2]
    
    if not filtered_words:
        return "Unknown"

    # 3. Create 2-word phrases (bigrams)
    bigrams = [f"{filtered_words[i]} {filtered_words[i+1]}" for i in range(len(filtered_words)-1)]
    
    # 4. Count how many times words and phrases appear
    word_counts = Counter(filtered_words)
    bigram_counts = Counter(bigrams)
    
    # 5. If a 2-word phrase appears 2 or more times, pick the most frequent one
    best_bigram = None
    for bigram, count in bigram_counts.most_common(10):
        if count >= 2:
            best_bigram = bigram
            break
            
    # 6. If no phrase repeats twice, just grab the most common 2-word phrase anyway
    if not best_bigram and bigram_counts:
        best_bigram = bigram_counts.most_common(1)[0][0]
        
    # 7. Otherwise, fallback to the most common single word
    best_word = word_counts.most_common(1)[0][0] if word_counts else "Unknown"
    
    # 8. Return the best result capitalized
    result = best_bigram if best_bigram else best_word
    return result.title()

def get_gist(text):
    # Split text into sentences
    sentences = sent_tokenize(text)
    
    # Grab the first 2 sentences
    if len(sentences) >= 2:
        gist = " ".join(sentences[:2])
    else:
        gist = text
        
    # Clean up extra spaces
    return " ".join(gist.split())

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/extract', methods=['POST'])
def extract():
    data = request.get_json()
    text = data.get('text', '')
    
    # Error handling for short text
    if len(text.split()) < 10:
        return jsonify({"error": "Please enter at least 10 words."})
        
    topic = get_topic(text)
    gist = get_gist(text)
    
    return jsonify({"topic": topic, "gist": gist})

if __name__ == '__main__':
    app.run(debug=True)