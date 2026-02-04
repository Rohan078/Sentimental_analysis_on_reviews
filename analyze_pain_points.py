import pandas as pd
from preprocess import clean
from sklearn.feature_extraction.text import CountVectorizer
import matplotlib.pyplot as plt
import seaborn as sns

def analyze_reviews():
    print("Loading data...")
    df = pd.read_csv("data.csv")
    df = df.dropna(subset=['Review text', 'Ratings'])
    
    # Convert ratings to sentiment
    def convert(r):
        if r >= 4:
            return "Positive"
        elif r == 3:
            return "Neutral"
        else:
            return "Negative"

    df['sentiment'] = df['Ratings'].apply(convert)
    
    print("Cleaning text...")
    df['clean'] = df['Review text'].apply(lambda x: clean(x))
    
    # Separate Positive and Negative
    pos_reviews = df[df['sentiment'] == 'Positive']['clean']
    neg_reviews = df[df['sentiment'] == 'Negative']['clean']
    
    print(f"\nTotal Positive Reviews: {len(pos_reviews)}")
    print(f"Total Negative Reviews: {len(neg_reviews)}")
    
    def get_top_n_grams(corpus, n=2, top_k=10):
        vec = CountVectorizer(ngram_range=(n, n), stop_words='english').fit(corpus)
        bag_of_words = vec.transform(corpus)
        sum_words = bag_of_words.sum(axis=0) 
        words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
        words_freq = sorted(words_freq, key = lambda x: x[1], reverse=True)
        return words_freq[:top_k]

    print("\n--- Top Pain Points (Negative Reviews Bigrams) ---")
    top_neg_bigrams = get_top_n_grams(neg_reviews, n=2, top_k=15)
    for gram, freq in top_neg_bigrams:
        print(f"{gram}: {freq}")

    print("\n--- Key Strengths (Positive Reviews Bigrams) ---")
    top_pos_bigrams = get_top_n_grams(pos_reviews, n=2, top_k=15)
    for gram, freq in top_pos_bigrams:
        print(f"{gram}: {freq}")

if __name__ == "__main__":
    analyze_reviews()
