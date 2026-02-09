import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


sns.set_style("whitegrid")
plt.rcParams['figure.figsize'
] = (10,
6)

def perform_eda():
    print("Loading Data for EDA...")
    df = pd.read_csv("data.csv")
    df = df.dropna(subset=['Review text', 'Ratings'
])
    
    # Ratings Distribution
    print("Generating Ratings Distribution...")
    plt.figure()
    sns.countplot(x='Ratings', data=df, palette='viridis')
    plt.title('Distribution of Product Ratings')
    plt.xlabel('Rating (1-5)')
    plt.ylabel('Count')
    plt.savefig('eda_ratings_dist.png')
    print("Saved 'eda_ratings_dist.png'")

    # Convert to Sentiment
    def get_sentiment(r):
        if r >= 4: return "Positive"
        elif r == 3: return "Neutral"
        else: return "Negative"
    
    df['sentiment'
] = df['Ratings'
].apply(get_sentiment)

    #  Sentiment Distribution
    print("Generating Sentiment Distribution...")
    plt.figure()
    sns.countplot(x='sentiment', data=df, palette='coolwarm', order=['Negative', 'Neutral', 'Positive'
])
    plt.title('Distribution of Sentiments')
    plt.savefig('eda_sentiment_dist.png')
    print("Saved 'eda_sentiment_dist.png'")

    print("\nEDA Completed! Images saved to project directory.")

if __name__ == "__main__":
    perform_eda()
