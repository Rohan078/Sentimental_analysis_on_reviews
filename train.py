import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score
import pickle
from preprocess import clean

print("Loading dataset...")
df = pd.read_csv("data.csv")

#  Preprocessing

df = df.dropna(subset=['Review text', 'Ratings'])


def get_sentiment(rating):
    if rating >= 4:
        return "Positive"
    elif rating == 3:
        return "Neutral"
    else:
        return "Negative"

df['sentiment'] = df['Ratings'].apply(get_sentiment)


print("Cleaning text...")
df['clean_text'] = df['Review text'].apply(clean)

# Model Training
print("Training model...")

# Convert text to numbers (TF-IDF)
tfidf = TfidfVectorizer(max_features=5000)
X = tfidf.fit_transform(df['clean_text'])
y = df['sentiment']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Logistic Regression (Simple & Effective)
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)


print("\nModel Evaluation:")
y_pred = model.predict(X_test)
print(f"F1 Score: {f1_score(y_test, y_pred, average='weighted'):.2f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Save Model
print("Saving model files...")
pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(tfidf, open("vector.pkl", "wb"))
print("Done! Model saved as 'model.pkl' and 'vector.pkl'")
