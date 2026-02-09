import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score
import pickle
from preprocess import clean
import mlflow
import mlflow.sklearn


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

mlflow.set_experiment("Sentiment_Analysis")

with mlflow.start_run(run_name="tfidf_logistic"):

    # ----- LOG PARAMETERS -----
    mlflow.log_param("vectorizer", "TFIDF")
    mlflow.log_param("model", "LogisticRegression")

    # your existing code
    model.fit(X_train, y_train)

    preds = model.predict(X_test)

    # ----- LOG METRICS -----
    from sklearn.metrics import accuracy_score, f1_score

    mlflow.log_metric("accuracy", accuracy_score(y_test, preds))
    mlflow.log_metric("f1_score", f1_score(y_test, preds, average="weighted"))

    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

    mlflow.log_metric("accuracy", accuracy_score(y_test, preds))

    mlflow.log_metric("f1_weighted", f1_score(y_test, preds, average="weighted"))

    mlflow.log_metric("precision", precision_score(y_test, preds, average="weighted"))

    mlflow.log_metric("recall", recall_score(y_test, preds, average="weighted"))
    
    # ----- LOG ARTIFACTS (Confusion Matrix Plot) -----
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.metrics import confusion_matrix
    
    cm = confusion_matrix(y_test, preds)
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.savefig("confusion_matrix.png")
    mlflow.log_artifact("confusion_matrix.png")
    plt.close()

    # Log report as text
    with open("classification_report.txt", "w") as f:
        f.write(classification_report(y_test, preds))
    mlflow.log_artifact("classification_report.txt")

    # ----- LOG MODEL & REGISTER -----
    mlflow.sklearn.log_model(
        sk_model=model, 
        artifact_path="sentiment_model",
        registered_model_name="SentimentAnalysisModel"
    )
    print("MLflow logging complete.")
