import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from collections import Counter
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_score, recall_score
import os

df = pd.read_csv("data/wiki_movie_plots_deduped.csv")

df.columns = df.columns.str.strip()
print(f"Dataset loaded with {len(df)} rows")
print(f"Columns: {df.columns.tolist()}")

df = df[['Plot', 'Genre']]
print(f"Rows with non-null Plot and Genre: {df.dropna().shape[0]}")

df = df[(df['Genre'].notna()) & (df['Genre'] != 'unknown')]
print(f"Rows after removing unknown/missing genres: {df.shape[0]}")

if df['Genre'].str.contains(',').any():
    print("Some genres contain commas, splitting them.")
    df['Genre'] = df['Genre'].apply(lambda x: [g.strip() for g in x.split(',')])
else:
    print("Genres don't contain commas, treating each as a single label.")
    df['Genre'] = df['Genre'].apply(lambda x: [x])

all_genres = [genre for sublist in df['Genre'] for genre in sublist]
genre_counts = Counter(all_genres)
print(f"Found {len(genre_counts)} unique genres")
print("Top 10 most common genres:")
for genre, count in genre_counts.most_common(10):
    print(f"  {genre}: {count}")

if len(df) < 1000:
    min_genre_count = 5
else:
    min_genre_count = 50

popular_genres = {g for g, c in genre_counts.items() if c >= min_genre_count}
print(f"Selected {len(popular_genres)} genres that appear at least {min_genre_count} times")

df['Genre'] = df['Genre'].apply(lambda genres: [g for g in genres if g in popular_genres])
df = df[df['Genre'].map(len) > 0]
print(f"Final dataset size: {len(df)} rows")

X_train, X_test, y_train, y_test = train_test_split(
    df['Plot'], df['Genre'], test_size=0.2, random_state=42
)
print(f"Training set: {len(X_train)} samples, Test set: {len(X_test)} samples")

mlb = MultiLabelBinarizer()
y_train_encoded = mlb.fit_transform(y_train)
y_test_encoded = mlb.transform(y_test)
print(f"Training with {len(mlb.classes_)} genre classes")

tfidf = TfidfVectorizer(
    max_features=5000,
    min_df=2,
    max_df=0.9,
    stop_words='english',
    ngram_range=(1, 2)
)
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)
print(f"TF-IDF feature matrix: {X_train_tfidf.shape}")

model = OneVsRestClassifier(
    LogisticRegression(
        C=1.0,
        class_weight='balanced',
        solver='liblinear',
        max_iter=1000,
        random_state=42
    )
)
model.fit(X_train_tfidf, y_train_encoded)
print("Model training completed")

y_pred = model.predict(X_test_tfidf)
y_pred_proba = model.predict_proba(X_test_tfidf)

f1 = f1_score(y_test_encoded, y_pred, average='micro')
precision = precision_score(y_test_encoded, y_pred, average='micro')
recall = recall_score(y_test_encoded, y_pred, average='micro')

print("\nModel Performance Metrics:")
print(f"F1 Score (micro): {f1:.4f}")
print(f"Precision (micro): {precision:.4f}")
print(f"Recall (micro): {recall:.4f}")

# Save the models
if not os.path.exists('models'):
    os.makedirs('models')

with open('models/model.pkl', 'wb') as f:
    pickle.dump(model, f)
with open('models/tfidf.pkl', 'wb') as f:
    pickle.dump(tfidf, f)
with open('models/mlb.pkl', 'wb') as f:
    pickle.dump(mlb, f)

print("Models saved successfully!")

print("\nTesting the model with examples:")
test_plots = [
    "A group of cowboys rob a train in the wild west and escape on horseback, but are pursued by a posse.",
    "A family moves to a quiet suburb but finds their new life disrupted by chaos and comedy.",
    "A young girl falls down a rabbit hole and encounters strange characters in a magical world."
]


def test_prediction(plot_summary, threshold=0.15):
    """Simple prediction function for testing model output"""
    X = tfidf.transform([plot_summary])

    try:
        decision_scores = model.decision_function(X)[0]
        results = []
        for i, score in enumerate(decision_scores):
            prob = 1 / (1 + np.exp(-score))
            if prob >= threshold:
                genre = mlb.classes_[i]
                results.append((genre, float(prob)))
    except Exception as e:
        print(f"Error in prediction: {e}")
        return ["Prediction failed"]

    results.sort(key=lambda x: x[1], reverse=True)

    if not results:
        return ["No genres met the confidence threshold"]

    return results


for i, plot in enumerate(test_plots):
    print(f"\nExample {i + 1}: {plot}")
    predictions = test_prediction(plot)
    if isinstance(predictions[0], str):
        print(predictions[0])
    else:
        print("Predicted genres:")
        for genre, score in predictions:
            print(f"- {genre}: {score:.2f}")