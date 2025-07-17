import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# ✅ Download only what’s required (NO punkt_tab issue)
nltk.download('stopwords')

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# ✅ Load dataset
df = pd.read_csv("Tweets.csv")  # Make sure this file exists in your folder
df = df[['text', 'airline_sentiment']]
df.columns = ['tweet', 'sentiment']  # Rename columns for clarity

# ✅ Preprocessing function (safe, no NLTK punkt dependency)
def preprocess(text):
    text = re.sub(r"http\S+|www\S+", "", text)         # Remove URLs
    text = re.sub(r"@\w+|#\w+", "", text)              # Remove mentions and hashtags
    text = re.sub(r"[^A-Za-z\s]", "", text)            # Remove special characters
    text = text.lower()                                # Lowercase
    tokens = text.split()                              # Simple tokenization
    filtered = [word for word in tokens if word not in stopwords.words('english')]
    stemmed = [PorterStemmer().stem(word) for word in filtered]
    return " ".join(stemmed)

# ✅ Apply preprocessing
df['cleaned_tweet'] = df['tweet'].astype(str).apply(preprocess)

# ✅ Encode sentiment labels: negative=0, neutral=1, positive=2
df['label'] = df['sentiment'].map({'negative': 0, 'neutral': 1, 'positive': 2})

# ✅ TF-IDF Vectorization
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(df['cleaned_tweet'])
y = df['label']

# ✅ Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ✅ Model Training
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

# ✅ Model Evaluation
y_pred = model.predict(X_test)

print("\n--- Classification Report ---")
print(classification_report(y_test, y_pred))
print("Accuracy:", accuracy_score(y_test, y_pred))

# ✅ Confusion Matrix Plot
plt.figure(figsize=(6, 5))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt="d", cmap="Blues",
            xticklabels=['Negative', 'Neutral', 'Positive'],
            yticklabels=['Negative', 'Neutral', 'Positive'])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.show()

# ✅ Predicting custom tweet
sample_tweet = "I love the new update, it's so fast and user-friendly!"
sample_clean = preprocess(sample_tweet)
sample_vec = vectorizer.transform([sample_clean])
pred = model.predict(sample_vec)[0]
sentiment_map = {0: "Negative", 1: "Neutral", 2: "Positive"}
print(f"\nSample Tweet: {sample_tweet}")
print("Predicted Sentiment:", sentiment_map[pred])