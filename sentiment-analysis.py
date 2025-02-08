import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from wordcloud import WordCloud

# Load dataset (IMDb movie reviews dataset from local file)
df = pd.read_csv("imdb_reviews.csv")  # Ensure this file is in your project folder

# Data Preprocessing
df.dropna(inplace=True)
df['sentiment'] = df['sentiment'].map({'positive': 1, 'negative': 0})

# Generate a Word Cloud for visualization
wordcloud = WordCloud(width=800, height=400, background_color='black').generate(' '.join(df['review']))
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.title("Word Cloud of Movie Reviews")
plt.show()

# Extract features and target variable
X = df['review']
y = df['sentiment']

# Convert text to numerical features using TF-IDF
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(X).toarray()

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Random Forest Classifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
print("Classification Report:")
print(classification_report(y_test, y_pred))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Save model and dataset
pd.to_pickle(model, "sentiment_analysis_model.pkl")
pd.to_pickle(vectorizer, "tfidf_vectorizer.pkl")

print("Sentiment Analysis Project Completed Successfully! Model and vectorizer saved.")
