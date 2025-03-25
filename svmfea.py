import pandas as pd
import numpy as np
import string
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

# Download stopwords
nltk.download('stopwords')
from nltk.corpus import stopwords

# Load the dataset (Spam CSV)
df = pd.read_csv("spam.csv", encoding='latin-1')
df = df.drop(['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], axis=1)
df.columns = ['label', 'message']
df['label'] = df['label'].map({'ham': 0, 'spam': 1})  # ham -> 0, spam -> 1

# Text Cleaning Function
stop_words = set(stopwords.words('english'))
def clean_text(text):
    text = text.lower()  # Lowercase
    text = ''.join([char for char in text if char not in string.punctuation])  # Remove punctuation
    text = ' '.join([word for word in text.split() if word not in stop_words])  # Remove stopwords
    return text

df['cleaned_message'] = df['message'].apply(clean_text)

# --- Feature Engineering ---
# Adding new features based on the message content
df['word_count'] = df['message'].apply(lambda x: len(x.split()))
df['char_count'] = df['message'].apply(lambda x: len(x))
df['special_char_count'] = df['message'].apply(lambda x: sum([1 for char in x if char in string.punctuation]))
df['uppercase_ratio'] = df['message'].apply(lambda x: sum(1 for c in x if c.isupper()) / (len(x) + 1))
df['avg_word_length'] = df['message'].apply(lambda x: np.mean([len(word) for word in x.split()]))

# Display engineered features
print(df.head())

# --- Vectorization with TF-IDF ---
tfidf_vectorizer = TfidfVectorizer(max_features=5000)
X_tfidf = tfidf_vectorizer.fit_transform(df['cleaned_message']).toarray()

# Combine TF-IDF features with newly engineered features
X_additional_features = df[['word_count', 'char_count', 'special_char_count', 'uppercase_ratio', 'avg_word_length']].values
X = np.hstack((X_tfidf, X_additional_features))
y = df['label'].values

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# --- Train SVM Classifier ---
svm_model = SVC(kernel='linear')
svm_model.fit(X_train, y_train)

# --- Evaluate Performance ---
y_pred = svm_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'\nAccuracy: {accuracy * 100:.2f}%')
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
