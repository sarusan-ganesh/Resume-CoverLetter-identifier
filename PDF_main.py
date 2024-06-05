import os
import string
from collections import defaultdict
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn import metrics

nltk.download('stopwords')
nltk.download('wordnet')

# Load the list of English stopwords from NLTK and extend it with custom stopwords
stop_words = set(stopwords.words('english'))
additional_stopwords = {"date", "regards", "thank", "sincerely", "best"}  # Add more as needed
stop_words.update(additional_stopwords)

# Initialize the lemmatizer
lemmatizer = WordNetLemmatizer()

folder_path = 'training'

# Function to preprocess text
def preprocess_text(text):
    # Remove punctuation and numbers, convert to lowercase, and split into words
    text = text.translate(str.maketrans('', '', string.punctuation))
    words = text.split()
    words = [word.lower() for word in words if word.isalpha() and word.lower() not in stop_words]
    words = [lemmatizer.lemmatize(word) for word in words]
    return ' '.join(words)

# Lists to hold the processed texts and their corresponding labels
texts = []
labels = []

# Loop through the files in the training folder
for filename in os.listdir(folder_path):
    if filename.endswith('.txt'):
        file_path = os.path.join(folder_path, filename)
        with open(file_path, 'r', encoding='utf-8') as file:
            text = file.read()
            processed_text = preprocess_text(text)
            texts.append(processed_text)
            if 'cover letter' in filename.lower():
                labels.append('cover_letter')
            elif 'resume' in filename.lower():
                labels.append('resume')

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(texts, labels, test_size=0.3, random_state=42)

# Create a pipeline for text classification
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('clf', MultinomialNB()),
])

# Train the model
pipeline.fit(X_train, y_train)

# Predict on the test set
y_pred = pipeline.predict(X_test)

# Evaluate the model
print(metrics.classification_report(y_test, y_pred))
print("Accuracy:", metrics.accuracy_score(y_test, y_pred))
