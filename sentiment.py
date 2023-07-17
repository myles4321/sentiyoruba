import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import joblib
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv('sentiment.csv')

# Preprocess the data
nltk.download('punkt')
nltk.download('stopwords')

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df['yo_mt_review'], df['sentiment'], test_size=0.2, random_state=42)

# Define the Yoruba stopwords
yoruba_stopwords = ["a","an","bá","bí","bẹ̀rẹ̀","fún","fẹ́","gbogbo","inú","jù","jẹ","jẹ́","kan","kì","kí","kò","láti","lè","lọ","mi","mo","máa","mọ̀","ni","náà","ní",
                    "nígbà","nítorí","nǹkan","o","padà","pé","púpọ̀","pẹ̀lú","rẹ̀","sì","sí","sínú","ṣ","ti","tí","wà","wá","wọn","wọ́n","yìí","àti","àwọn","é","í",
                    "òun","ó","ń","ńlá","ṣe","ṣé","ṣùgbọ́n","ẹmọ́","ọjọ́","ọ̀pọ̀lọpọ̀"]

# Define the tokenizer with custom Yoruba stopwords
def tokenizer(text):
    tokens = nltk.word_tokenize(text)
    filtered_tokens = [token for token in tokens if token.lower() not in yoruba_stopwords]
    return filtered_tokens

df['filtered_tokens'] = df['yo_review'].apply(tokenizer)

# Vectorize the text data
vectorizer = TfidfVectorizer(tokenizer=tokenizer)
X_train_vectorized = vectorizer.fit_transform(X_train)
X_test_vectorized = vectorizer.transform(X_test)

# Train the sentiment analysis model
model = LogisticRegression()
model.fit(X_train_vectorized, y_train)

# Evaluate the model
y_pred = model.predict(X_test_vectorized)
accuracy = accuracy_score(y_test, y_pred)
print('Model Accuracy:', accuracy)

# Save the model
joblib.dump(model, 'sentiment_analysis_model.joblib')

# Load the sentiment analysis model
loaded_model = joblib.load('sentiment_analysis_model.joblib')

# Count the number of positive and negative reviews
positive_reviews = df[df['sentiment'] == 'positive'].shape[0]
negative_reviews = df[df['sentiment'] == 'negative'].shape[0]

# Create a bar chart
labels = ['Positive', 'Negative']
values = [positive_reviews, negative_reviews]

plt.bar(labels, values)
plt.xlabel('Sentiment')
plt.ylabel('Count')
plt.title('Number of Positive and Negative Reviews')
plt.show()

# Calculate the length of positive and negative reviews
positive_lengths = df[df['sentiment'] == 'positive']['yo_mt_review'].apply(len)
negative_lengths = df[df['sentiment'] == 'negative']['yo_mt_review'].apply(len)

# Create a bar chart for review lengths
labels = ['Positive', 'Negative']
values = [positive_lengths.mean(), negative_lengths.mean()]

plt.bar(labels, values)
plt.xlabel('Sentiment')
plt.ylabel('Review Length')
plt.title('Average Review Length')
plt.show()
