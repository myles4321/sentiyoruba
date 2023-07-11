import pandas as pd
import streamlit as st
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib
from io import BytesIO
import chardet

# Load the dataset
df = pd.read_csv('sentiment.csv')

# Preprocess the data
nltk.download('punkt')
nltk.download('stopwords')

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df['yo_mt_review'], df['sentiment'], test_size=0.2, random_state=42)

# Vectorize the text data
vectorizer = TfidfVectorizer()
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

st.header('Sentiment Analysis')

with st.expander('Analyze Text'):
    text = st.text_input('Text here: ')
    if text:
        sentiment = loaded_model.predict(vectorizer.transform([text]))[0]  # Reshape not required
        st.write('Sentiment:', sentiment)

with st.expander('Analyze CSV'):
    upl = st.file_uploader('Upload file')

    def analyze_sentiment(x):
        sentiment = loaded_model.predict(vectorizer.transform([x]))[0]  # Reshape not required
        return sentiment

    if upl:
        content = upl.read()
        result = chardet.detect(content)
        encoding = result['encoding']

        try:
            df = pd.read_csv(BytesIO(content), encoding=encoding)
        except UnicodeDecodeError:
            df = pd.read_csv(BytesIO(content), encoding='latin-1')

        column_heading = df.columns[0]  # Get the first column heading

        df['analysis'] = df[column_heading].apply(analyze_sentiment)
        st.write(df.head(10))

        @st.cache
        def convert_df(df):
            # IMPORTANT: Cache the conversion to prevent computation on every rerun
            return df.to_csv().encode('utf-8')

        csv = convert_df(df)

        st.download_button(
            label="Download data as CSV",
            data=csv,
            file_name='sentiment.csv',
            mime='text/csv',
        )
