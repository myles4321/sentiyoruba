import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from matplotlib import style
style.use('ggplot')
import re
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))
from wordcloud import WordCloud
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

df = pd.read_csv('YOSM-main/data/yosm/dataset.csv')
df.head()
df.shape
df.info()


# In[7]:


sns.countplot(x='sentiment', data=df)
plt.title("Sentiment distribution")


# In[8]:


for i in range(5):
    print("Yoruba Review: ", [i])
    print(df['yo_review'].iloc[i], "\n")
    print("Sentiment: ", df['sentiment'].iloc[i], "\n\n")


def no_of_words(text):
    words= text.split()
    word_count = len(words)
    return word_count

df['word count'] = df['yo_review'].apply(no_of_words)
df.head()



#word count to view distribution of positive and negative reviews
fig, ax = plt.subplots(1,2, figsize=(10,6))
ax[0].hist(df[df['sentiment'] == 'positive']['word count'], label='Positive', color='blue', rwidth=0.9);
ax[0].legend(loc='upper right');
ax[1].hist(df[df['sentiment'] == 'negative']['word count'], label='Negative', color='red', rwidth=0.9);
ax[1].legend(loc='upper right');
fig.suptitle("Number of words in review")
plt.show()


#the length of positive and negative reviews
fig, ax = plt.subplots(1,2, figsize=(10,6))
ax[0].hist(df[df['sentiment'] == 'positive']['yo_review'].str.len(), label='Positive', color='orange', rwidth=0.9);
ax[0].legend(loc='upper right');
ax[1].hist(df[df['sentiment'] == 'negative']['yo_review'].str.len(), label='Negative', color='yellow', rwidth=0.9);
ax[1].legend(loc='upper right');
fig.suptitle("length of positive and negative reviews")
plt.show()


df.sentiment.replace("positive", 1, inplace=True)
df.sentiment.replace("negative", 2, inplace=True)


df.head()

def data_processing(text):
    text= text.lower()
    text = re.sub('<br />', '', text)
    text = re.sub(r"https\S+|www\S+|http\S+", '', text, flags = re.MULTILINE)
    text = re.sub(r'\@w+|\#', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    text_tokens = word_tokenize(text)
    filtered_text = [w for w in text_tokens if not w in stop_words]
    return " ".join(filtered_text)

df.sentiment = df['yo_review'].apply(data_processing)

