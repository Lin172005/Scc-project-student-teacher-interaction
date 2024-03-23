#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download('stopwords')
nltk.download('wordnet')

# create preprocess_text function
def preprocess_text(text):

    # Remove punctuation
    punctuationfree = "".join([i for i in text if i not in string.punctuation])

    # Tokenize the text
    tokens = word_tokenize(punctuationfree)

    # Remove stop words
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [token for token in tokens if not token.lower() in stop_words]

    # Lemmatize the tokens
    lemmatizer = WordNetLemmatizer()
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in filtered_tokens]

    # Join the tokens back into a string
    processed_text = " ".join(lemmatized_tokens)

    return processed_text


# In[ ]:


import pandas as pd

# Load the dataset
df = pd.read_csv('student_teacher_interactions.csv')

# Preprocess the text data
df['preprocessed_text'] = df['text'].apply(preprocess_text)


# In[ ]:


import matplotlib.pyplot as plt

# Count word frequencies
word_freq = pd.Series(' '.join(df['preprocessed_text']).split()).value_counts()

# Plot the top 10 words
word_freq.head(10).plot(kind='bar')
plt.show()

