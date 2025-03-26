# Sentiment Analysis for YouTube Videos Using BiLSTM and GRU

## Overview
This project performs sentiment analysis on YouTube video comments using deep learning models, specifically Bidirectional LSTMs (BiLSTM) and Gated Recurrent Units (GRU). The goal is to classify comments as positive or negative based on sentiment polarity.

## Approaches to Sentiment Analysis
There are many ways to perform sentiment analysis, each with different levels of complexity and accuracy. Some common approaches include:

- **Rule-Based Approach**: Uses predefined lexicons to determine sentiment polarity based on word scores.
- **Machine Learning Approach**: Uses classification models like Decision Trees and Naïve Bayes to predict sentiment from labeled data.
- **Deep Learning Approach**: Uses neural networks, such as LSTMs and GRUs, to understand complex linguistic patterns for more accurate sentiment prediction.

### Deep Learning Approach
We use deep learning because it helps in handling sequential data more effectively. Sentiment analysis involves understanding context and dependencies between words, which deep learning models like LSTMs and GRUs excel at. These models capture long-range dependencies and contextual relationships, making them ideal for sentiment classification.
Deep learning models process large amounts of text data using artificial neural networks. These models, such as Recurrent Neural Networks (RNNs), Long Short-Term Memory Networks (LSTMs), and Gated Recurrent Units (GRUs), are capable of learning contextual relationships in text, making them highly effective for sentiment analysis.


## Text Preprocessing Techniques
Text preprocessing is a crucial step in sentiment analysis that involves cleaning and transforming text data into a structured format suitable for modeling. It includes tasks such as:

- **Lowercasing**: Converts all text to lowercase to ensure uniformity.  
  *Example:* "Book" and "book" are treated as the same word.

- **Tokenization**: Splits text into words.  
  *Example:* "I'm happy today!" → ["I", "'m", "happy", "today", "!"]

- **Stemming and Lemmatization**: Converts words to their root form.  
  *Example:* "running" → "run" (stemming), "better" → "good" (lemmatization)

- **Stopword Removal**: Eliminates common words that add little meaning.  
  *Example:* Removing "a", "the", "is" from "This is a great movie" results in "great movie".

- **Normalization**: Converts informal text to standard form.  
  *Example:* "gooood" → "good", "luv" → "love".

- **Noise Removal**: Removes unnecessary characters.  
  *Example:* In tweets, removing special characters except hashtags (e.g., "@user123 wow!! #awesome" → "wow #awesome").
Text preprocessing is a crucial step in sentiment analysis that involves cleaning and transforming text data into a structured format suitable for modeling. It includes tasks such as tokenization, stopword removal, lemmatization, and normalization to enhance the model's ability to extract meaningful patterns.
```python
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9]', ' ', text)
    words = word_tokenize(text)
    words = [w for w in words if w not in stopwords.words('english')]
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(w) for w in words]
    return ' '.join(words)
```

## Data Collection
```python
import pandas as pd
from googleapiclient.discovery import build

def get_comments(video_id, api_key):
    youtube = build('youtube', 'v3', developerKey=api_key)
    comments = []
    response = youtube.commentThreads().list(part='snippet', videoId=video_id, textFormat='plainText').execute()
    for item in response['items']:
        comments.append(item['snippet']['topLevelComment']['snippet']['textDisplay'])
    return comments
```

## Building the Deep Learning Model
Deep learning models, particularly Bidirectional LSTM (BiLSTM) and Gated Recurrent Units (GRU), are used in this project to improve sentiment analysis accuracy. These models capture sequential dependencies and context within text data, making them well-suited for understanding sentiment patterns in YouTube comments.

- **Bidirectional LSTM**: Processes input from both past and future contexts, improving accuracy.
  *Example:* If a review says, "The food was not bad," a normal LSTM might struggle, but a BiLSTM considers both words "not" and "bad" together to understand that the sentiment is positive.

- **GRU**: A simpler alternative to LSTMs, effective in capturing dependencies in long texts while reducing computation time.
Deep learning models, particularly Bidirectional LSTM (BiLSTM) and Gated Recurrent Units (GRU), are used in this project to improve sentiment analysis accuracy. These models capture sequential dependencies and context within text data, making them well-suited for understanding sentiment patterns in YouTube comments.
### Bidirectional LSTM
```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Embedding, Bidirectional, SpatialDropout1D

model = Sequential()
model.add(Embedding(5000, 128, input_length=100))
model.add(SpatialDropout1D(0.2))
model.add(Bidirectional(LSTM(100, dropout=0.2, recurrent_dropout=0.2)))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
```

### Gated Recurrent Units (GRU)
```python
from tensorflow.keras.layers import GRU

model = Sequential()
model.add(Embedding(5000, 128, input_length=100))
model.add(GRU(100, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
```

## Hyperparameter Tuning
Hyperparameter tuning is the process of optimizing model parameters such as batch size, number of epochs, and optimizer selection to achieve better performance. It helps improve accuracy and reduce overfitting by selecting the best combination of hyperparameters for the model.

- **Batch Size**: Number of samples processed before model updates.  
  *Example:* Batch size 32 vs. 64 affects speed and stability of learning.

- **Epochs**: Number of times the model sees the entire dataset.  
  *Example:* More epochs may lead to better learning but risk overfitting.

- **Optimizer Selection**: Determines how weights are updated.  
  *Example:* Adam optimizer balances speed and accuracy better than SGD.

- **Learning Rate**: Defines how quickly the model updates weights.  
  *Example:* Too high = unstable, too low = slow learning.
Hyperparameter tuning is the process of optimizing model parameters such as batch size, number of epochs, and optimizer selection to achieve better performance. It helps improve accuracy and reduce overfitting by selecting the best combination of hyperparameters for the model.
```python
from sklearn.model_selection import GridSearchCV
from keras.wrappers.scikit_learn import KerasClassifier

def build_model(optimizer='adam'):
    model = Sequential()
    model.add(Embedding(5000, 128, input_length=100))
    model.add(Bidirectional(LSTM(100, dropout=0.2, recurrent_dropout=0.2)))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model

model = KerasClassifier(build_fn=build_model, verbose=0)
param_grid = {'batch_size': [16, 32], 'epochs': [5, 10], 'optimizer': ['adam', 'rmsprop']}
grid = GridSearchCV(estimator=model, param_grid=param_grid)
grid_result = grid.fit(X_train, y_train)
```

## Model Performance & Results
| Model Type | Accuracy Before Hyperparameter Tuning | Accuracy After Hyperparameter Tuning |
|------------|------------------------------------|----------------------------------|
| Simple LSTM | 72% | 80.07% |
| Bidirectional LSTM | 74% | 86.09% |
| Bidirectional LSTM + GRU | 76% | 93.35% |
| Final Optimized Model | - | **93.35%** |

## Conclusion & Future Scope
Future work includes implementing transformer models like BERT and GPT for improved sentiment analysis performance.

