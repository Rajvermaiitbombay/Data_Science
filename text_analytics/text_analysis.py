# -*- coding: utf-8 -*-
"""
Created on Tue Nov 12 18:29:21 2019

@author: Rajkumar
"""
import nltk
import matplotlib.pyplot as plt
from nltk.tokenize import sent_tokenize, word_tokenize, RegexpTokenizer
from nltk.probability import FreqDist
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
import pandas as pd
from textblob import TextBlob
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
from autocorrect import spell
from string import punctuation
''' Tokenization '''
text = """Hello Mr. Smith, how are you doing today? The weather is great, and city is awesome.
            The sky is pinkish-blue. You shouldn't eat cardboard"""
''' Sentence Tokenization '''

tokenized_text=sent_tokenize(text)
print(tokenized_text)

''' Word Tokenization '''

tokenized_word=word_tokenize(text)
print(tokenized_word)
''' Frequency Distribution '''

fdist = FreqDist(tokenized_word)
print(fdist.most_common())
fdist.plot(30,cumulative=False)
plt.show()

''' Stopwords '''

stop_words=set(stopwords.words("english"))
print(stop_words)

''' Lexicon Normalization '''
''' performing stemming and Lemmatization '''

nltk.download('wordnet')
lem = WordNetLemmatizer()
stem = PorterStemmer()

word = "flying"
print("Lemmatized Word:",lem.lemmatize(word,"v"))
print("Stemmed Word:",stem.stem(word))

''' Part-of-Speech(POS) tagging '''

nltk.download('averaged_perceptron_tagger')
sent = "Albert Einstein was born in Ulm, Germany in 1879."
tokens=nltk.word_tokenize(sent)
nltk.pos_tag(tokens)

''' sentiment analysis '''
''' 1. Lexicon-based  2. Machine learning '''
'''Text Classification'''
'''
            1. Tokenization
            preprocessing text
            2. removing stopwords
            3. stemming and lemmatization
            feature engineering
            4. bag of word
            5. TF-IDF
            6. Word embadding
            7. model building
            8. model evaluation
        '''
data=pd.read_csv('files/train.tsv', sep='\t')
### EDA #####
Sentiment_count=data.groupby('Sentiment').count()
plt.bar(Sentiment_count.index.values, Sentiment_count['Phrase'])
plt.xlabel('Review Sentiments')
plt.ylabel('Number of Review')
plt.show()

'''1. preprocessing data '''
data['Phrase'] = data['Phrase'].apply(lambda x: " ".join(x.lower() for x in x.split()))
# remove special charactors
data['Phrase'] = data['Phrase'].str.replace('[^\w\s]','')
# remove numbers
data['Phrase'] = data['Phrase'].str.replace('\d+', '')
# remove white space
data['Phrase'] = data['Phrase'].str.strip()
# remove stopwords
stop = stopwords.words('english')
data['Phrase'] = data['Phrase'].apply(lambda x: " ".join(x for x in x.split() if x not in stop))
## remove punctuation
data['Phrase'] = data['Phrase'].apply(lambda x: " ".join(x for x in x.split() if x not in punctuation))
## Common word removal
freq = pd.Series(' '.join(data['Phrase']).split()).value_counts()[:10]
freq = list(freq.index)
data['Phrase'] = data['Phrase'].apply(lambda x: " ".join(x for x in x.split() if x not in freq))
## Rare words removal
freq = pd.Series(' '.join(data['Phrase']).split()).value_counts()[-10:]
freq = list(freq.index)
data['Phrase'] = data['Phrase'].apply(lambda x: " ".join(x for x in x.split() if x not in freq))
## Spelling correction
#data['Phrase'] = data['Phrase'].apply(lambda x: " ".join([str(TextBlob(word).correct()) for word in x.split()]))
#data['Phrase'] = data['Phrase'].apply(lambda x: " ".join([spell(word) for word in x.split()]))
## Lemmatization
lem = WordNetLemmatizer()
data['Phrase'] = data['Phrase'].apply(lambda x: " ".join([lem.lemmatize(word,"v") for word in x.split()]))
#data = data.drop_duplicates('Phrase').reset_index(drop=True)

'''2. Advance Text Processing '''


'''3. feature engineering '''
''' a. Bag-of-words model(BoW) '''
#tokenizer to remove unwanted elements from out data like symbols and numbers
token = RegexpTokenizer(r'[a-zA-Z0-9]+')
cv = CountVectorizer(lowercase=True,stop_words='english',
                     ngram_range = (1,1),tokenizer = token.tokenize)
text_counts= cv.fit_transform(data['Phrase'])
X_train, X_test, y_train, y_test = train_test_split(text_counts, data['Sentiment'],
                                                    test_size=0.3, random_state=1)

''' b. Term Frequency-Inverse Document Frequency (TF-IDF) '''
tf=TfidfVectorizer(lowercase=True, analyzer='word',
                   stop_words= 'english',ngram_range=(1,1))
text_tf= tf.fit_transform(data['Phrase'])
X_train, X_test, y_train, y_test = train_test_split(
    text_tf, data['Sentiment'], test_size=0.3, random_state=123)

''' 4. model building '''
clf = MultinomialNB().fit(X_train, y_train)
predicted= clf.predict(X_test)
print("MultinomialNB Accuracy:",metrics.accuracy_score(y_test, predicted))
## SVM
from sklearn.svm import SVC
text_clf_svm = SVC(kernel = 'linear', random_state = 0)
text_clf_svm.fit(X_train, y_train)
predicted = text_clf_svm.predict(X_test)
cm = metrics.confusion_matrix(y_test, predicted)
print("SVM Accuracy:",metrics.accuracy_score(y_test, predicted))
Recall = metrics.recall_score(y_test, predicted, average='weighted')
Precision = metrics.precision_score(y_test, predicted, average='weighted')

''' word cloud for bad/good reviews'''
from wordcloud import WordCloud
all_words = ' '.join([text for text in data[data['Sentiment'].isin([2])]['Phrase']])
wordcloud = WordCloud(width=800, height=500, random_state=21, max_font_size=110, background_color="white").generate(all_words)
plt.figure(figsize=(10, 7))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis('off')
plt.show()











