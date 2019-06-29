# from keras.preprocessing.sequence import pad_sequences
from sklearn import preprocessing
from sklearn.linear_model import SGDClassifier
import sys
from gensim.models import Word2Vec
from gensim.corpora import WikiCorpus
import pandas as pd
import numpy as np
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import re
from sklearn.decomposition import PCA
import collections

from sklearn.svm import SVC
from matplotlib import pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
import time

# stop_words = stopwords()

# print(stop_words)
start = time.time()
print("Checkpoint 1 : Load Dataset")

dataset = pd.read_csv(
    "bbc/bbc_sentiment.csv", encoding="ISO-8859-1")
dataset = dataset[["Sentiment", "SentimentText"]]
y = dataset["Sentiment"]


dataset = dataset[:500]
print(">>>>>>>>> : ", dataset.shape)
# remove citation

filtered_sentences = []
all_words = []
# print(dataset.loc["SentimentText"])


print("Checkpoint 2 : Preprocessing")

# Preprocessing
# remove non alpha numeric character
for line in dataset.SentimentText:

    datasetToken = re.sub('[0-9]', ' ', line)
    datasetToken = re.sub('[^0-9a-zA-Z]', ' ', datasetToken)
    datasetToken = re.sub('[\n]', ' ', datasetToken)
    datasetToken = word_tokenize(datasetToken)
    datasetToken = [word.lower() for word in datasetToken if not word in set(
        stopwords.words("english"))]

    # remove token which word length is less than 3 characters
    datasetToken = [word for word in datasetToken if len(word) > 3]

    # all_words contains all the words after preprocessing
    # print(datasetToken)
    all_words = all_words + datasetToken

all_words = set(all_words)

print("Checkpoint 3 : Removing Token from dataset")

# remove token  if not in in all word list
for line in dataset.SentimentText:
    split_sentence = line.split(" ")
    datasetToken = [word for word in split_sentence if word in all_words]

    filtered_sentences.append(datasetToken)


print("Checkpoint 4 : Remove 0 sentence")
filtered_sentences_X = []
filtered_sentences_y = []
for id, sentence in enumerate(filtered_sentences):
    if(len(sentence) > 0):
        filtered_sentences_X.append(filtered_sentences[id])
        filtered_sentences_y.append(y[id])


print("Checkpoint 5 : Getting max length")

# get max len of the sentence for padding
maxLen = max(len(x) for x in filtered_sentences_X)
print(" Max Length : ", maxLen)
# sg = 1 for skipgram
# negative = 10 : number of negative smapling to use
# hs = 0 : use negative sampling

print("Checkpoint 6 : Word2Vec")

word2vec = Word2Vec(filtered_sentences_X, min_count=1,
                    size=100, window=3, hs=0, negative=10, sg=1)


print("Checkpoint 7  : Pad sequence")

filtered_wordVectors = []
for id, sentence in enumerate(filtered_sentences_X):

    diff = maxLen - len(sentence)

    w2vSeq = word2vec[sentence]
    sequence = np.array(w2vSeq)

    padded_Sequence = np.pad(sequence, ((0, diff), (0, 0)), 'constant')
    res = np.reshape(padded_Sequence,
                     padded_Sequence.shape[0] * padded_Sequence.shape[1])
    # print(res.shape)

    # filtered_wordVectors.append(padded_Sequence)
    filtered_wordVectors.append(res)


filtered_wordVectors = np.array(filtered_wordVectors)


print("Checkpoint 8 : Split train Test")


xTrain, xTest, yTrain, yTest = train_test_split(
    filtered_wordVectors, filtered_sentences_y, test_size=0.2, random_state=42)


print("Trainin size  : ", xTrain.shape)
print("Test Size  : ", xTest.shape)

# le = preprocessing.LabelEncoder()
# Y_val = le.fit_transform(filtered_sentences_y)
# print(Y_val)
# print(list(le.classes_))
# sys.exit(0)


# print(filtered_wordVectors)
# print(filtered_wordVectors.shape)

print("Checkpoint 9 : Fit classifier")


clf = SGDClassifier(loss='hinge', max_iter=250,
                    learning_rate='adaptive', eta0=0.001)
clf.fit(X=xTrain, y=yTrain)

print("Checkpoint 10 : Predict Test Data")
y_pred = clf.predict(xTest)

print("Checkpoint 11 : SGD Accuracy Calculator")

acc = accuracy_score(yTest, y_pred)
print(acc)

print("Checkpoint 11 : SCV Classifier")

svcClassifier = SVC()
svcClassifier.fit(X=xTrain, y=yTrain)
y_pred2 = svcClassifier.predict(xTest)
acc2 = accuracy_score(yTest, y_pred2)
print(acc2)

print("√ : Done with Model")
end = time.time()


# this is in milliseconds
diff = end - start

# calculate to seconds
diff = diff / 60
print("√ : Total Seconds is  : ", diff)


sys.exit(0)
