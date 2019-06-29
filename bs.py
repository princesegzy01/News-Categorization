from gensim.models import Word2Vec
from nltk.corpus import stopwords
import bs4 as bs
import urllib.request
import re
import nltk

scrapped_data = urllib.request.urlopen(
    'https://en.wikipedia.org/wiki/Artificial_intelligence')


parsed_data = bs.BeautifulSoup(scrapped_data, 'lxml')
paragraphs = parsed_data.find_all('p')

article_text = ""

for p in paragraphs:
    # print("====================")
    article_text += p.text


# print(article_text)

# Cleaing the text
processed_article = article_text.lower()
processed_article = re.sub('[^a-zA-Z]', ' ', processed_article)
processed_article = re.sub(r'\s+', ' ', processed_article)

# Preparing the dataset
all_sentences = nltk.sent_tokenize(processed_article)

all_words = [nltk.word_tokenize(sent) for sent in all_sentences]


for i in range(len(all_words)):
    all_words[i] = [w for w in all_words[i]
                    if w not in stopwords.words('english')]
    # res = [w for w in all_words[i] if w not in stopwords.words('english')]

print(all_words)

word2vec = Word2Vec(all_words, min_count=2, size=4)
# vocabulary = list(word2vec.wv.vocab)
# print(vocabulary)

# print(word2vec["computer", "intelligence"])
