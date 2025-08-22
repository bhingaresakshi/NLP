import nltk
from nltk.stem import WordNetLemmatizer 
from nltk.corpus import stopwords
import re


nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

paragraph = """The news mentioned here is fake.
 Audience do not encourage fake news. 
 Fake news is false or misleading"""


sentences = nltk.sent_tokenize(paragraph)
lemmatizer = WordNetLemmatizer()

corpus = []

for i in range(len(sentences)):
    sent = re.sub('[^a-zA-Z]', ' ', sentences[i])   # Remove non-alphabets
    sent = sent.lower()                             # Convert to lowercase
    sent = sent.split()                             # Tokenize
    sent = [lemmatizer.lemmatize(word) for word in sent if word not in set(stopwords.words('english'))]
    sent = ' '.join(sent)   
    corpus.append(sent)

print("Corpus after preprocessing:\n", corpus)

# Bag of Words Model
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()
independentFeatures = cv.fit_transform(corpus).toarray()
print("\nBag of Words (CountVectorizer):\n", independentFeatures)


from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer() 
independentFeatures_tfIDF = tfidf.fit_transform(corpus).toarray()
print("\nTF-IDF Features:\n", independentFeatures_tfIDF)

import gensim
from gensim import corpora
from gensim.utils import simple_preprocess


text2 = ["""I love programming
         Python is my favorite programming language.
         Programming allows me to solve real-world problems."""]
         

tokens2 = [[item for item in line.split()] for line in text2]


g_dict2 = corpora.Dictionary(tokens2)
print("\nThe dictionary has: " + str(len(g_dict2)) + " tokens\n")
print(g_dict2.token2id)

# Bag of Words
g_bow2 = [g_dict2.doc2bow(token, allow_update=True) for token in tokens2]
print("Bag of Words : ", g_bow2)

# Another Example with simple_preprocess
text3 = ["""I love programming
         Python is my favorite programming language.
         Programming allows me to solve real-world problems."""]

g_dict3 = corpora.Dictionary([simple_preprocess(line) for line in text3])
g_bow3 = [g_dict3.doc2bow(simple_preprocess(line)) for line in text3]

print("\nDictionary (using simple_preprocess): ")
for item in g_bow3:
    print([[g_dict3[id], freq] for id, freq in item])
