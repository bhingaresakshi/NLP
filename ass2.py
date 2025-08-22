import nltk
from nltk.stem import WordNetLemmatizer 
from nltk.corpus import stopwords
import re

paragraph = """The news mentioned here is fake. Audience do not encourage fake news. Fake news is false or misleading"""
	

sentences = nltk.sent_tokenize(paragraph)
lemmatizer = WordNetLemmatizer()
	
corpus = []
	
for i in range(len(sentences)):
    sent = re.sub('[^a-zA-Z]', ' ', sentences[i])
    sent = sent.lower()
    sent= sent.split()
    sent = [lemmatizer.lemmatize(word) for word in sent if not word in set(stopwords.words('english'))]
    sent = ' '.join(sent)   
    corpus.append(sent)
print(corpus)

from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()
independentFeatures = cv.fit_transform(corpus).toarray()

# Creating the TF-IDF model
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer() 
independentFeatures_tfIDF = tfidf.fit_transform(corpus).toarray()

