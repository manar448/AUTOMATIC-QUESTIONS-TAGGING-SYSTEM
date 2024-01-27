import pandas as pd
from bs4 import BeautifulSoup
import lxml
import re
import lxml.etree
from nltk.corpus import stopwords
from nltk.tokenize import ToktokTokenizer
from nltk.stem.wordnet import WordNetLemmatizer
import string
import pandas as pd 
import numpy as np
from sklearn.model_selection import train_test_split

questions = pd.read_csv("C:/Users/manar/nlp project/archive/Questions.csv",encoding="ISO-8859-1")
tags = pd.read_csv("C:/Users/manar/nlp project/archive/Tags.csv",encoding="ISO-8859-1", dtype={'Tag': str})

questions.drop(columns=['OwnerUserId', 'CreationDate', 'ClosedDate'], inplace=True)
#no nan values

questions['Body'] = questions['Body'].astype(str)
questions['Title'] = questions['Title'].astype(str)
tags['Tag'] = tags['Tag'].astype(str)

questions = questions.dropna(axis=0, how="all")
tags.drop_duplicates(inplace = True)
tags = tags.dropna(subset = ['Tag'])

#print(tags.isna().sum())
group_tags = tags.groupby("Id")['Tag'].apply(lambda tags: ' '.join(tags))
group_tags.head(5)
group_tags.reset_index()
group_tags.head(5)
group_tags_final = pd.DataFrame({'Id':group_tags.index, 'Tags':group_tags.values})
df = questions.merge(group_tags_final, on='Id')
df.head(5)

df = df[df['Score']>5]

x = df.iloc[:,2:4]
y = df.iloc[:,4:5]

#x_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)
#X_train, X_val, Y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state= 0)

def remove_htmltags(text):
    clean = re.compile('<.*?>')
    return re.sub(clean, '', text)

def remove_stopWords(text):
    stop_words = set(stopwords.words("english"))
    token = ToktokTokenizer()
    words = token.tokenize(text)
    filter_text = [word for word in words if word.casefold() not in stop_words]
    return ' '.join(map(str, filter_text))

def text_Lemmatizing(text):
    token = ToktokTokenizer()
    lemma = WordNetLemmatizer()
    words = token.tokenize(text)
    lemmatized_words = [lemma.lemmatize(word) for word in words]
    return ' '.join(map(str, lemmatized_words))

def remove_specialCharacters(text):
    # Define a string of punctuation characters
    punct = string.punctuation  
    # Remove all punctuation characters from the input text
    no_punct = ''.join(char for char in text if char not in punct)   
    return no_punct


'''print("X_train shape: {}".format(X_train.shape))
print("X_test shape: {}".format(X_test.shape))
print("y_train shape: {}".format(Y_train.shape))
print("y_test shape: {}".format(y_test.shape))
print("X_val shape: {}".format(X_val.shape))
print("y val shape: {}".format(y_val.shape))

x = df.iloc[:,2:4]
y = df.iloc[:,4:5]'''

#x_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)
#X_train, X_val, Y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state= 0)

def preprocessing(X):
    X['Body'] = X['Body'].apply(lambda x: remove_htmltags(x))
    X['Body'] = X['Body'].apply(lambda x: remove_stopWords(x))
    X['Body'] = X['Body'].apply(lambda x: text_Lemmatizing(x))
    X['Body'] = X['Body'].apply(lambda x: remove_specialCharacters(x))

    X['Title'] = X['Title'].apply(lambda x: str(x)) 
    X['Title'] = X['Title'].apply(lambda x: remove_specialCharacters(x)) 
    X['Title'] = X['Title'].apply(lambda x: remove_stopWords(x)) 
    X['Title'] = X['Title'].apply(lambda x: text_Lemmatizing(x))
    return X

# preprocessing x values , y is already preprocessed
'''X_train = preprocessing(X_train)
X_test = preprocessing(X_test)
X_test = preprocessing(X_val)'''
df = preprocessing(df)

from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from scipy.sparse import hstack

# Define X, y
X1 = df['Body']
X2 = df['Title']
y = df['Tags']

'''x1_test = X_test['Body']
X2_test = X_test['Title']

x1_val = X_val['Body']
X2_val = X_val['Title']

y = Y_train['Tags']
y2 = y_test['Tags']
y3 = y_val['Tags']'''
print(len(X1), len(X2), len(y))

def td_idf(x1,x2,y):
    multilabel_binarizer = MultiLabelBinarizer()
    y_bin = multilabel_binarizer.fit_transform(y)

    vectorizer_X1 = TfidfVectorizer(max_df=0.8, max_features=3000)
    vectorizer_X2 = TfidfVectorizer(max_df=0.8, max_features=3000)

    X1_tfidf = vectorizer_X1.fit_transform(x1)
    X2_tfidf = vectorizer_X2.fit_transform(x2)
    X_tfidf = hstack([X1_tfidf,X2_tfidf])

    return y_bin,X_tfidf
# Define multilabel binarizer

y_bin, X_tfidf = td_idf(X1,X1,y)
'''y_bin, X_tfidf = td_idf(X1,X1,y)
ybin_test , Xtest_tfidf = td_idf(x1_test,X2_test,y2)
ybin_val , Xval_tfidf = td_idf(x1_val,X2_val,y3)'''

print(y_bin.shape)

print(X_tfidf.shape)
x_train, X_test, y_train, y_test = train_test_split(X_tfidf, y_bin, test_size = 0.2, random_state = 0)
#X_train, X_val, Y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state= 0)

def save_model(model, filename):
    # Check if the file exists
    if os.path.exists(filename):
        # oberwrite the model to file
        joblib.dump(model, filename)
        print(f"Model Overwrited to '{filename}'.")
    else:
        # Save the model to file
        joblib.dump(model, filename)
        print(f"Model saved to '{filename}'.")

def load_model(filename):
    # Check if the file exists
    if os.path.exists(filename):
        # Load the model from file
        model = joblib.load(filename)
        print(f"Model loaded from '{filename}'.")
        return model
    else:
        print(f"File '{filename}' does not exist.")
        return None



from sklearn.metrics import accuracy_score
from sklearn.svm import LinearSVC
from sklearn.metrics import hamming_loss
from sklearn.metrics import f1_score
from skmultilearn.problem_transform import LabelPowerset
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
#from sklearn.metrics import jaccard_similarity_score
from sklearn import model_selection
from sklearn.metrics import make_scorer
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score


def print_score(y_pred, clf):
    print("Clf: ", clf.__class__.__name__)
    print("Accuracy score: {}".format(accuracy_score(y_test, y_pred)))
    print("Recall score: {}".format(recall_score(y_true=y_test, y_pred=y_pred, average='weighted')))
    print("Precision score: {}".format(precision_score(y_true=y_test, y_pred=y_pred, average='weighted')))
    print("Hamming loss: {}".format(hamming_loss(y_pred, y_test)*100))
    print("F1 score: {}".format(f1_score(y_pred, y_test, average='weighted')))
    print("---")    



from skmultilearn.problem_transform import BinaryRelevance
from sklearn.naive_bayes import GaussianNB


svc = LinearSVC()
sgd = SGDClassifier(n_jobs=-1)
# initialize binary relevance multi-label classifier
# with a gaussian naive bayes base classifier
clf = LabelPowerset(svc)

clf.fit(x_train, y_train)

y_pred = clf.predict(X_test)
print(y_pred.shape)
print(y_test.shape)

print_score(y_pred, clf)
