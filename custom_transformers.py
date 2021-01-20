from sklearn.base import BaseEstimator
from sklearn.base import TransformerMixin
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

        
import nltk
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer 
import string
nltk.download('wordnet')
nltk.download('punkt')
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
nltk.download('stopwords')



### Custom transformer class for preprocessing & lematizing the texts
class String_preprocessor(BaseEstimator, TransformerMixin):
    def __init__(self):
        #print('\n>>>>>>>init() called.\n')
        just_to_add_a_line = 42

    def fit(self, X, y = None):
        return self
    
    def transform(self, X, y = None):

    
        X = X.copy()
        
        #Removing numbers 
        for digit in range(10):
            X = [text.replace(str(digit), ' ') for text in X]
            
            
        # Removing punctuation
        for sign in string.punctuation:
            X = [text.replace(sign, '') for text in X]
            
        ### Lowering letters 
        
        X = [text.lower() for text in X]
        
        
        # Tokenizing 

        lists_of_tokens = [word_tokenize(text) for text in X]
        
        
        ### Removing words with less than 3 letters
        
        lists_of_tokens = [[token for token in list_of_tokens if len(token) > 3] 
                        for list_of_tokens in lists_of_tokens]
        
        ### Remove 'stop words'
        
        cleaned_lists_of_tokens = [[token for token in list_of_tokens 
                                if token not in stopwords.words('english')] 
                                        for list_of_tokens 
                                    in lists_of_tokens]
        
        
        ###Lemmatize
        lemmatizer = WordNetLemmatizer()
        
        lists_of_lemmas = [[lemmatizer.lemmatize(token) for token in list_of_tokens] for list_of_tokens 
                in cleaned_lists_of_tokens]
        
        
        ### Convert list of strings to string
        new_corpus = [' '.join(list_of_lemmas) for list_of_lemmas in lists_of_lemmas]
        
        
        return new_corpus




### Custom transformer class for generating the features
class Extracting_bow_features(BaseEstimator, TransformerMixin):
    def __init__(self):
        #print('\n>>>>>>>init() called.\n')
        just_to_add_a_line = 42

    def fit(self, X, y = None):
        
        
        ### Fitting count vectorizer
        vec = CountVectorizer()
        vec.fit(X)

        self.vec = vec

        return self
    
    def transform(self, X, y = None):

        X = X.copy()

        ### Performs transformation on corpus BOW
        X = self.vec.transform(X).toarray()

        ### Normalizing by total occurrences to get frequencies of each word

        X = [vector/sum(vector) for vector in X]
        X = np.array(X)

        return X

        
    


