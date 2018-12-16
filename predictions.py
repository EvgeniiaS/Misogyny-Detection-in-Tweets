
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import re
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM, RNN
from keras.layers import Dropout, Activation
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras.callbacks import EarlyStopping
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import np_utils
from sklearn.model_selection import StratifiedShuffleSplit
import utility
import pickle
from keras.models import load_model


# In[8]:


def predict(df, colname = 'text'):
    '''
    Makes predictions using three pretrained models (BOW based, lexicon based, NN) and soft voting
    
    df - dataframe of test data. Can be a string, which is transformed to a dataframe
    colname - the name of a column with tweets
    '''
    
    # check the input; if a string, transform to a dataframe
    if isinstance(df, str):
        df = pd.DataFrame(data = {'text':[df]})
    
    # preprocess text: to lower case, http instead of a full link, @ instead of a full @-mention
    utility.text_preprocessing(df, colname)
    
    # predict class probabilities for three models
    prob_bow = predict_bow(df, colname)
    prob_nn = predict_nn(df, colname)
    prob_lexicon = predict_lexicon(df, colname)
    
    predictions = []

    # make a final prediction, based on the class with the highest probability
    for i in range(df.shape[0]):
        all_prob = [prob_bow[i][0], prob_bow[i][1], prob_nn[i][0], prob_nn[i][1], prob_lexicon[i][0], prob_lexicon[i][1]]
                  
        class0 = (all_prob[0] + all_prob[2] + all_prob[4]) / 3
        class1 = (all_prob[1] + all_prob[3] + all_prob[5]) / 3
        
        if class0 > class1:
            predictions.append(0)
        else:
            predictions.append(1)
            
    return predictions
    


# In[2]:


def predict_bow(df, colname = 'text'):
    '''
    Makes predictions using BOW and a pretrained ensemble model
    
    df - dataframe of test data
    colname - the name of a column with tweets
    '''
    
    # load a pretrained vectorizer and model
    with open('Models/Count_vectorizer.pickle', 'rb') as f:
        count_vectorizer = pickle.load(f)
    with open('Models/BOW_model.pickle', 'rb') as f:
        BOW_model = pickle.load(f)
        
    # transform emoji to text
    utility.demojize(df, colname)
    
    # convert tweets to a matrix of unigrams and bigrams counts
    X_test = count_vectorizer.transform(df[colname])
    
    # return predicted class probabilities
    return BOW_model.predict_proba(X_test)


# In[3]:


def predict_nn(df, colname = 'text'):    
    '''
    Makes predictions using a pretrained NN
    
    df - dataframe of test data
    colname - the name of a column with tweets
    '''
    
    # load a pretrained tokenizer and model
    with open('Models/NN_tokenizer.pickle', 'rb') as f:
        tokenizer = pickle.load(f)
    model = load_model('Models/NN_model.h5')
    
    # turn tweets to sequences and pad sequences to the length = 50
    texts_test = list(df[colname])
    sequences_test = tokenizer.texts_to_sequences(texts_test)
    data_test = pad_sequences(sequences_test, maxlen=50)
    
    # return predicted class probabilities
    return model.predict(data_test)


# In[4]:


def predict_lexicon(df, colname = 'text'):
    '''
    Makes predictions using a pretrained lexicon-based model
    
    df - dataframe of test data
    colname - the name of a column with tweets
    '''
    
    # Female genitalia words presence (from Swearing words)
    utility.add_binary_feature(df,colname,'FG', 'Data/FG.txt')
    # Male genitalia words presence (from Swearing words)
    utility.add_binary_feature(df,colname,'MG','Data/MG.txt')
    # Sex related words presence (from Swearing words)
    utility.add_binary_feature(df,colname,'SA','Data/SA.txt')
    # Swearing words presence 
    utility.add_binary_feature(df,colname,'Swearing_binary','Data/REST Swearing words.txt')
    # Extracting language features using NRC emotion lexicon (internal file)
    utility.NRC_features(df, colname)
    # Presence of a special character or link in a tweet
    utility.pattern_presence(df, colname, 'http')
    utility.pattern_presence(df, colname, '#')
    utility.pattern_presence(df, colname, '@')
    # Sentiment features using nltk.sentiment.vader
    utility.vader_sentiment(df, colname)
    # LIWC features
    utility.LIWC_features(df,colname, 'Data/LIWC2007_English100131.dic')
    
    features = ['FG', 'MG', 'SA', 'Swearing_binary', 'NRC1', 'NRC2', 'NRC3', 'NRC4', 'NRC5', 'NRC6', 'NRC7', 'NRC8', 'NRC9', 'NRC10',
     'http_presence', '#_presence', '@_presence', 'funct', 'pronoun',
     'ppron', 'i', 'we', 'you', 'shehe', 'they', 'ipron', 'article', 'verb', 'auxverb', 'past', 'present', 'future', 'adverb',
     'preps', 'conj', 'negate', 'quant', 'number', 'swear', 'social', 'family', 'friend', 'humans', 'affect', 'posemo', 'negemo',
     'anx', 'anger', 'sad', 'cogmech', 'insight', 'cause', 'discrep', 'tentat', 'certain', 'inhib', 'incl', 'excl', 'percept',
     'see', 'hear', 'feel', 'bio', 'body', 'health', 'sexual', 'ingest', 'relativ', 'motion', 'space', 'time', 'work', 'achieve',
     'leisure', 'home', 'money', 'relig', 'death', 'assent', 'nonfl', 'filler']
    
    with open('Models/Lexicon_model.pickle', 'rb') as f:
        model = pickle.load(f)
    
    # return predicted class probabilities
    return model.predict_proba(df.loc[:,features])

