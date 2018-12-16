
# coding: utf-8

# In[1]:


import pandas as pd
import re, string
from nltk.tokenize.casual import TweetTokenizer
from emotionNRCDetector import extractOneEmotion
import liwc
import emoji
from collections import Counter
from sklearn import preprocessing
import numpy as np
import nltk
import gensim
import pickle
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.corpus import wordnet as wn
from nltk.stem.wordnet import WordNetLemmatizer
from gensim import corpora
import pickle
import pyLDAvis.gensim


# In[24]:


def check_duplicates(df, colname):
    '''
    Checks duplicates in a given data frame
    '''
    try:
        return pd.concat(g for _, g in df.groupby(colname) if len(g) > 1)
    except:
        return 'No duplicates'


# In[6]:


def text_preprocessing(df, column):
    '''
    Text to a lower case, http instead of a full link, @ instead of a full @-mention
    '''
    for i in range(df.shape[0]):
        df.loc[i, column] = df.loc[i, column].lower()
        df.loc[i, column] = re.sub(r"http\S*", 'http', df.loc[i, column])
        df.loc[i, column] = re.sub(r"@\S*", '@', df.loc[i, column])  


# In[80]:


def add_binary_feature(df, colname, new_colname, file):
    '''
    Creates a binary feature that is based on the presence of words from a file in text
    '''
    words = []
    try:
        with open(file) as f:
            words = [line.rstrip('\n') for line in f]
    except:
        return 'File doesn\'t exist'
    
    tknzr = TweetTokenizer()
    df[new_colname] = 0
    
    for i in range(df.shape[0]):
        t = tknzr.tokenize(df.loc[i,colname])
        common_elements = list(set(words).intersection(set(t)))
    
        if len(common_elements) > 0:
            df.loc[i,new_colname] = 1
        else:
            df.loc[i,new_colname] = 0    


# In[85]:


def add_count_feature(df, colname, new_colname, file):
    '''
    Creates a feature that is a count of words from a file in a given text
    '''
    words = []
    try:
        with open(file) as f:
            words = [line.rstrip('\n') for line in f]
    except:
        return 'File doesn\'t exist'
    
    tknzr = TweetTokenizer()
    df[new_colname] = 0
    
    for i in range(df.shape[0]):
        t = tknzr.tokenize(df.loc[i,colname])
        count = 0
        for word in t:
            if word in words:
                count += 1
        df.loc[i,new_colname] = count   


# In[9]:


def NRC_features(df, colname):
    '''
    Ads NRC features to a given dataframe
    Source of the lexicon: https://saifmohammad.com/WebPages/NRC-Emotion-Lexicon.htm
    '''
    df['NRC1'] = 0
    df['NRC2'] = 0
    df['NRC3'] = 0
    df['NRC4'] = 0
    df['NRC5'] = 0
    df['NRC6'] = 0
    df['NRC7'] = 0
    df['NRC8'] = 0
    df['NRC9'] = 0
    df['NRC10'] = 0
    for i in range(df.shape[0]):
        vector = extractOneEmotion(df.loc[i,colname])
        df.loc[i,'NRC1'] = vector[0]
        df.loc[i,'NRC2'] = vector[1]
        df.loc[i,'NRC3'] = vector[2]
        df.loc[i,'NRC4'] = vector[3]
        df.loc[i,'NRC5'] = vector[4]
        df.loc[i,'NRC6'] = vector[5]
        df.loc[i,'NRC7'] = vector[6]
        df.loc[i,'NRC8'] = vector[7]
        df.loc[i,'NRC9'] = vector[8]
        df.loc[i,'NRC10'] = vector[9]


# In[7]:


def demojize(df, colname):
    '''
    Converts emoji to text 
    '''
    for i in range(df.shape[0]):
        df.loc[i, colname] = emoji.demojize(df.loc[i,colname])


# In[25]:


def LIWC_features(df, colname, dictionary):
    """
    Adds LIWC features to a given dataframe
    Use 'LIWC2007_English100131.dic' file
    """
    parse, category_names = liwc.load_token_parser(dictionary)
    tknzr = TweetTokenizer()
    
    for i in range(len(category_names)):
        df[category_names[i]] = 0
    
    for i in range(df.shape[0]):
        t = tknzr.tokenize(df.loc[i,colname])
        features_counts = Counter(category for token in t for category in parse(token))
        for key, value in features_counts.items():
            df.loc[i,key] = value


# In[90]:


def pattern_presence(df, colname, pattern):
    """
    Adds presence of a particular pattern in text as a binary feature to a given dataframe
    'http', '#', '@' as patterns
    """
    df[pattern + '_presence'] = 0

    for i in range(df.shape[0]):
        if re.search(re.escape(pattern), df.loc[i,'text']) is not None:
            df.loc[i, pattern + '_presence'] = 1


# In[134]:


def anew_features(df, colname, file):
    '''
    Adds ANEW features to a given dataframe
    Source: Bradley, M.M., & Lang, P.J. (1999). Affective norms for English words (ANEW): Instruction manual and affective ratings.Technical Report C-1, The Center for Research in Psychophysiology, University of Florida. 
    '''
    try:
        d =  pd.read_csv(file, sep='\t')
    except:
        print('File doesn\'t exist')
        
    anew = {}
    for i in range(d.shape[0]):
        anew[d.loc[i,'Word']] = [d.loc[i,'ValMn'],d.loc[i,'AroMn'], d.loc[i,'DomMn']]
    
    tknzr = TweetTokenizer()

    df['Val'] = 0
    df['Aro'] = 0
    df['Dom'] = 0
    
    for i in range(df.shape[0]):
        val = 0
        aro = 0
        dom = 0
        
        words = tknzr.tokenize(df.loc[i,colname])
        for word in words:
            if word in anew.keys():
                val += anew[word][0]
                aro += anew[word][1]
                dom += anew[word][2]
            
        df.loc[i,'Val'] = val
        df.loc[i,'Aro'] = aro
        df.loc[i,'Dom'] = dom


# In[166]:


def normalize(df, colname, new_colname):
    '''
    Scales a column individually to unit norm
    '''
    array = np.array(df[colname])
    df[new_colname] = preprocessing.normalize([array])[0]


# In[179]:


def standardize(df, colname):
    '''
    Standardizes column values by removing the mean and scaling to unit variance
    '''
    scaler = preprocessing.StandardScaler()
    scaled = scaler.fit_transform(df[colname].reshape(-1,1))
    df[colname] = scaled


# In[194]:


def vader_sentiment(df, colname):
    '''
    Adds Vader features to a given dataframe
    '''
    df['vader_neg'] = 0
    df['vader_neu'] = 0
    df['vader_pos'] = 0
    df['vader_compound'] = 0
    
    sid = SentimentIntensityAnalyzer()
    for i in range(df.shape[0]):
        ss = sid.polarity_scores(df.loc[i,colname])
        df.loc[i,'vader_neg'] = ss['neg']
        df.loc[i,'vader_neu'] = ss['neu']
        df.loc[i,'vader_pos'] = ss['pos']
        df.loc[i,'vader_compound'] = ss['compound']





