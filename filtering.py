#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  6 13:49:36 2018

@author: kasumi
"""
from functools import reduce

import nltk
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer
import pandas as pd
import string
import re

#Inspecting data
full_corpus = pd.read_csv('resources/tsv_files/SMSSpamCollection.tsv', sep='\t', header=None, names=['label', 'msg_body'])
#print("Input data has {} rows and {} columns".format(len(full_corpus), len(full_corpus.columns)))
#print(full_corpus.info())

# Separating messages into ham and spam
ham_text = []
spam_text = []

def separate_msgs():
    for index, column in full_corpus.iterrows():
        label = column[0]
        message_text = column[1]
        if label == 'ham':
            ham_text.append(message_text)

        elif label == 'spam':
            spam_text.append(message_text)

separate_msgs()

# Preprocessing of text

#removing punctuation marks from the email messages
def remove_msg_punctuations(email_msg):
    puntuation_removed_msg = "".join([word for word in email_msg if word not in string.punctuation])
    return puntuation_removed_msg

#converting text into lowercase and word tokenizing
def tokenize_into_words(text):
    tokens = re.split('\W+', text)
    return tokens

#lemmatizing
word_lemmatizer = WordNetLemmatizer()
def lemmatization(tokenized_words):
    lemmatized_text = [word_lemmatizer.lemmatize(word)for word in tokenized_words]
    return ' '.join(lemmatized_text)

def preprocessing_msgs(corpus):
    categorized_text = pd.DataFrame(corpus)
    categorized_text['non_punc_message_body'] = categorized_text[0].apply(lambda msg: remove_msg_punctuations(msg))
    categorized_text['tokenized_msg_body'] = categorized_text['non_punc_message_body'].apply(lambda msg: tokenize_into_words(msg.lower()))
    categorized_text['lemmatized_msg_words'] = categorized_text['tokenized_msg_body'].apply(lambda word_list: lemmatization(word_list))
    return categorized_text['lemmatized_msg_words']

# Extracting features i.e. n-grams
unigrams = []
bigrams = []
def feature_extraction(preprocessed_text):
    unigrams_lists = []
    for msg in preprocessed_text:
        # adding end of and start of a message
        msg = '<s> ' +msg +' </s>'
        unigrams_lists.append(msg.split())
    unigrams = [uni_list for sub_list in unigrams_lists for uni_list in sub_list]
    bigrams.extend(nltk.bigrams(unigrams))
    return bigrams

# removing bigrams only with stop words
stopwords = nltk.corpus.stopwords.words('english')
def filter_stopwords_bigrams(bigram_list):
    filtered_bigrams = []
    for bigram in bigram_list:
        if bigram[0] in stopwords and bigram[1] in stopwords:
            continue
        filtered_bigrams.append(bigram)
    return filtered_bigrams

# Acquiring frequencies of features
def ham_bigram_feature_frequency():
    # features frequency for ham messages
    ham_bigrams = feature_extraction(preprocessing_msgs(ham_text))
    ham_bigram_frequency = nltk.FreqDist(filter_stopwords_bigrams(ham_bigrams))
    return ham_bigram_frequency

def spam_bigram_feature_frequency():
    # features frequency for spam messages
    spam_bigrams = feature_extraction(preprocessing_msgs(spam_text))
    spam_bigram_frequency = nltk.FreqDist(filter_stopwords_bigrams(spam_bigrams))
    return spam_bigram_frequency

# calculating bigram probabilities
def bigram_probability(message):
    probability_h = 1
    probability_s = 1
    # preprocessing input messages
    punc_removed_message = "".join(word for word in message if word not in string.punctuation)
    punc_removed_message = '<s> ' +punc_removed_message +' </s>'
    tokenized_msg = re.split('\s+', punc_removed_message)
    lemmatized_msg = [word_lemmatizer.lemmatize(word)for word in tokenized_msg]
    # bigrams for message
    bigrams_for_msg = list(nltk.bigrams(lemmatized_msg))
    # stop words removed unigrams for vocabulary
    ham_unigrams = [word for word in feature_extraction(preprocessing_msgs(ham_text)) if word not in stopwords]
    spam_unigrams = [word for word in feature_extraction(preprocessing_msgs(spam_text)) if word not in stopwords]

    ham_frequency = ham_bigram_feature_frequency()
    spam_frequency  = spam_bigram_feature_frequency()
    print('========================== Calculating Probabilities ==========================')
    print('----------- Ham Freuquencies ------------')
    for bigram in bigrams_for_msg:
        # probability of first word in bigram
        ham_probability_denominator = 0
        # probability of bigram (smoothed)
        ham_probability_of_bigram = ham_frequency[bigram] + 1
        print(bigram, ' occurs ', ham_probability_of_bigram)
        vocabulary = len(set(ham_unigrams))
        for (first_unigram, second_unigram) in filter_stopwords_bigrams(ham_unigrams):
            if(first_unigram == bigram[0]):
                ham_probability_denominator += ham_frequency[first_unigram, second_unigram]
        probability = ham_probability_of_bigram / (ham_probability_denominator + (vocabulary ** 2))
        probability_h *= probability
    print('\n')
    print('----------- Spam Freuquencies ------------')
    for bigram in bigrams_for_msg:
        # probability of first word in bigram
        spam_probability_denominator = 0
        # probability of bigram (smoothed)
        spam_probability_of_bigram = spam_frequency[bigram] + 1
        print(bigram, ' occurs ', spam_probability_of_bigram)
        vocabulary = len(set(spam_unigrams))
        for (first_unigram, second_unigram) in filter_stopwords_bigrams(spam_unigrams):
            if(first_unigram == bigram[0]):
                spam_probability_denominator += spam_frequency[first_unigram, second_unigram]
        probability = spam_probability_of_bigram / (spam_probability_denominator + (vocabulary ** 2))
        probability_s *= probability
        print('\n')
    print('Ham Probability: ' +str(probability_h))
    print('Spam Probability: ' +str(probability_s))
    print('\n')
    if(probability_h >= probability_s):
        print('\"' +message +'\" is a Ham message')
    else:
        print('\"' +message +'\" is a Spam message')
    print('\n')
bigram_probability('Sorry,  ..use your brain dear')
bigram_probability('SIX chances to win CASH.')
