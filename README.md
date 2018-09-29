# mail-type-prediction
A basic application with necessary steps for filtering spam messages using bigram model with python language. 

N-grams are used for language modeling which is based on word prediction that is it predicts next word of a sentence from previous N-1 words. Bigram is the 2-word sequence of N-grams which predicts the next word of a sentence using the previous one word. Instead of considering the whole history of a sentence or a particular word sequence, a model like bigram can be occupied in terms of approximation of history by occupying a limited history.

Identification of a message as ‘ham’ or ‘spam’ is a classification task since the target variable has got discrete values that is ‘ham’ or ‘spam’. In order to use bigram model to assign a given message as ‘spam’ or ‘ham’, there are several steps that needed to be achieved:

##### 1.Inspecting and Separating messages given into ‘ham’ and ‘spam’ categories
Initially the data set should be inspected in order to occupy and approach to achieve the task. The given corpus of messages has flagged each message as either ham or spam. Furthermore, there are 5568 messages in a DataFrame written in English which are not null objects. Therefore tsv file can be read using DataFrame in python to classify those messages accordance with the given flag.

##### 2.Preprocessing text 
Preprocessing is the task of performing the preparation steps on the raw text corpus for efficient completion of a text-mining or Natural Language Processing or any other raw text included task. Text preprocessing consists of several steps although some of them may not apply to a particular task due to the nature of the data set available.

In this task, preprocessing of text includes the following steps in accordance with the data set:

###### Removal of Punctuation Marks

###### Converting to Lowercase
The Conversion of all characters in text into a common context such as lowercase supports to prevent identifying two words differently where one is in lowercase and the other one is not. For an instance, ‘First’ and ‘first’ should be identified as the same, therefore lowercasing all the characters makes the task easier. Moreover, the stop words are also in lowercase, so that this would make removing stop words later is also feasible.

###### Tokenizing
Tokenization is the task of breaking up text into meaningful pieces that is tokens including sentences and words. A token can be considered as an instance of a sequence of characters in a particular text that are grouped together for providing a useful semantic unit for further processing.
In this task, word tokenizing is done by matching whitespaces between words as delimiter. This is achieved in Python using regular expressions to split a string into substrings with split() function which is a basic tokenizer.  

###### Lemmatizing Words
Stemming is the process of eliminating affixes (suffixed, prefixes, infixes, circum-fixes) from a word in order to obtain its word stem. Although lemmatization is related to stemming, it differs since lemmatization is able to capture canonical forms based on the lemma of a word. Lemmatization occupies a vocabulary and morphological analysis of words which make it faster and accurate than stemming. Lemmatizing has been achieved by WordNetLemmatizer in Python language.

##### 3.Feature extraction
After preprocessing stage, the features should be extracted from the text. The features are the units that supports for the classifying task, and bigrams are the features in this task of messages classification. The bigrams or the features are extracted from the preprocessed text. Initially, the unigrams are acquired, and then those unigrams are used to obtain the unigrams in each corpus (‘ham’ and ‘spam’).

##### 4.Stop words removing 
There are certain words in a language (here English) which are necessary for a sentence or a sequence of words although they do not contribute to the meaning of a considered phrase. The Natural Language Toolkit (NLTK) library in Python provides common stop words for some languages.

Instead of removing stop words in the preprocessing step, it is done after extracting features from the corpus in order to avoid the absence the bigrams with one stop words ( ('use', 'your'), ('to', 'win') ) when acquiring the features since they have an impact on the final outcome of the application. The stop words can be ignored in this keyword-oriented Information Retrieval because the effect of it on retrieval accuracy.

##### 5.Get frequency distribution of  features
The frequency distribution is used to obtain the frequency of occurrence of each vocabulary items in a certain text. 

##### 6.Building a model for prediction
The model for classifying a given message as ‘ham’ or ‘spam’ has been approached by calculating bigram probabilities within each corpus. Initially, the given message should be preprocessed in order to progress with classification including punctuation removing, changing all characters to lowercase, tokenizing and lemmatizing. Then the bigrams are extracted from the preprocessed text for finally calculating the probability of the text to be in each corpus ‘ham’ or ‘spam’.

##### 7.Smoothing
Smoothing algorithms are occupied in order to mitigate the zero probability issue in language modeling applications. Here, Laplace (Add-1) Smoothing techniques has been used which overcomes the issue of zero probability by pretending the non-existent bigrams have been seen once before. 

Then the message is assigned ‘ham’ if the probability calculated with ‘ham’ corpus is greater than ‘spam’ corpus probability or ‘spam’ if not.

##### Assumptions:
A message being ‘ham’ or ‘spam’ depends only upon its text within the message.

There is no effect on changing all words into lowercase.

#### Improving the classification performance of the spam filter:

Analyze other details of the message to improve the result of the task:
    ISP (Internet Service Provider) path, sender/ receiver address, attached files to the message and the heading of the message.
    Occupy regular expressions and pattern matching to identify text stylings and word misspellings.
    Consider punctuation and special symbols because their frequency is higher in ‘spam’ messages.
    Consider grammatical errors and strange abbreviations (f*r*e*e) within text.
    Usage of other efficient smoothing techniques such as Interpolation, Back-off, Good-Turing Discounting instead of Laplace Smoothing to enhance the performance and the accuracy of final retrieval. 


Unigram and trigram models are also can be tested for the given data sample in order to find the most suitable technique to receive a highest accuracy. Furthermore, trigram model can be appropriate when having fixed order of words (subject – verb - object) within the language.

Remove fixed order condition for N-gram in order to have more possible combinations of words within the given text.

Use a Machine Learning classifier such as Support Vector Machine, Naïve Bayes along with advanced text preprocessing tools including Count Vectorizer, Term Frequency – Inverse Document to train the classifier.

Employ the concept of data sparsity and reserve certain portion of corpus for testing purpose of trained model.
