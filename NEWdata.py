import os
import nltk
import re
import string
import numpy as np
import pandas as pd
from nltk import word_tokenize
from nltk.util import ngrams
from collections import Counter
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier



fundedTexts = []
positivePhrases = [(re.sub(r'\d+','',line.rstrip('\n'))).rstrip('\t.') for line in open('Phrases/Gg.KS.pos')]
stop_words = set(stopwords.words('english'))
allPhrases = {}
train =[]
clean_text_tokens = []

def getTexts():
    global train
    train = pd.read_csv("Kiva Data/training_data.txt", names=['text', 'result'], delimiter="\t", quoting=3)


def processTexts():

    for text in train['text']:
        onlyLettersText = re.sub("[^a-zA-Z]"," ", text )
        text = onlyLettersText.lower()
        token = nltk.word_tokenize(text)
        bigrams = ngrams(token,2)
        trigrams = ngrams(token,3)

        # print(list(bigrams))
        tokenStrings = [ ''.join(grams) for grams in token]
        bigramStrings = [ ' '.join(grams) for grams in bigrams]
        trigramStrings = [ ' '.join(grams) for grams in trigrams]

        # vprint(text)
        # print('---------BIGRAMS:---------')
        numMatches = 0
        # removeBadPhrases(token)
        tokenStrings = removeBadPhrases(tokenStrings)
        bigramStrings = removeBadPhrases(bigramStrings)
        trigramStrings = removeBadPhrases(trigramStrings)


        clean_text_tokens.append(' '.join(tokenStrings))

        # findPositivePhrases(bigramStrings)


def removeBadPhrases(ngramStrings):
    ''' remove phrases with punctuation, solely comprised of stop words
    '''
    newNgrams = []
    for phrase in ngramStrings:
        allStop = True
        noPunct = True
        for w in phrase.split(' '):
            if w not in stop_words:
                allStop = False
            if w in string.punctuation:
                noPunct = False
            if w.startswith("'"):
                noPunct = False

        if (not allStop) and noPunct:
            newNgrams.append(phrase)
            if phrase in allPhrases: #keeping counts - might be done with bag of words so can delete later if so
                allPhrases[phrase] += 1
            else:
                allPhrases[phrase] = 1
        # else:
        #     print(phrase + " = all stop words")
    return newNgrams

#
#
# def findPositivePhrases(ngramStrings):
#     numMatches = 0
#     for ngram in ngramStrings:
#
#         if ngram in positivePhrases:
#             # print(ngram + ' = positive ')
#         # else:
#         #     print(ngram + ' !!')

def outputPhraseHash():
    phraseOut = open('phrases.txt', 'w')
    sortedPhrases = sorted(allPhrases.items() , key=lambda t : t[1] , reverse=True)
    for key, val in sortedPhrases:
        phraseOut.write(key + "\t" + str(val) + "\n")

getTexts()
processTexts()
outputPhraseHash()

######BAG OF WORDS#####

vectorizer = CountVectorizer(analyzer = "word",   \
                         tokenizer = None,    \
                         preprocessor = None, \
                         stop_words = None,   \
                         max_features = 5000)
train_data_features = vectorizer.fit_transform(clean_text_tokens)
train_data_features = train_data_features.toarray()
# print(train_data_features.shape)
# print(train_data_features)
vocab = vectorizer.get_feature_names()
# Sum up the counts of each vocabulary word
dist = np.sum(train_data_features, axis=0)

# For each, print the vocabulary word and the number of times it
# appears in the training set
# for tag, count in zip(vocab, dist):
#     print(count, tag)

# Initialize a Random Forest classifier with 100 trees
forest = RandomForestClassifier(n_estimators = 100)

# Fit the forest to the training set, using the bag of words as
# features and the sentiment labels as the response variable
#
# This may take a few minutes to run
forest = forest.fit( train_data_features, train["result"] )



#test on development set
# Read the dev test data
test = pd.read_csv("Kiva Data/test_data.txt", names=['text', 'result'], delimiter="\t", quoting=3)

# Verify that there are 25,000 rows and 2 columns

# Create an empty list and append the clean reviews one by one
num_texts = len(test["text"])
clean_test_texts = []

print("Cleaning and parsing the", num_texts, " test set texts\n")
for i in range(0,num_texts):
    if( (i+1) % 1000 == 0 ):
        print("Review %d of %d\n" % (i+1, num_texts))
    onlyLettersText = re.sub("[^a-zA-Z]"," ", test["text"][i] )
    text = onlyLettersText.lower()
    token = nltk.word_tokenize(text)
    tokenStrings = [ ''.join(grams) for grams in token]
    clean_text = ' '.join(tokenStrings)
    clean_test_texts.append( clean_text )

# Get a bag of words for the test set, and convert to a numpy array
test_data_features = vectorizer.transform(clean_test_texts)
test_data_features = test_data_features.toarray()

# Use the random forest to make sentiment label predictions
result = forest.predict(test_data_features)

# Copy the results to a pandas dataframe with an "id" column and
# a "sentiment" column
output = pd.DataFrame( data={"result":result} )

# Use pandas to write the comma-separated output file
output.to_csv( "Bag_of_Words_model.csv", header=None,index=False, quoting=3 )


