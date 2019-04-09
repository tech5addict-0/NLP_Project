""" Reconstruct the dataset from all the folds and separate it into train, validation and test set  """


import pandas as pd
import numpy as np
from random import shuffle
import os
import json
import nltk

# Global variables which is the corpus of english stopwords
global stopWordSet


def getAllCsvFileName(directory):
    ''' Return from a directory all the names of .csv files in a list
        These names will include the name of the directory as well
        each file name will be : directory/fileName.csv '''

    result = []
    for fileName in os.listdir(directory):
        if fileName.endswith(".csv"):
            result.append(os.path.join(directory,fileName))
    return result



def createDictFromDirectory(directory):
    ''' Create a dictionnary of all the data that will encompas every fold in the directory.
        The dictionnary will be {claimID : {'articles' : {articleID : (articleHeadline, stance) },'claim' : theClaim} }'''


    # Even if we are for now only using one file, this could be useful for the fold process for example

    
    listCsvFiles = getAllCsvFileName(directory)

    dataDict = {}

    # for each file to read
    for fileName in listCsvFiles:

        # We read the csv using pandas
        tmpCsvData = pd.read_csv(fileName)
        tmpCsvData.dropna()
        
        #print(tmpCsvData.loc[1]['claimHeadline'])

        # For each line in the dataFrame
        for ind in tmpCsvData.index:

            # We get all the necessary infos from the line
            currentLine =  tmpCsvData.loc[ind]
            claimId = currentLine['claimId']
            articleId = currentLine['articleId']
            stance = currentLine['articleHeadlineStance']

            #print(str(claimId) + "\n" + str(articleId) + "\n" "\n" + articleHeadline + "\n" + stance)

            # We add the data to the dictionnary

            # We create the claim in the dict if it is not here already
            if claimId not in dataDict:
                # We get the asssociated headline
                claimHeadline = currentLine['claimHeadline'].lstrip(" ")

                # We remove "Claim :" if it is here
                if claimHeadline[:len("Claim: ")] == "Claim: ":
                    claimHeadline = claimHeadline[len("Claim: "):]
                    print(claimHeadline)

                # We create the entry in the dictionnary
                dataDict[claimId] = {'articles' : {}, 'claim' : claimHeadline}

            # We add the article to the claim only if it's not already there
            if articleId not in dataDict[claimId]['articles']:
                # We get the article Headline
                articleHeadline = currentLine['articleHeadline']

                # We add the article
                dataDict[claimId]['articles'][articleId] = (articleHeadline, stance)
            
    return dataDict

            
                

def cutTrainTestValidationSet(dataset, sizeTrain, sizeValidation):
    ''' Return the data given as parameter in 3 sets 
        sizeTrain is in percentage, sizeValidation is also in percent (between 0 and 1)
        The returned format is a tuple : (trainClaims, validationClaims, testClaims)'''

    # We check the size
    if sizeTrain + sizeValidation >= 1:
        print("Size of the training set and the validation set are too big (bigger than 1)")
        return 0

    
    else:
        nbTrain = round(len(dataset) * sizeTrain)
        nbValidation = round(len(dataset) * sizeValidation)

        # We get the list of claims and shuffle it
        claimList = list(dataset.keys())
        shuffle(claimList)

        # We return the separated sets
        return (claimList[:nbTrain], claimList[nbTrain:nbTrain+nbValidation], claimList[nbTrain+nbValidation:])



def preprocessString(stringToProcess):
    ''' preprocess a string, it uses stemming and removal of stopwords 
    The tokens are then added together by just adding whitespace after every word'''
    tokens = nltk.word_tokenize(stringToProcess)

    # stemming :
    stemmer = nltk.stem.PorterStemmer()
    for i in range(0,len(tokens)):
        tokens[i] = stemmer.stem(tokens[i])

    # removal of stopwords
    filteredTokens = []
    for word in tokens:
        if word not in stopWordSet:
            filteredTokens.append(word)

    # We combine tokens together to create a sentence
    newSentence = " ".join(filteredTokens)
    return newSentence
    

    
    
def preprocessDataset(dataset):
    ''' take the original dataset and preprocess every article and claim headline 
        The preprocessing consists of stemming and removing stop words '''

    # We access every claim
    for claimId in dataset:
        #print(dataset[claimId])
        #preprocessString(dataset[claimId]['claim'])

        #print("ATTATAT")
        #print("Claim : " + dataset[claimId]['claim'])
        #print("Claim Id : " + str(claimId))
        
        # We process the claim
        dataset[claimId]['claim'] = preprocessString(dataset[claimId]["claim"])
        
        # For every article
        for articleId in dataset[claimId]["articles"]:
            # We process the article headline
            print(articleId)
            print(dataset[claimId]["articles"][articleId][0])
            preprocessedArticle = preprocessString(dataset[claimId]["articles"][articleId][0])
            dataset[claimId]["articles"][articleId] = (preprocessedArticle, dataset[claimId]["articles"][articleId][1])

    return dataset



# We define the stopwords list
stopWordSet = set(nltk.corpus.stopwords.words('english'))


# Writing part
#data = createDictFromDirectory("emergent/")
#with open('datasets/dataset.json',"w+") as flux:
#    json.dump(data,flux)


#Writing the preprocessed dataset
#data = createDictFromDirectory("emergent/")
#data = preprocessDataset(data)
#with open('datasets/datasetPreproc.json',"w+") as flux:
#    json.dump(data,flux)


#We read the data
#with open('datasets/dataset.json','r') as json_file:  
#   data = json.load(json_file)

# test stuff
#print(len(data))
#print("\n\n")
#(train, val, test) = cutTrainTestValidationSet(data, 0.5, 0.2)
#print(len(train))
#print(len(val))
#print(len(test))

#preprocessDataset(data)








