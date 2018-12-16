# This NRC Detector has been developed at University of California (Santa Cruz, WA) by Golnoosh Farnadi, who kindly gave me a permission to use it for Misogyny Detection project. I slightly modified the code in order to extract emotions for a string input.


from nltk.corpus import wordnet 
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
import pickle
import re,os,sys,getopt

def getEmotionPath():
    return 'Data/emotion.p'

def extractTokens(text):
    tokens = word_tokenize(text)
    return tokens

def saveFile(path, content):
    file(path,"wb").writelines(content)

def readContent(path):    
    myfile = open(path, "r", encoding='utf-8')
    return myfile.read()

def readLines(file_name):
    items = []
    if os.path.exists(file_name):
        f = open(file_name)
        lines = f.readlines()
        f.close()
        for line in lines:
            item = line.replace('\n', '')
            items.append(item)
        return items
    else:
        return None 
    
def makeDictionary(filePath):
    stemmer = SnowballStemmer('english')
    lines = readLines(filePath)
    emotionDict = {}
    for line in lines:
        info = line.split(',')
        key = stemmer.stem(info[0])
        value = info[1:]
        emotionDict[key] = value
    pickle.dump(emotionDict, open( getEmotionPath(), "wb" ))
    

def extractEmotion(filePath):
    stemmer = SnowballStemmer('english')
    emotionVector = [0,0,0,0,0,0,0,0,0,0]
    text = readContent(filePath) 
    tokens =  extractTokens(text)  
    emotionDict = pickle.load(open(getEmotionPath(), "rb"))
    for token in tokens:
        token = stemmer.stem(token)
        if token in emotionDict.keys():
            emotionVector = [x + y for x, y in zip(emotionVector, [float(x) for x in emotionDict[token]])]
    sentiValue = emotionVector[0]+emotionVector[1]
    sumValue = sum(emotionVector[2:])
    result = []
    index = 0
    while index<10:
        if index <2:
            result.append(emotionVector[index]/float(sentiValue))
        else:
            result.append(emotionVector[index]/float(sumValue))
        index = index+1
    return result

def extractOneEmotion(text):
    stemmer = SnowballStemmer('english')
    emotionVector = [0,0,0,0,0,0,0,0,0,0]
    #text = readContent(filePath) 
    tokens =  extractTokens(text)  
    emotionDict = pickle.load(open(getEmotionPath(), "rb"))
    for token in tokens:
        token = stemmer.stem(token)
        if token in emotionDict.keys():
            emotionVector = [x + y for x, y in zip(emotionVector, [float(x) for x in emotionDict[token]])]
    sentiValue = emotionVector[0]+emotionVector[1]
    sumValue = sum(emotionVector[2:])
    result = []
    index = 0
    while index<10:
        try:
            if index <2:
                result.append(emotionVector[index]/float(sentiValue))
            else:
                result.append(emotionVector[index]/float(sumValue))
        except:
            result.append(0)
        index = index+1
    return result
    
def makeCSV(features):
    text = str(features[0])
    index = 1
    while index<len(features):
        text = text+','+str(features[index])
        index = index+1
    return text



def main(argv):
    inputTextFile = ''
    outputTextFile = ''
    try:
        opts, args = getopt.getopt(argv,"hi:o:",["ifile=","ofile="])
    except getopt.GetoptError:
        print('emotionDetector.py -i <inputTextFile> -o <outputTextFile>')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print('emotionDetector.py -i <inputTextFile> -o <outputTextFile>')
            sys.exit()
        elif opt in ("-i", "--ifile"):
            inputTextFile = arg
        elif opt in ("-o", "--ofile"):
            outputTextFile = arg
    print('Input file is ', inputTextFile)
    print('Output file is ', outputTextFile)
    result = extractEmotion(inputTextFile)
    saveFile(outputTextFile, makeCSV(result))
    

if __name__ == "__main__":
   main(sys.argv[1:])    
        

    
    
    
    
    
    
    
    
    