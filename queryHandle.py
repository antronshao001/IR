# coding=UTF-8
from pyquery import PyQuery as pyq
import codecs
import numpy, scipy.sparse
import pickle

queryfile = '/Users/anthony/Desktop/Courses/IR/program 1/queries/query-train.xml'
modeldir = '/Users/anthony/Desktop/Courses/IR/program 1/model/'

# tfidf calculation function definition (BM25)
# test stop list = dict((k,v) for k,v in termDf.items() if v>30000)
# 20000 remove 180 common terms, 30000 remove 58 terms, 40000 remove 12 terms
def tfidf(tf,df,doclen):
    k = 1.8 #from wiki
    n = 46972.0  # dataset
    tempTf = (k + 1.0) * tf / (tf + k)
    if df > 10000: tempIdf=0  # remove stop term
    else: tempIdf = numpy.log((n-df+0.5)/(df+0.5))
    return tempTf

qType = ['number','title','question','narrative','concepts']  # query information type
qw = [0, 1.0, 1.0, 1.0, 1.0]  # query information type weight

# open vocab for term mapping
with codecs.open(modeldir+'vocab.all','r',encoding='utf-8') as files:
    vocab = files.read().split('\n')

# convert vocabs list to term mapping list(int list)
def convertString(strings):
    # convert string to int list
    index=0
    indexList = []
    while index < len(strings):
        shift=1
        while shift < 10:
            try:
                tempIndex=vocab.index(strings[index:index+shift])
                indexList.append(tempIndex)
                break
            except ValueError:
                pass
            shift+=1
        index+=shift
    return indexList

# extract query into format string dict
def exQuery(filename):
    # open query html format
    with codecs.open(filename) as file:
        js = pyq(file.read())
    # out put formal string ((queryNUmber,dataType):string, ...)
    queryCount = len(js('topic'))
    queryDict = {}
    numList = []
    for topics in range(0,queryCount):
        try:
            queryNumber = js('number').eq(topics).text()[-3:]
            numList.append(queryNumber)
        except ValueError:
            break
        for dataType in qType:
            queryDict[(queryNumber,dataType)] = js(dataType).eq(topics).text()
    return queryDict,numList

# function for scipy.sparse csr type matrix loading
def load_sparse_csr(filename):
    loader = numpy.load(filename)
    return scipy.sparse.csr_matrix((loader['data'], loader['indices'], loader['indptr']),
                         shape = loader['shape'])

# load pre-handling data
termDf = pickle.load(open("termDf.pkl","rb"))  # for df
termIndex = pickle.load(open("termIndex.pkl","rb"))  # for term index
weightMat = load_sparse_csr("weightMat.npz")

# use indexList to retrive its doc weight(similarity score of each doc)
def SimilarityToDoc(indexList):
    #first derive all possible term combination, create dict{(term,tuple):count}
    doclen=len(indexList)
    termDict = {}
    checkNum=len(indexList)
    for i in range(0,checkNum):
        try:  # term exist in files
            df = termDf[(indexList[i],-1)]
            try:  # term exist in termDict
                termDict[(indexList[i],-1)] += 1
            except KeyError:  # term not exist
                termDict[(indexList[i],-1)] = 1
        except KeyError:  # term not exist in files
            pass
        if i != checkNum-1:
            try:  # term exist in files
                df = termDf[(indexList[i],indexList[i+1])]
                try:  # term exist in termDict
                    termDict[(indexList[i],indexList[i+1])] += 1
                except KeyError:  # term not exist
                    termDict[(indexList[i],indexList[i+1])] = 1
            except KeyError:  # term not exist in files
                pass
    #then for each term calculate TFIDF
    tdim = len(termDf)  # term number
    dataMatrix = []  # [t,0,count]
    for term in termDict:
        df = termDf[term]
        tf = termDict[term]
        weight = tfidf(tf,df,doclen)
        dataMatrix.append([termIndex.index(term),0,weight])
    sparseArray = numpy.array(dataMatrix)
    qweightMat = scipy.sparse.csr_matrix((sparseArray[:,2],(sparseArray[:,0],sparseArray[:,1])),shape=(tdim,1))
    result = weightMat * qweightMat
    return result

queryDict, numList = exQuery(queryfile)
scoreMat=[]
for num in numList:
    tempScore = SimilarityToDoc(convertString(queryDict[(num,qType[1])])) * qw[1]
    for q in range(2,5):
        tempScore += SimilarityToDoc(convertString(queryDict[(num,qType[q])])) * qw[q]
    scoreMat.append(numpy.array(tempScore.todense()).reshape(-1))  # append a matrix to scoreList
# Top 100 score List for each query
topScore = [score.argsort()[-100:][::-1] for score in scoreMat]  # [::-1] reverse the array


# open file-list to an array
with open(modeldir+'file-list','r') as files:
    fList = files.read().split('\n')
fList = [string[-15:].lower() for string in fList]

# open query-file ans for MAP calculation
with open('/Users/anthony/Desktop/Courses/IR/program 1/queries/ans-train','r') as files:
    ans = files.read().split('\n')
# convert ans to dict[list(file)]
ansDict={}
for line in ans:
    try:
        tempString = [string for string in line.split(' ')]
        if len(tempString) == 2:
            try:
                ansDict[tempString[0]].append(tempString[1])
            except KeyError:
                ansDict[tempString[0]]=[tempString[1]]
    except ValueError:
        continue
# convert to ansList with {query:[file-index list]}
ansList={}
for k,v in ansDict.items():
    for string in v:
        try:
            tempIndex = fList.index(string)
            try:
                ansList[k].append(tempIndex)
            except KeyError:
                ansList[k]=[tempIndex]
        except:
            continue

# MAP calculation for all queries
MAP=[]
for i in range(len(numList)):
    tempIndex = []
    for ans in ansList[numList[i]]:
        temp = numpy.argwhere(topScore[i] == ans)
        if temp:
            tempIndex.append(temp[0][0]+1)
    start_pad = len(tempIndex)
    end_pad = len(ansList[numList[i]])
    for pad in range(start_pad,end_pad):
        tempIndex.append(101)
    tempIndex.sort()
    AP = [(float(j)+1.0)/float(tempIndex[j]) for j in range(end_pad)]
    result = 0.0
    print AP
    for t in AP:
        result += t
    MAP.append(result/len(AP))

# calculate MAP to tune parameters
# precision when hit and average all hit -> AP, average all AP ->MAP
# set no-hit hit index to 101
print MAP

