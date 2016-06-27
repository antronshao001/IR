# coding=UTF-8
from pyquery import PyQuery as pyq
import codecs
import numpy, scipy.sparse
import pickle
import sys

outfile = '/Users/anthony/Desktop/Courses/IR/program 1/queries/ans_out'
queryfile = '/Users/anthony/Desktop/Courses/IR/program 1/queries/query-test.xml'
modeldir = '/Users/anthony/Desktop/Courses/IR/program 1/model/'
option = False

# get infor mation from argv
arg=1
while arg<len(sys.argv):
    if sys.argv[arg] == '-r':
        option = True
        arg += 1
    else:
        if sys.argv[arg] == '-i':
            queryfile = sys.argv[arg+1]
        if sys.argv[arg] == '-o':
            outfile = sys.argv[arg+1]
        if sys.argv[arg] == '-m':
            modeldir = sys.argv[arg+1]
        if sys.argv[arg] == '-d':
            nctirdir = sys.argv[arg+1]
        arg += 2

# tfidf calculation function definition (BM25)
# test stop list = dict((k,v) for k,v in termDf.items() if v>30000)
# 20000 remove 180 common terms, 30000 remove 58 terms, 40000 remove 12 terms
def tfidf(tf,df,doclen):
    k = 2.4 #from wiki
    n = 46972.0  # dataset
    tempTf = (k + 1.0) * tf / (tf + k)
    if df > 20000: tempIdf=0  # remove stop term
    else: tempIdf = numpy.log((n-df+0.5)/(df+0.5))
    return tempTf

qType = ['number','title','question','narrative','concepts']  # query information type
qw = [0, 1.0, 1.0, 1.0, 1.0]  # query information type weight

# open vocab for term mapping
with codecs.open(modeldir+'vocab.all','r',encoding='utf-8') as files:
    vocab = files.read().split('\n')

# convert vocabs list to term mapping list(int list)
def convertString(strings):
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

# extract query into format int dict
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

def fbquery():
    return load_sparse_csr("fbquery.npz").transpose()  # csr sparse array(tdim,1)

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
    if option:
        qweightMat = scipy.sparse.csr_matrix((sparseArray[:,2],(sparseArray[:,0],sparseArray[:,1]))\
                                             ,shape=(tdim,1))+fbquery()*0.01
    else:
        qweightMat = scipy.sparse.csr_matrix((sparseArray[:,2],(sparseArray[:,0],sparseArray[:,1])),shape=(tdim,1))
    result = weightMat * qweightMat
    return result

queryDict, numList = exQuery(queryfile)
scoreMat=[]
for num in numList:
    tempPart = [SimilarityToDoc(convertString(queryDict[(num,qType[1])])) * qw[1]]
    for q in range(2,5):
        tempPart.append(SimilarityToDoc(convertString(queryDict[(num,qType[q])])) * qw[q])
    tempScore = tempPart[0]+tempPart[1]+tempPart[2]+tempPart[3]  #\
                # + tempPart[0].multiply(tempPart[1]) + tempPart[0].multiply(tempPart[2])\
                # + tempPart[0].multiply(tempPart[3]) + tempPart[1].multiply(tempPart[2])\
                # + tempPart[1].multiply(tempPart[3]) + tempPart[2].multiply(tempPart[3])\
                # + tempPart[0].multiply(tempPart[1].multiply(tempPart[2]))\
                # + tempPart[0].multiply(tempPart[1].multiply(tempPart[3]))\
                # + tempPart[0].multiply(tempPart[2].multiply(tempPart[3]))\
                # + tempPart[1].multiply(tempPart[2].multiply(tempPart[3]))\
                # + tempPart[0].multiply(tempPart[1].multiply(tempPart[2].multiply(tempPart[3])))
    scoreMat.append(numpy.array(tempScore.todense()).reshape(-1))  # append a matrix to scoreList

# Top 100 score List for each query
topScore = [score.argsort()[-100:][::-1] for score in scoreMat]  # [::-1] reverse the array

# open file-list to an array
with open(modeldir+'file-list','r') as files:
    fList = files.read().split('\n')
fList = [string[-15:].lower() for string in fList]

# output result in format
output = []
for i in range(len(topScore)):
    for fileIndex in topScore[i]:
        output.append(numList[i]+' '+fList[fileIndex]+'\n')

with open(outfile,'w') as files:
    files.writelines(output)