# coding=UTF-8
from pyquery import PyQuery as pyq
import codecs
import numpy, scipy.sparse
import pickle
import sys

modeldir = '/Users/anthony/Desktop/Courses/IR/program 1/model/'
queryfile = '/Users/anthony/Desktop/Courses/IR/program 1/queries/query-train.xml'
qType = ['number','title','question','narrative','concepts']  # query information type

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

def load_sparse_csr(filename):
    loader = numpy.load(filename)
    return scipy.sparse.csr_matrix((loader['data'], loader['indices'], loader['indptr']),
                         shape = loader['shape'])

weightMat = load_sparse_csr("weightMat.npz")
queryDict, numList = exQuery(queryfile)


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

# use tempIndex consult doc term vector
for num in numList:
    for out in ansList[num]:
        try:
            exq += weightMat[out]
        except:
            exq = weightMat[out]

exq /= len(numList)
