# coding=UTF-8
from pyquery import PyQuery as pyq
import codecs
import copy
import numpy, scipy.sparse
import pickle

class invDoc:
    def __init__(self,docFrequency,fileCount=[]):
        self.df = docFrequency
        self.fc = fileCount  # fileCount[][] = [[fid,count],...]

# tfidf calculation function definition (BM25)
def tfidf(tf,df,doclen):
    k = 1.5  #from wiki
    b = 0.75
    avgdoc = 54.0  # calculated from sparseArray[:,2].sum()/len(invFileList)
    n = 46972.0  # dataset
    tempTf = (k + 1.0) * tf / (tf + k * (1.0 - b + b * doclen / avgdoc))
    if df > 30000: tempIdf=0  # remove stop term
    else: tempIdf = numpy.log((n-df+0.5)/(df+0.5))
    return tempTf * tempIdf

# open inv-file to a consult index dict
with codecs.open('/Users/anthony/Desktop/Courses/IR/program 1/model/inverted-file','r',encoding='utf-8') as files:
    invFile = files.read().split('\n')
    invFileList = {}  # inverse file dict {(word1,word2):invDoc}
    termIndex = []  # term index table for consulting
    termDf ={}
    for line in invFile:
        try:
            tempInt = [int(temp) for temp in line.split(' ')]
        except ValueError:
            continue
        if len(tempInt) == 3:
            tempInvDoc = copy.deepcopy(invDoc(tempInt[2]))  # temp inverse doc for term count in doc
            tempTerm = tuple(tempInt[0:2])  # temp term tuple dict save
            invFileList[tempTerm] = tempInvDoc
            termDf[tempTerm] = tempInt[2]
            termIndex.append(tempTerm)
        elif len(tempInt) == 2:
            tempInvDoc.fc.append(tempInt)  # update with same invDoc + newDoc
            invFileList[tempTerm] = tempInvDoc  # update with the same term
        else:
            print 'error at line'+str(line)

# open file-list to an array
with open('/Users/anthony/Desktop/Courses/IR/program 1/model/file-list','r') as files:
    fList = files.read().split('\n')
    if len(fList[-1]) < 2: del fList[-1]

#create a file tfidf sparse matrix       term(,)->term index -> fid, count
tdim = len(invFileList)  # term number
fdim = len(fList)  # file number
dataMatrix = []  # [f,t,weight]
dfList = pickle.load(open("dfList.pkl", "rb"))
for t in range(tdim):
    for fidCount in invFileList[termIndex[t]].fc:
        weight = tfidf(fidCount[1], invFileList[termIndex[t]].df, dfList[fidCount[0]])
        dataMatrix.append([fidCount[0],t,weight])

sparseArray = numpy.array(dataMatrix)
fileTfMatrix = scipy.sparse.csr_matrix((sparseArray[:,2],(sparseArray[:,0],sparseArray[:,1])),shape=(fdim,tdim))

# function for scipy.sparse csr type matrix saving
def save_sparse_csr(filename,array):
    numpy.savez(filename, data = array.data ,indices=array.indices,
             indptr =array.indptr, shape=array.shape)

####### output temp data #########
pickle.dump(termDf, open("termDf.pkl", "wb"))
pickle.dump(termIndex, open("termIndex.pkl", "wb"))
save_sparse_csr("weightMat", fileTfMatrix)
# dataMatrix = numpy.load("dataMat.npy")