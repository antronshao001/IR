# coding=UTF-8
import codecs
from pyquery import PyQuery as pyq
import pickle

nctirdir = '/Users/anthony/Desktop/Courses/IR/program 1/'
modeldir = '/Users/anthony/Desktop/Courses/IR/program 1/model/'

with open(modeldir+'file-list','r') as files:
    fList = files.read().split('\n')

fList = [nctirdir+fileName for fileName in fList if fileName]

df=[]
for fileIndex in range(len(fList)):
    with codecs.open(fList[fileIndex]) as files:
        js = pyq(files.read())
    df.append(len(js('text').text()))

pickle.dump(df, open("dfList.pkl", "wb"))

