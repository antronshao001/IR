# coding=UTF-8
import codecs
from pyquery import PyQuery as pyq
import pickle

with open('/Users/anthony/Desktop/Courses/IR/program 1/model/file-list','r') as files:
    fList = files.read().split('\n')

fList = ['/Users/anthony/Desktop/Courses/IR/program 1/'+fileName for fileName in fList if fileName]

df=[]
for fileIndex in range(len(fList)):
    with codecs.open(fList[fileIndex]) as files:
        js = pyq(files.read())
    df.append(len(js('text').text()))

pickle.dump(df, open("dfList.pkl", "wb"))

