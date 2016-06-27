# coding=UTF-8
import codecs
from pyquery import PyQuery as pyq
import pickle
import sys

nctirdir = '/Users/anthony/Desktop/Courses/IR/program 1/'
modeldir = '/Users/anthony/Desktop/Courses/IR/program 1/model/'

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

with open(modeldir+'file-list','r') as files:
    fList = files.read().split('\n')

fList = [nctirdir+fileName for fileName in fList if fileName]

df=[]
for fileIndex in range(len(fList)):
    with codecs.open(fList[fileIndex]) as files:
        js = pyq(files.read())
    df.append(len(js('text').text()))

pickle.dump(df, open("dfList.pkl", "wb"))

