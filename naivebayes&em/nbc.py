import os
import sys
import string
import copy
import numpy as np
import pickle
from stemming.porter2 import stem
#from nltk.stem.lancaster import LancasterStemmer

data_path = '/Users/anthony/Desktop/Courses/IR/pg2/20news'
out_file = 'result'
partial_data = False
label_size = 5
ans_path = 'ans.test'

arg=1
while arg<len(sys.argv):
    if sys.argv[arg] == '-i':
        data_path = sys.argv[arg+1]
    if sys.argv[arg] == '-o':
        out_file = sys.argv[arg+1]
    if sys.argv[arg] == '-n':
        label_size = int(sys.argv[arg+1])
        partial_data = True
    arg += 2

print label_size

def extractDir(folder):
    list = os.listdir(folder)
    try:
        list.remove('.DS_Store')
    except ValueError:
        pass
    return list

def isascii(word):
    try:
        word.decode('ascii')
        return True
    except UnicodeDecodeError:
        return False

def extractWord(path):
    with open(path,'r') as file:
        word_list = file.read().lower()
    word_list = word_list.translate(string.maketrans('',''), string.punctuation).split()
    #st = LancasterStemmer()
    word_list = [stem(word) for word in word_list if isascii(word)]
    #word_list = [st.stem(word) for word in word_list if isascii(word)]
    return word_list

def createVocab(word_list,vocab): # create vocab from word list
    for word in word_list:
        try:
            tempValue = vocab[word]
            vocab[word] = tempValue+1
        except KeyError:
            vocab[word]=1

def joinVocab(word_list,doc_vocab): # join doc_vocab to word_list
    for word,value in doc_vocab.iteritems():
        if word in word_list:
            word_list[word] += value
        else:
            word_list[word] = value

def removeStop(sub_vocab, vocab):
    return {key:value for key, value in sub_vocab.iteritems() if key in vocab}

def loadAns(path):
    with open(path,'r') as file:
        ans_list = file.read().split('\n')
    ans = {}
    for string in ans_list:
        if len(string) > 2:
            doc = string.split(' ')
            if len(doc) != 2:
                print 'ans line error' + string
            else:
                ans[doc[0]] = doc[1]
    return ans

def NBScore(cat_term, vocab_len):
    cat_term_p = {}
    for cat,cat_vocab in cat_term.iteritems():
        term_score = {}
        cat_count = 0
        for va in cat_vocab.values():
            cat_count += va
        cat_count += vocab_len
        for word,value in cat_vocab.iteritems():
            print value
            term_score[word] = float(value*10000+1000000)/float(cat_count)
        cat_term_p[cat] = copy.deepcopy(term_score)
    return cat_term_p

def computeNaiveBayesClass(doc_vocab, cat_prior, cat_term_p):
    cat_score = {}
    for cat,term_p in cat_term_p.iteritems():
        score = 0
        for word,value in term_p.iteritems():
            if word in doc_vocab:
                score += np.log10(value) * doc_vocab[word]
        score += np.log10(cat_prior[cat])
        cat_score[cat] = score
    return max(cat_score, key=lambda i: cat_score[i])

def outAns(result, path):
    output = []
    index = []
    for num, cat in result.iteritems():
        output.append(num+' '+cat+'\n')
        index.append(int(num))
    output = [line for (i, line) in sorted(zip(index, output))] # sort output
    with open(path,'w') as file:
        file.writelines(output)

# find all class
categories = extractDir(data_path+'/Train')

############### save pre build vocab ############
# build vocab term freq
vocab = {}
for root, subFolders, files in os.walk(data_path+'/Train'):
    for file in files:
        if file.isdigit():
             createVocab(extractWord(root+'/'+file),vocab)

# remove stop word
vocab = {word:freq for word, freq in vocab.iteritems() if freq < 1100}
print 'building vocab complete'

# build category term freq {'cat':{'doc':{'term':freq}}} and cat prior {'cat':prob}
cat_term = {}  # category term frequency
cat_doc = {}  # category document number
for cat in categories:
    subfiles = extractDir(data_path+'/Train/'+cat)
    cat_doc[cat] = len(subfiles)
    cat_vocab = {}
    label_count = 0 # for label count control
    for subfile in subfiles:
        if label_count < label_size or not partial_data: # label count control
            createVocab(extractWord(data_path+'/Train/'+cat+'/'+subfile), cat_vocab)
        label_count += 1
    cat_term[cat] = removeStop(cat_vocab,vocab)
print 'building category vocab complete'
total_doc = 0
for cat in cat_doc:
    total_doc += cat_doc[cat]
cat_prior = {cat:float(value)/float(total_doc) for cat,value in cat_doc.iteritems()}
print 'building prior complete'

pickle.dump(vocab, open("vocab.pkl", "wb"))
pickle.dump(cat_term, open("cat_term.pkl", "wb"))
pickle.dump(cat_prior, open("cat_prior.pkl", "wb"))
############### save/load pre build vocab ############
# vocab = pickle.load(open("vocab.pkl","rb"))
# cat_term = pickle.load(open("cat_term.pkl","rb"))
# cat_prior = pickle.load(open("cat_prior.pkl","rb"))
############### load pre build vocab ############

# load ans sheet
ans = loadAns(ans_path)
print 'load ans complete'

# cat word probability in multinomial
# doc use extractWord + createVocab then computeNBC
vocab_len = len(vocab)
print vocab_len
test_list = extractDir(data_path+'/Test')
result ={}
cat_term_p = NBScore(cat_term, vocab_len)
correct_num=0
total_num=0
for test in test_list:
    doc_vocab = {}
    createVocab(extractWord(data_path+'/Test/'+test),doc_vocab)
    doc_vocab = removeStop(doc_vocab,vocab)
    result[test] = computeNaiveBayesClass(doc_vocab,cat_prior,cat_term_p)
    if result[test]==ans[test]:
        correct_num+=1
    total_num+=1
    print float(correct_num)/float(total_num)
print 'Naive Bayes Classification Complete'

# output result
outAns(result, out_file)
print 'save output file complete'