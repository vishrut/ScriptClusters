from nltk.corpus import wordnet as wn
from nltk.corpus.reader.plaintext import PlaintextCorpusReader
from nltk.tokenize import word_tokenize, wordpunct_tokenize, sent_tokenize
from nltk.tokenize.punkt import PunktWordTokenizer
from nltk.tokenize import *
from nltk.stem.lancaster import LancasterStemmer
from nltk import cluster
from nltk.cluster import euclidean_distance
from numpy import array
import stemming.porter2 as stemport
from pylab import *
import pylab
import numpy, scipy
import scipy.cluster.hierarchy as hier
import scipy.spatial.distance as dist
from nltk.cluster import GAAClusterer
import textmining
import glob
import re

def processscript(filename):
    print '\n' + filename
    f = open(filename,'r')

    s = f.read()

    s  = s.replace('\\n',' ')
    s  = s.replace('\\t',' ')
    s  = re.sub(r'[^a-zA-Z]', r'\t', s)

    #print 'Tokenizing...'
    x = wordpunct_tokenize(s)
    tokenized = len(x)

    #print 'Removing words of length 1-2...'
    list=[]
    for word in x:
        if len(word)>2:
            list.append(word)  
    remove12 = len(list)

    fin={}

    #print 'Stemming...'
    for wd in list:
        tem=stemport.stem(wd)
        if tem in fin:
            fin[tem]=fin[tem]+1
        else :
            fin[tem]=1
    stemmed = len(fin)

    #print 'Removing stop words...'
    f = open('stop.txt', 'r')
    for line in f:
        for word in line.split():
            if(word in fin):
                #print word    
                del fin[word]
    #print fin
    stopped = len(fin)

    
    print 'tokenized:'
    print tokenized

    print 'remove12:'
    print remove12

    print 'stemmed:'
    print stemmed

    print 'stopped:'
    print stopped
    

    str = ''.join('%s ' % (k) for k,v in fin.iteritems())
    return str


scriptlist = []
rowlist = []
doc_count=[]

scriptlist = glob.glob("./moviescripts/*.txt")
tdm = textmining.TermDocumentMatrix()
processed = []
i=0
for script in scriptlist:
    #if(i<20):
    stringed = processscript(scriptlist[i])
    doc_count.append(len(stringed.split(" ")))
    tdm.add_doc(stringed)
    i = i+1

ctr = 0
for row in tdm.rows(cutoff=1):
    if(ctr>0):
        rowlist.append(row)
    ctr = 1       

print '\nusing scipy...'
data = rowlist

total_word_freq =[]
num_of_doc = []
tf = data
idf = data
dat = data
n = len(data[0])
m = len(data)
print n,m
for i in range(n):
    total_word_freq.append(0)
    num_of_doc.append(0)

for i in range(n):
    for j in range(m):
        total_word_freq[i] += data[j][i] 
        if(data[j][i]!=0):
            num_of_doc[i]+=1;
        
for i in range(n):
    for j in range(m):
        data[j][i] = data[j][i]/float(doc_count[j])
    #data[j][i] = tf[j][i]

for i in range(n):
    for j in range(m):
        data[j][i] *= math.log(len(data) / float(num_of_doc[i]))

#calculate a distance matrix
distMatrix = dist.pdist(data)


def llf(id):
    return str(id)

#convert the distance matrix to square form. The distance matrix 
#calculated above contains no redundancies, you want a square form 
#matrix where the data is mirrored across the diagonal.
distSquareMatrix = dist.squareform(distMatrix)
print '\ndistance matrix:'
print distSquareMatrix

#calculate the linkage matrix
fig = pylab.figure(figsize=(100,100))
linkageMatrix = hier.linkage(distSquareMatrix, method = 'ward')
dendro = hier.dendrogram(linkageMatrix,orientation='left', labels=scriptlist)

#fig.show()
fig.savefig('dendrogram.png')
print '\nlinkage matrix:'
print linkageMatrix
print '\ndendrogram:'
print dendro


answer = []

vectors = [array(f) for f in data]
clusterer = cluster.KMeansClusterer(8,   euclidean_distance, repeats=10, avoid_empty_clusters=True)
print  '\nK-means results using NLTK:'
answer = clusterer.cluster(vectors, True)

i=0
j=0
for j in range(8):
    print '\n cluster:'
    print j
    for i in range(len(answer)):
        if(answer[i]==j):
            print scriptlist[i]


# classify a new vector
'''
vector = numpy.array([3, 3])
print('classify(%s):' % vector, end='')
print(clusterer.classify(vector))
print()
'''



                


            
          
            
    




