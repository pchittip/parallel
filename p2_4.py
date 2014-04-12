import numpy as np
import scipy.sparse as sp
import pylab as pl

from os import listdir
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import jaccard_similarity_score
from sklearn.metrics.pairwise import linear_kernel

docnames = listdir('big_doc_set')

vectorizer = TfidfVectorizer(ngram_range=(3,3),encoding = 'latin1')
X_train = vectorizer.fit_transform((open('/clam/u3/students/pc563/hw2/big_doc_set/' +f).read() 
					for f in listdir('big_doc_set')))

print X_train.shape

X = X_train.toarray()
X=X*1000

#  use high values for zeros
X[X==0]=100000
X=X.astype(int)

# create sig matrix
# Signature matrix has columns as documents and every row is a hash function
maxdoc = X.shape[0] 
sig = np.ones((24,maxdoc))
sig = sig.astype(int)
s = np.arange(X.shape[1])
i=-1

while (i<23):
    a,b = np.random.random_integers(1,32452867,2)

    h = lambda x:((((a*x)+b)%32454867)%81111)

    hv = np.vectorize(h)

    #check for zero mod value
    t = hv(s)
    #print t
    if (np.count_nonzero(t) != X.shape[1]):
	continue
    else:
	i+=1
#	print "mod value is zero"

    # apply hash function to X
    sigt = ((X*t).min(axis=1)).astype(int)

    sig[i] = sigt
#    print sig[i]

    # the above gives a  hash value for every document - one row

# check if documents are similar in any of the four bands
i=0
coli = 0

while coli < (maxdoc - 1):
    k= 1
    while ((coli+k) < maxdoc):
	if ((np.array_equal(sig[0:6,coli],sig[0:6,coli+k])) or (np.array_equal(sig[6:12,coli] ,sig[6:12,coli+k])) or (np.array_equal(sig[12:18,coli] ,sig[12:18,coli+k])) or (np.array_equal(sig[18:24,coli] ,sig[18:24,coli+k]))):
		print 'found matching documents', coli, coli+k, docnames[coli],docnames[coli+k]

	k += 1
    coli += 1












