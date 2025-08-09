 
# #A Positional Encoding Example
# Copyright 2023 Denis Rothman, MIT License
# 
# Reference 1 for Positional Encoding:
# Attention is All You Need paper, page 6,Google Brain and Google Research
# 
# Reference 2 for word embedding:
# https://www.geeksforgeeks.org/python-word-embedding-using-word2vec/
# Reference 3 for cosine similarity:
# SciKit Learn cosine similarity documentation
# 
# The goal of this notebook is to understand positional encoding and cosine similarity. Cosine similarity remains a solid NLP approach.
# 
# The text.txt file is just to illustrate the concepts in the notebook.

 
# Downloading text file

# #text.file
# !curl -L https://raw.githubusercontent.com/Denis2054/Transformers-for-NLP-and-Computer-Vision-3rd-Edition/master/Chapter02/text.txt --output "text.txt"

#!pip install gensim # Version Gensim 4.0.0 and above
import torch
import nltk
nltk.download('punkt')


import math
import numpy as np
from nltk.tokenize import sent_tokenize, word_tokenize
import gensim
from gensim.models import Word2Vec
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings(action = 'ignore')


dprint=0 # prints outputs if set to 1, default=0

# read ‘text.txt’ file
sample = open("news.txt", "r")
s = sample.read()

# processing escape characters
f = s.replace("\n", " ")

data = []

# sentence parsing
for i in sent_tokenize(f):
	temp = []
	# tokenize the sentence into words
	for j in word_tokenize(i):
		temp.append(j.lower())
	data.append(temp)

# Creating Skip Gram model
model2 = gensim.models.Word2Vec(data, min_count = 1, vector_size = 512, window = 5, sg = 1)

# 1-The 2-black 3-cat 4-sat 5-on 6-the 7-couch 8-and 9-the 10-brown 11-dog 12-slept 13-on 14-the 15-rug.
def readFromInput():
    word1=input("Enter first word: ")
    word2=input("Enter second word: ")
    pos1=int(input("Enter position of first word: "))
    pos2=int(input("Enter position of second word: "))
    return word1,word2,pos1,pos2

word1,word2,pos1,pos2=readFromInput()
print(word1,word2,pos1,pos2)
a=model2.wv[word1]
b=model2.wv[word2]

print(f'word1: {word1}, word2: {word2}')
print(f'word1 vector: {a}')
print(f'word2 vector: {b}')

if(dprint==1):
        print(a)

# compute cosine similarity
dot = np.dot(a, b)
norma = np.linalg.norm(a)
normb = np.linalg.norm(b)
cos = dot / (norma * normb)

aa = a.reshape(1,512)
ba = b.reshape(1,512)
cos_lib = cosine_similarity(aa, ba)

 
# A Positional Encoding example using one line of basic Python using a few lines of code for the sine and cosine functions.
# I added a Pytorch method inspired by Pytorch.org to explore these methods.
# The main idea to keep in mind is that we are looking to add small values to the word embedding output so that the positions are taken into account. This means that as long as the cosine similarity, for example, displayed at the end of the notebook, shows the positions have been taken into account, the method can apply. Depending on the Transformer model, this method can be fine-tuned as well as using other methods.


pe1=aa.copy()
pe2=aa.copy()
pe3=aa.copy()
paa=aa.copy()
pba=ba.copy()
d_model=512
max_print=d_model
max_length=20

for i in range(0, max_print,2):
                # compute pe1 for word1 at position pos1 (sin and cos) 
                pe1[0][i] = math.sin(pos1 / (10000 ** ((2 * i)/d_model))) # for even dimensions
                paa[0][i] = (paa[0][i]*math.sqrt(d_model))+ pe1[0][i] # add the positional encoding to the word embedding, later used for positional encoding similarity
                pe1[0][i+1] = math.cos(pos1 / (10000 ** ((2 * i)/d_model))) # for odd dimensions
                paa[0][i+1] = (paa[0][i+1]*math.sqrt(d_model))+pe1[0][i+1] # add the positional encoding to the word embedding, later used for positional encoding similarity
                if dprint==1:
                        print(i,pe1[0][i],i+1,pe1[0][i+1])
                        print(i,paa[0][i],i+1,paa[0][i+1])
                        print("\n")

#print(pe1)
# A  method in Pytorch using torch.exp and math.log :
max_len=max_length
pe = torch.zeros(max_len, d_model)
print(f'pe.shape {pe.shape}')
position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
pe[:, 0::2] = torch.sin(position * div_term)
pe[:, 1::2] = torch.cos(position * div_term)
print(pe[:, 0::2])



for i in range(0, max_print,2):
                pe2[0][i] = math.sin(pos2 / (10000 ** ((2 * i)/d_model)))
                pba[0][i] = (pba[0][i]*math.sqrt(d_model))+ pe2[0][i]

                pe2[0][i+1] = math.cos(pos2 / (10000 ** ((2 * i)/d_model)))
                pba[0][i+1] = (pba[0][i+1]*math.sqrt(d_model))+ pe2[0][i+1]

                if dprint==1:
                        print(i,pe2[0][i],i+1,pe2[0][i+1])
                        print(i,paa[0][i],i+1,paa[0][i+1])
                        print("\n")

print(word1,word2)
cos_lib = cosine_similarity(aa, ba)
print(cos_lib,"word similarity")
cos_lib = cosine_similarity(pe1, pe2)
print(cos_lib,"positional similarity")
cos_lib = cosine_similarity(paa, pba)
print(cos_lib,"positional encoding similarity")

# As long as the cosine similarity is different from the word similarity, the positional encoding has been applied, and the positional similarity should be similar to the positional encoding similarity.

if dprint==1:
        print(word1)
        print("embedding")
        print(aa)
        print("positional encoding")
        print(pe1)
        print("encoded embedding")
        print(paa)

        print(word2)
        print("embedding")
        print(ba)
        print("positional encoding")
        print(pe2)
        print("encoded embedding")
        print(pba)




