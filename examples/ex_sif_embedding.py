import sys
#sys.path.append('../src')
from sif_embedding import data_io 
from sif_embedding import params
from sif_embedding import sif_weight

# input
wordfile = '../../../nlp_projects/semantic_search/models/embedding_v6.txt' # word vector file, can be downloaded from GloVe website
weightfile = '../../../nlp_projects/semantic_search/data/wordfreq_input_v3_preproc.txt' # each line is a word and its frequency
weightpara = 1e-3 # the parameter in the SIF weighting scheme, usually in the range [3e-5, 3e-3]
rmpc = 1 # number of principal components to remove in SIF weighting scheme
sentences = ['estudios cardíacos', 'afecciones coronariascd ']

# load word vectors
(words, We) = data_io.getWordmap(wordfile)
# load word weights
word2weight = data_io.getWordWeight(weightfile, weightpara) # word2weight['str'] is the weight for the word 'str'
weight4ind = data_io.getWeight(words, word2weight) # weight4ind[i] is the weight for the i-th word
# load sentences
x, m = data_io.sentences2idx(sentences, words) # x is the array of word indices, m is the binary mask indicating whether there is a word in that location
w = data_io.seq2weight(x, m, weight4ind) # get word weights

# set parameters
prms = params.params()
prms.rmpc = rmpc
# get SIF embedding
embedding = sif_weight.SIF_embedding(We, x, w, prms) # embedding[i,:] is the embedding for sentence i
print(embedding)
