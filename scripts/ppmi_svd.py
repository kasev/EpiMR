import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
from gensim.corpora import Dictionary
from gensim.matutils import corpus2csc
from sklearn.manifold import TSNE
from gensim.models.keyedvectors import KeyedVectors




def normalize_ppmi3_matrix(pmi_matrix_df):
    minval, maxval = pmi_matrix_df.min().min(), pmi_matrix_df.max().max()
    diff = abs(maxval-minval)
    minval_doubled = minval - diff
    pmi_matrix_df.fillna(minval_doubled, inplace=True)
    pmi_matrix_norm_df = (pmi_matrix_df - minval_doubled) / (maxval - minval_doubled)
    return pmi_matrix_norm_df

def get_ppmi_df(cooc, vocabulary, normalize=True, exp=2):
    pmi_rows_list = []
    for i in range(cooc.shape[1]):
        ab = np.array([row_el for row_el in list(cooc[i].toarray()[0])], dtype=float)
        ab_exp = np.power(ab, exp)
        axb = np.array([cooc[row_el[0], row_el[0]] * cooc[i, i] for row_el in enumerate(list(cooc[i].toarray()[0]))], dtype=float)
        pmi_row = np.divide(ab_exp, axb, out=np.zeros_like(ab_exp), where=axb!=0)
        pmi_row = [np.log(n) if n>0 else None for n in pmi_row]
        pmi_rows_list.append(pmi_row)
    pmi_matrix_df = pd.DataFrame(pmi_rows_list, columns=vocabulary, index=vocabulary)
    if normalize == True:
        pmi_matrix_df = normalize_ppmi3_matrix(pmi_matrix_df)
        np.fill_diagonal(pmi_matrix_df.to_numpy(), 1)
    return pmi_matrix_df #pmi_matrix_norm_df

def svd_reduction(cooc_matrix, n_components=150, random_state=1, n_iter=10):
    svd = TruncatedSVD(n_components=n_components, random_state=random_state, n_iter=n_iter)
    svd_matrix = svd.fit_transform(cooc_matrix)
    return svd_matrix

def from_bows_to_embeddings(corpus_bows, dct, svd_dims=150, vocabulary=None):
    term_doc_mat = corpus2csc(corpus_bows, num_terms=len(dct))
    if vocabulary==None:
        vocabulary = list(dct.values())
    else:
        vocab_ids = [dct.token2id[t] for t in vocabulary]
        term_doc_mat = term_doc_mat[vocab_ids,:]
    cooc = np.dot(term_doc_mat, term_doc_mat.T)
    pmi_matrix = get_ppmi_df(cooc, vocabulary)
    word_vectors_array = svd_reduction(pmi_matrix, n_components=svd_dims, random_state=1, n_iter=10)
    keyed_vectors = KeyedVectors(svd_dims)
    keyed_vectors.add_vectors(vocabulary, word_vectors_array)
    pmi_svd_cos = pd.DataFrame(cosine_similarity(word_vectors_array), columns=vocabulary, index=vocabulary)
    return [cooc, vocabulary, pmi_matrix, keyed_vectors, pmi_svd_cos]

def get_tsne_coors(keyed_vectors, perplexity=18):
    # inverse similarity to distance
    #data = (1 - sim_matrix) / 1
    words = keyed_vectors.index_to_key
    #data.round(5)
    # tSNE to project all words into a 2-dimensional space
    tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity, metric='cosine', n_iter=1000) # dissimilarity="precomputed",
    #tsne = TSNE(n_components=2, random_state=42, perplexity=18, metric='precomputed', n_iter=5000) # dissimilarity="precomputed",
    pos = tsne.fit_transform(keyed_vectors.vectors) # project all points into space
    xs, ys = pos[:, 0], pos[:, 1]
    # extract minimal and maximal values
    minmax = [pos[:, 0].min(), pos[:, 0].max(), pos[:, 1].min(), pos[:, 1].max()]
    # normalize on scale from 0 to 1
    xs = (xs - minmax[0]) / (minmax[1] - minmax[0])
    ys = (ys - minmax[2]) / (minmax[3] - minmax[2])
    return xs, ys, words