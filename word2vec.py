import math
import random
import re

from datetime import datetime

####################################
# Helper Utils
####################################

# reference: http://c-faq.com/lib/gaussian.html

def gaussrand():

    x = 0
    sum_count = 25
    for i in range(sum_count):
        x += random.random()
    x -= (sum_count/2.0)
    x /= math.sqrt(sum_count/ 12.0)

    return x

def dot_product(mat1, mat2):

    res_mat = [[0 for j in range(len(mat2[0]))] for i in range(len(mat1))]
    for i in range(len(mat1)):
        for j in range(len(mat2[0])):
            for k in range(len(mat2)):
                res_mat[i][j] += mat1[i][k] * mat2[k][j]
    return res_mat

def np_sum_mat_axis_zero(mat):

    res_mat =[[0 for i in range(len(mat[0]))]]
    for i in range(len(mat)):
        for j in range(len(mat[0])):
            res_mat[0][j] += mat[i][j]

    return res_mat

def np_sum_mat_axis_none(mat):

    res = 0
    for i in range(len(mat)):
        for j in range(len(mat[0])):
            res += mat[i][j]

    return res

def add_to_all_mat_elem(mat, value):

    res_mat = list()
    for i in range(len(mat)):
        res_mat.append(list())
        for j in range(len(mat[i])):
            res_mat[i].append(mat[i][j] + value)

    return res_mat

def np_divide_mat_by_vector(mat, vec):

    res_mat = [[0 for j in range(len(mat[0]))] for i in range(len(mat))]
    for i in range(len(mat)):
        for j in range(len(mat[i])):
            res_mat[i][j] = mat[i][j]*1.0/vec[j]

    return res_mat

####################################
# Data Prep
####################################

def tokenize(text):

    # obtains tokens with a least 1 alphabet
    pattern = re.compile(r'[A-Za-z]+[\w^\']*|[\w^\']*[A-Za-z]+[\w^\']*')

    return pattern.findall(text.lower())

def mapping(tokens):

    word_to_id = dict()
    id_to_word = dict()

    for i, token in enumerate(set(tokens)):
        word_to_id[token] = i
        id_to_word[i] = token

    return word_to_id, id_to_word

def generate_training_data(tokens, word_to_id, window_size):

    N = len(tokens)
    X_Y, Y_Y = [], []

    for i in range(N):
        nbr_inds = list(range(max(0, i - window_size), i)) + \
                   list(range(i + 1, min(N, i + window_size + 1)))
        for j in nbr_inds:
            X_Y.append(word_to_id[tokens[i]])
            Y_Y.append(word_to_id[tokens[j]])

    X = [X_Y]
    Y = [Y_Y]

    return X, Y

####################################
# Initialization
####################################

def initialize_wrd_emb(vocab_size, emb_size):

    # WRD_EMB = np.random.randn(vocab_size, emb_size) * 0.01
    WRD_EMB = [[(gaussrand() * 0.01) for j in range(emb_size)] for i in range(vocab_size)]

    assert(len(WRD_EMB) == vocab_size)
    assert(len(WRD_EMB[0]) == emb_size)

    return WRD_EMB

def initialize_dense(input_size, output_size):

    # W = np.random.randn(output_size, input_size) * 0.01
    W = [[(gaussrand() * 0.01) for j in range(input_size)] for i in range(output_size)]

    assert(len(W) == output_size)
    assert(len(W[0]) == input_size)

    return W

def initialize_parameters(vocab_size, emb_size):

    WRD_EMB = initialize_wrd_emb(vocab_size, emb_size)
    W = initialize_dense(emb_size, vocab_size)
    
    parameters = {}
    parameters['WRD_EMB'] = WRD_EMB
    parameters['W'] = W
    
    return parameters

def get_batch(data_l, start, batch_size):

    batch = list()
    for i in range(len(data_l)):
        batch.append(data_l[i][i:i+batch_size])

    return batch

####################################
# Forward Propagation
####################################

def ind_to_word_vecs(inds, parameters):

    m = len(inds[0])
    WRD_EMB = parameters['WRD_EMB']

    # word_vec = WRD_EMB[inds.flatten(), :].T
    word_vec = [WRD_EMB[inds[0][i]] for i in range(m)]

    word_vec_T = [[] for i in range(len(word_vec[0]))]
    for i in range(len(word_vec)):
        for j in range(len(word_vec[i])):
            word_vec_T[j].append(word_vec[i][j])

    assert(len(word_vec_T) == len(WRD_EMB[0]))
    assert(len(word_vec_T[0]) == m)
    
    return word_vec_T

def linear_dense(word_vec, parameters):

    m = len(word_vec[0])
    W = parameters['W']
    Z = dot_product(W, word_vec)

    assert(len(Z) == len(W))
    assert(len(Z[0]) ==  m)
    
    return W, Z

def softmax(Z):

    # np.exp(Z)
    Z_exp = [[math.exp(Z[i][j]) for j in range(len(Z[i]))] for i in range(len(Z))]


    # np.sum(np.exp(Z), axis=0, keepdims=True)
    Z_sum = np_sum_mat_axis_zero(Z_exp)

    # np.sum(np.exp(Z), axis=0, keepdims=True) + 0.001
    divisor = add_to_all_mat_elem(Z_sum, 0.001)

    # softmax_out = np.divide(np.exp(Z), np.sum(np.exp(Z), axis=0, keepdims=True) + 0.001)
    softmax_out = np_divide_mat_by_vector(Z_exp, divisor[0])
    
    assert(len(softmax_out) == len(Z))
    assert(len(softmax_out[0]) == len(Z[0]))

    return softmax_out

def forward_propagation(inds, parameters):
    word_vec = ind_to_word_vecs(inds, parameters)
    W, Z = linear_dense(word_vec, parameters)
    softmax_out = softmax(Z)
    
    caches = {}
    caches['inds'] = inds
    caches['word_vec'] = word_vec
    caches['W'] = W
    caches['Z'] = Z

    return softmax_out, caches

####################################
# Cost Function
####################################

def cross_entropy(softmax_out, Y):

    m = len(softmax_out[0])
    Y_flatten = Y[0] 

    # softmax_out[Y.flatten(), np.arange(Y.shape[1])]
    softmax_out_rearranged = [softmax_out[Y_flatten[i]][i] for i in range(len(Y_flatten))]

    # softmax_out[Y.flatten(), np.arange(Y.shape[1])] + 0.001
    softmax_out_add = [(i + 0.001) for i in softmax_out_rearranged]

    # np.log(softmax_out[Y.flatten(), np.arange(Y.shape[1])] + 0.001))
    softmax_out_log = [math.log(i) for i in softmax_out_add]

    # np.sum(np.log(softmax_out[Y.flatten(), np.arange(Y.shape[1])] + 0.001))
    softmax_out_sum = sum(softmax_out_log)

    cost = -(1 / m) * softmax_out_sum

    return cost

####################################
# Backward Propagation
####################################

def softmax_backward(Y, softmax_out):

    m = len(Y[0])

    # dL_dZ = softmax_out[Y.flatten(), np.arange(m)] -= 1.0
    Y_flatten = Y[0] 
    dL_dZ = softmax_out
    for i in range(len(Y_flatten)):
        dL_dZ[Y_flatten[i]][i] -= 1.0

    assert(len(dL_dZ) == len(softmax_out))
    assert(len(dL_dZ[0]) == len(softmax_out[0]))

    return dL_dZ

def dense_backward(dL_dZ, caches):

    W = caches['W']
    W_T = [[] for i in range(len(W[0]))]
    for i in range(len(W)):
        for j in range(len(W[i])):
            W_T[j].append(W[i][j])

    # word_vec.T
    word_vec = caches['word_vec']
    m = len(word_vec[0])
    word_vec_T = [[] for i in range(len(word_vec[0]))]
    for i in range(len(word_vec)):
        for j in range(len(word_vec[i])):
            word_vec_T[j].append(word_vec[i][j])

    # np.dot(dL_dZ, word_vec.T)
    dL_dZ_dot_word_vec_T = dot_product(dL_dZ, word_vec_T)

    # dL_dW = (1 / m) * np.dot(dL_dZ, word_vec.T)
    dL_dW = dL_dZ_dot_word_vec_T
    for i in range(len(dL_dW)):
        for j in range(len(dL_dW[i])):
            dL_dW[i][j] = dL_dW[i][j]*1.0/m

    # dL_dword_vec = np.dot(W.T, dL_dZ)
    dL_dword_vec = dot_product(W_T, dL_dZ)

    assert(len(W) == len(dL_dW))
    assert(len(W[0]) == len(dL_dW[0]))

    assert(len(word_vec) == len(dL_dword_vec))
    assert(len(word_vec[0]) == len(dL_dword_vec[0]))

    return dL_dW, dL_dword_vec

def backward_propagation(Y, softmax_out, caches):

    dL_dZ = softmax_backward(Y, softmax_out)
    dL_dW, dL_dword_vec = dense_backward(dL_dZ, caches)
    
    gradients = dict()
    gradients['dL_dZ'] = dL_dZ
    gradients['dL_dW'] = dL_dW
    gradients['dL_dword_vec'] = dL_dword_vec
    
    return gradients

def update_parameters(parameters, caches, gradients, learning_rate):
    vocab_size, emb_size = len(parameters['WRD_EMB']), len(parameters['WRD_EMB'][0])
    inds = caches['inds']
    dL_dword_vec = gradients['dL_dword_vec']
    m = len(inds)
    inner_inds = inds[0]
    while (type(inner_inds) == list):
        m = len(inner_inds)
        inner_inds = inner_inds[0]

    # dL_dword_vec.T
    dL_dword_vec_T = [[] for i in range(len(dL_dword_vec[0]))]
    for i in range(len(dL_dword_vec)):
        for j in range(len(dL_dword_vec[i])):
            dL_dword_vec_T[j].append(dL_dword_vec[i][j])

    # dL_dword_vec.T * learning_rate
    for i in range(len(dL_dword_vec_T)):
        for j in range(len(dL_dword_vec_T[i])):
            dL_dword_vec_T[i][j] = dL_dword_vec_T[i][j]*learning_rate

    #parameters['WRD_EMB'][inds.flatten(), :] -= dL_dword_vec.T * learning_rate
    inds_flatten = inds[0]
    for i in range(len(inds_flatten)):
        for j in range(len(parameters['WRD_EMB'][inds_flatten[i]])):
            parameters['WRD_EMB'][inds_flatten[i]][j] -= dL_dword_vec_T[i][j]

    #parameters['W'] -= learning_rate * gradients['dL_dW']
    for i in range(len(parameters['W'])):
        for j in range(len(parameters['W'][i])):
            parameters['W'][i][j] -= learning_rate*gradients['dL_dW'][i][j]


def skipgram_model_training(X, Y, vocab_size, emb_size, learning_rate, epochs, batch_size=256, parameters=None, print_cost=False):
    costs = []
    m = len(X[0])
    
    if parameters is None:
        parameters = initialize_parameters(vocab_size, emb_size)

    begin_time = datetime.now()
    for epoch in range(epochs):
        epoch_cost = 0
        batch_inds = list(range(0, m, batch_size))
        random.shuffle(batch_inds)
        for i in batch_inds:
            X_batch = get_batch(X, i, batch_size)
            Y_batch = get_batch(Y, i, batch_size)
            
            softmax_out, caches = forward_propagation(X_batch, parameters)
            cost = cross_entropy(softmax_out, Y_batch)
            gradients = backward_propagation(Y_batch, softmax_out, caches)
            update_parameters(parameters, caches, gradients, learning_rate)
            epoch_cost += cost
            
        costs.append(epoch_cost)
        if print_cost and epoch % (epochs // 500) == 0:
            print("Cost after epoch {}: {}".format(epoch, epoch_cost))
        if epoch % (epochs // 100) == 0:
            learning_rate *= 0.98
    end_time = datetime.now()
    print('training time: {}'.format(end_time - begin_time))

    return parameters

def main():

    doc = "After the deduction of the costs of investing, " \
        "beating the stock market is a loser's game."
    tokens = tokenize(doc)
    word_to_id, id_to_word = mapping(tokens)

    X, Y = generate_training_data(tokens, word_to_id, 3)

    vocab_size = len(id_to_word)

    paras = skipgram_model_training(X, Y, vocab_size, 50, 0.05, 5000, batch_size=128, parameters=None, print_cost=True)
    X_test = [list(range(vocab_size))]
    softmax_test, _ = forward_propagation(X_test, paras)
    top_sorted_inds = [[i for (v, i) in sorted((v, i) for (i, v) in enumerate(softmax_test[k]))] for k in range(len(softmax_test))]
    top_sorted_inds = top_sorted_inds[-4:]

    for input_ind in range(vocab_size):
        input_word = id_to_word[input_ind]
        output_words = list()
        for i in range(len(top_sorted_inds)-1, -1, -1):
            output_ind = top_sorted_inds[i][input_ind]
            output_words.append(id_to_word[output_ind])
        print("{}'s neighbor words: {}".format(input_word, output_words))

if __name__ == "__main__":

    main()
