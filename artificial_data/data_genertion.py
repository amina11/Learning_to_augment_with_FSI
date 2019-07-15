
### 2018-5-17
### this file is made to generate artificial data
from __future__ import division
import numpy as np
from sklearn.cluster import KMeans
from itertools import combinations
from numpy import linalg as LA
dim_train = 5000
dim_feature = 3000
dim_output = 5
dim_latentfeature = np.int(0.2 * dim_feature)


## to match the real world data set we have, we sparcify the matrix x, 50% element are zero
np.random.seed(502)
x_rand =np.random.random([dim_train, dim_feature]) 
###sparsify the matrix x
sparsity_mat = np.random.randint(2, size = (dim_train, dim_feature))
x = np.multiply(x_rand, sparsity_mat)


### generate class membership for featurs to cluster them into latent variables
class_id1 = np.arange(dim_latentfeature)
class_id2 = np.random.randint(dim_latentfeature, size = (1, dim_feature - dim_latentfeature)).reshape(dim_feature - dim_latentfeature)
class_id = np.concatenate((class_id1, class_id2), axis=0)
to_cluster = np.zeros([dim_feature, dim_latentfeature])
S1 = np.zeros([dim_feature, dim_feature])
for i in range(dim_latentfeature):
    d = class_id == i
    to_cluster[d, i] =1
    index = [k for k, xx in enumerate(d) if xx]  ### get the index of the d, index of the features belong to the same class
    if len(index)>1:
        combo = list(combinations(index, 2))
        mm=len(combo)
        for j in range(mm):
            S1[combo[j][0], combo[j][1]] =1
            S1[combo[j][1], combo[j][0]] = 1

diag_s = np.eye(dim_feature)            
S = S1 + diag_s   
m = np.matmul(x, to_cluster)


def sigmoid(x):
    return (1 / (1 + np.exp(-x)))


L = np.random.random(([dim_latentfeature, dim_output])) - 1/2
linear_transform_m = np.matmul(m, L) 
nonlinear_transform_m = sigmoid(linear_transform_m)
yy = (nonlinear_transform_m).argmax(axis=1)  # label of the classes
y = np.zeros([dim_train, dim_output])
row_index = np.arange(dim_train)
y[row_index, yy] = 1

y.sum(axis = 0)

a = S[S>0]
a.shape[0] / (dim_feature * dim_feature)  ## sparsity level of S 0.19% total nonzero elements of S 17420


### add noise to the S
noisy_S = np.copy(S)
noise_level = 36000
d1 = np.random.randint(dim_feature, size = (noise_level))
d2 = np.random.randint(dim_feature, size = (noise_level))
noisy_S[d1, d2] = 1
a = noisy_S[noisy_S>0]
a.shape[0] / (dim_feature * dim_feature)  ## sparsity level of S now  0.29%

np.save('./SD2/x', x)
np.save('./SD2/y', y)
np.save('./SD2/S', S)
np.save('./SD2/noisy_S', noisy_S)
np.save('./SD2/latent_representation', m)
