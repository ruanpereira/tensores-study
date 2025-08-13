import numpy as np
import matplotlib.pyplot as plt
import scipy.io 

mat_data = scipy.io.loadmat('homeworkmultilin05/Practice_5_unfolding_folding.mat')
ten_X = mat_data['tenX']

mat_data_mode = scipy.io.loadmat('homeworkmultilin05/Practice_5_product_mode_n.mat')
ten_X_mode = mat_data_mode['tenX']
Z = mat_data_mode['Z']

def unfold(tensor, n):
    n = n - 1
    tensor = np.moveaxis(tensor, n, 0)
    tensor_unfolding = tensor.reshape(tensor.shape[0], -1)

    return tensor_unfolding

def fold(tensor_unfolding, tensor_shape, n):
    n = n-1
    # Transforming the shape of tensor tuple into a list for easy manipulation.
    shape = list(tensor_shape)
    # Extracting the external dimension that is presented in the unfolding tensor as the number of rows.
    n_dimension = shape.pop(n)
    # Inserting the previously dimension at the begining of the shape vector so this way we have a dinamic reshape
    # that will change in accord with the unfolding mode.
    shape.insert(0, n_dimension)

    # Reorganizing the unfolded tensor as a tensor.
    tensor = tensor_unfolding.reshape(shape)

    # Moving back the axis that was changed at the unfolding function.
    tensor = np.moveaxis(tensor, 0, n)

    return tensor


def ten_mat_prod(tensor, matrix, n):
    
    shape = list(tensor.shape)
    shape[n-1] = matrix.shape[0]

    tensor = matrix@ unfold(tensor,n)
    tensor = fold(tensor, shape, n)

    return tensor

def ten_mat_multiprod(tensor,list_of_matrices):
    shape = list(tensor.shape)
    
    for i in range(0,list_of_matrices.shape[0]):
        
        tensor = list_of_matrices[i,]@(unfold(tensor,i))
        Z = list_of_matrices[i]
        shape.pop(i)
        shape.insert(i,Z.shape[0])
        tensor = fold(tensor,shape,i)
        shape = list(tensor.shape)
        
    return tensor

#para o problema 1
tenX_1 = unfold(ten_X, 1)
tenX_2 = unfold(ten_X, 2)
tenX_3 = unfold(ten_X, 3)

# para o problema 2
unf_tenX_1 = fold(tenX_1, ten_X.shape, 1)
unf_tenX_2 = fold(tenX_2, ten_X.shape, 2)
unf_tenX_3 = fold(tenX_3, ten_X.shape, 3)

# para o problema 3

ten_Y = ten_mat_prod(ten_X_mode, Z, 1)