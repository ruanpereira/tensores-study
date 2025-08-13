import numpy as np
import matplotlib.pyplot as plt
import scipy.io

mat_data = scipy.io.loadmat('homeworkmultilin06/Practice_6_hosvd.mat')
ten_X = mat_data['tenX']

def unfold(tensor, n):
    n = n - 1
    tensor = np.moveaxis(tensor, n, 0)
    tensor_unfolding = tensor.reshape(tensor.shape[0], -1)

    return tensor_unfolding

def fold(tensor_unfolding, tensor_shape, n):
    n = n - 1
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

def HOSVD(tensor):
    
    modes = len(tensor.shape)
    U = []
    
    for i in range(modes):
        tensor_unfolded = unfold(tensor, i+1)
        [u, _, _] = np.linalg.svd(tensor_unfolded)
        U.append(u)
        s_tensor = ten_mat_prod(tensor, u.T, i+1) 
    return s_tensor, U

# para o problema 1:
ten_S_est, U_est = HOSVD(ten_X)

print("Shape of ten_S_est:", ten_S_est.shape)
print("Shapes of tensor : ", ten_X.shape)

print("Shapes of U matrices:")
for i, u in enumerate(U_est):
    print(f"U[{i+1}]: {u.shape}")

