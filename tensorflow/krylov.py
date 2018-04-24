import numpy as np
from scipy.sparse import diags
import tensorflow as tf
import functools


def identity_mult_fn(v, n):
    return v

# Multiplication by (Z_{f,v} + diag(d))^T
def circ_diag_transpose_mult_fn(v_f, d, x, n):
    #sess = tf.InteractiveSession()
    #tf.initialize_all_variables().run()

    #print sess.run(x)

    # Circular shift x to the left
    y = tf.concat([x[1:], [x[0]]], axis=0)

    # Scale by [v f]
    return tf.multiply(y, v_f) + tf.multiply(d, x)

# Multiplication by Z_{f,v}^T
def circ_transpose_mult_fn(v_f, x, n):
    #sess = tf.InteractiveSession()
    #tf.initialize_all_variables().run()

    #print sess.run(x)

    # Circular shift x to the left
    y = tf.concat([x[1:], [x[0]]], axis=0)

    # Scale by [v f]
    return tf.multiply(y, v_f)

def circ_mult_fn(f_v, x, n):
    # Circular shift x to the right
    y = tf.concat([x[n-1:], x[:n-1]], axis=0)

    # Scale by [f v]
    return tf.multiply(y, f_v)

def symm_tridiag_mult_fn(diag, off_diag, x, n):
    sub_result = tf.multiply(x[1:], off_diag)
    sup_result = tf.multiply(x[:n-1], off_diag)

    sup_result = tf.concat([[0], sup_result], axis=0)
    sub_result = tf.concat([sub_result, [0]], axis=0)

    return sup_result + sub_result + tf.multiply(x, diag)

def tridiag_corners_mult_fn(subdiag, diag, supdiag, f_ur, f_ll, x, n):
    sup_result = tf.multiply(x[1:], supdiag)
    sup_result = tf.concat([sup_result, [0]], axis=0)
    sub_result = tf.multiply(x[:n-1], subdiag)
    sub_result = tf.concat([[0], sub_result], axis=0)
    z1 = tf.zeros(n-1, dtype=tf.float64)
    z2 = tf.zeros(n-1, dtype=tf.float64)
    scaled_f_ll = tf.multiply(f_ll, x[0])     
    f_ll_result = tf.concat([z1, scaled_f_ll], axis=0)     

    scaled_f_ur = tf.multiply(f_ur, x[-1])     
    f_ur_result = tf.concat([scaled_f_ur, z2], axis=0)     

    return sup_result + sub_result + tf.multiply(x,diag) + f_ll_result + f_ur_result


def tridiag_corners_transpose_mult_fn(subdiag, diag, supdiag, f_ur, f_ll, x, n):
    return tridiag_corners_mult_fn(supdiag, diag, subdiag, f_ll, f_ur, x, n)
"""
# f1: top right. f2: bottom left.
def tridiag_corners_transpose_mult_fn(subdiag, diag, supdiag, f1, f2, x, n):
    sub_result = tf.multiply(x[1:], subdiag)
    sup_result = tf.multiply(x[:n-1], supdiag)

    sup_result = tf.concat([[0], sup_result], axis=0)
    sub_result = tf.concat([sub_result, [0]], axis=0)

    z = tf.zeros(n-1, dtype=tf.float64)
    scaled_f1 = tf.multiply(f1, x[0])
    scaled_f2 = tf.multiply(f2, x[n-1])

    f1_result = tf.concat([z, scaled_f1], axis=0)
    f2_result = tf.concat([scaled_f2, z], axis=0)

    return sup_result + sub_result + tf.multiply(x, diag) + f1_result + f2_result
"""

# subdiag, diag, supdiag of Z_f
# multiplies by Z_f^T
# subdiag: last n-1 entries
# supdiag: first n-1 entries
def tridiag_corner_transpose_mult_fn(subdiag,diag,supdiag,f,x,n):
    sub_result = tf.multiply(x[1:], subdiag)
    sup_result = tf.multiply(x[:n-1], supdiag)
    sup_result = tf.concat([[0], sup_result], axis=0)
    sub_result = tf.concat([sub_result, [0]], axis=0)
    #sess = tf.InteractiveSession()
    #tf.initialize_all_variables().run()
    z = tf.zeros(n-1, dtype=tf.float64)
    scaled_f = tf.multiply(f, x[0])
    f_result = tf.concat([z, scaled_f], axis=0)
    return sup_result + sub_result + tf.multiply(x, diag) + f_result

# subdiag, diag, supdiag of the transposed operator
# f: bottom left
"""
def tridiag_corner_transpose_mult_fn(subdiag, diag, supdiag, f, x, n):
    sub_result = tf.multiply(x[1:], subdiag)
    sup_result = tf.multiply(x[:n-1], supdiag)

    sup_result = tf.concat([[0], sup_result], axis=0)
    sub_result = tf.concat([sub_result, [0]], axis=0)

    z = tf.zeros(n-1, dtype=tf.float64)
    scaled_f = tf.multiply(f, x[0])

    f_result = tf.concat([z, scaled_f], axis=0)

    return sup_result + sub_result + tf.multiply(x, diag) + f_result
"""

# Multiplication by Z_{subdiag, diag, supdiag, f}^T
def tridiag_corner_mult_fn(subdiag, diag, supdiag, f, x, n):
    sup_result = tf.multiply(x[1:], supdiag)
    sub_result = tf.multiply(x[:n-1], subdiag)

    sup_result = tf.concat([sup_result, [0]], axis=0)
    sub_result = tf.concat([[0], sub_result], axis=0)

    z = tf.zeros(n-1, dtype=tf.float64)
    scaled_f = [tf.multiply(f, x[n-1])]

    f_result = tf.concat([scaled_f, z], axis=0)

    return sup_result + sub_result + tf.multiply(x, diag) + f_result

# Multiplication by diag(d)
def diag_mult_fn(d, x, n):
    return tf.multiply(d, x)

def krylov(fn, v, n):
    # fn: takes as input a vector and multiplies by a matrix.
    v_exp = tf.expand_dims(v,1)
    cols = [v_exp]
    this_col = v

    for i in range(n-1):
        this_col = fn(this_col, n)

        cols.append(tf.expand_dims(this_col,1))

    K = tf.stack(cols)

    return tf.transpose(tf.squeeze(K))

def test_circ_sparsity():
    n = 4
    subdiag = np.array([2,3,4])
    supdiag = np.zeros(n-1)
    diag = np.zeros(n)
    # Subdiag corresponds to Z_f, we multiply by Z_f^T
    A = diags([subdiag, diag, supdiag], [-1, 0, 1], (n,n)).toarray().T

    f = 5.0
    A[-1,0] = f

    print('A:', A)
    x = np.array([1,2,3,4])

    subdiag = tf.constant(subdiag, dtype=tf.float64)
    f = tf.constant([f], dtype=tf.float64)
    subdiag_f = tf.concat([subdiag, f], axis=0)

    fn = functools.partial(circ_transpose_mult_fn, subdiag_f)

    print('x: ', x)
    print('Ax: ', np.dot(A,x))

    x = tf.constant(x, dtype=tf.float64)
    result = fn(x,n)

    #result = krylov(fn, x, n)

    sess = tf.InteractiveSession()
    tf.initialize_all_variables().run()

    print(sess.run(result))

def test_tridiag_corner():
    n = 4

    # Subdiag, supdiag, diag corresponds to Z_e, we multiply by Z_e^T
    subdiag = np.array([2,3,4])
    supdiag = np.array([4,5,6])
    diag = np.array([1,1,1,1])
    f = 5.0

    A = diags([subdiag, diag, supdiag], [-1, 0, 1], (n,n)).toarray().T

    A[-1, 0] = f
    print('subdiag: ', subdiag)
    print('supdiag: ', supdiag)
    print('diag: ', diag)
    print(A)

    x = np.array([1,2,3,4])

    print('A: ', A)

    print('Ax: ', np.dot(A,x))

    fn = functools.partial(tridiag_corner_transpose_mult_fn, tf.constant(subdiag, dtype=tf.float64),
        tf.constant(diag, dtype=tf.float64), tf.constant(supdiag, dtype=tf.float64), tf.constant(f, dtype=tf.float64))

    print('x: ', x)
    x = tf.constant(x, dtype=tf.float64)
    result = fn(x,n)

    #result = krylov(fn, x, n)

    sess = tf.InteractiveSession()
    tf.initialize_all_variables().run()

    print(sess.run(result))

def test_symm_tridiag():
    n = 4
    off_diag = np.array([2,3,4])
    diag = np.array([1,1,1,1])

    A = diags([off_diag, diag, off_diag], [-1, 0, 1], (n,n)).toarray()

    print(A)

    print(np.linalg.norm(A - A.T))

    x = np.array([1,2,3,4])

    print(np.dot(np.linalg.matrix_power(A, 3), x))

    fn = functools.partial(symm_tridiag_mult_fn, tf.constant(diag, dtype=tf.float64),
        tf.constant(off_diag, dtype=tf.float64))

    x = tf.constant(x, dtype=tf.float64)
    result = krylov(fn, x, n)

    sess = tf.InteractiveSession()
    tf.initialize_all_variables().run()

    print(sess.run(result))

if __name__ == '__main__':
    test_circ_sparsity()
    #test_tridiag_corner()
    #test_symm_tridiag()
