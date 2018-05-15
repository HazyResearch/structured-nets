from scipy.linalg import toeplitz, circulant, solve_sylvester
from scipy.sparse import diags
import numpy as np
import tensorflow as tf
import time, subprocess
import functools

def kth_diag_indices(A, k):
 rows, cols = np.diag_indices_from(A)
 if k < 0:
  return rows[-k:], cols[:k]
 elif k > 0:
  return rows[:-k], cols[k:]
 else:
  return rows, cols



def symm_tridiag_corner_mask(n):
  mask = np.zeros((n,n))
  mask[0, -1] = 1
  mask[-1, 0] = 1
  subdiag = kth_diag_indices(mask, -1)
  supdiag = kth_diag_indices(mask, -1)
  diag = kth_diag_indices(mask, 0)
  mask[subdiag] = 1
  mask[supdiag] = 1
  mask[diag] = 1

  return mask

def sylvester_disp(M, A, B):
  return np.dot(A,M) - np.dot(M,B)

def stein_disp(M, A, B):
  return M - np.dot(A,np.dot(M,B))

def compute_disp(disp_type, M, A, B):
  if disp_type == 'sylvester':
    return sylvester_disp(M,A,B)
  elif disp_type == 'stein':
    return stein_disp(M,A,B)
  else:
    print('disp_type not supported: ', disp_type)
    assert 0    

def gen_tridiag_corner_transpose(subdiag,supdiag,diag,f):
  T = diags([subdiag, diag, supdiag], [-1, 0, 1]).toarray()
  T[-1, 0] = f
  return T

def gen_tridiag_corner(subdiag, supdiag, diag, f):
  T = diags([subdiag, diag, supdiag], [-1, 0, 1]).toarray()
  T[0, -1] = f

  return T

def gen_tridiag_corners(subdiag, supdiag, diag, f_ur, f_ll):
  T = diags([subdiag, diag, supdiag], [-1, 0, 1]).toarray()
  T[0, -1] = f_ur
  T[-1, 0] = f_ll

  return T

# Two fs: upper right, lower left
def get_fs(learn_corner, init_type, stddev):
  if learn_corner:
    if init_type == 'toeplitz':
      f_ur_A = tf.Variable([1], dtype=tf.float64)
      f_ur_B = tf.Variable([-1], dtype=tf.float64)
      f_ll_A = tf.Variable([0], dtype=tf.float64)
      f_ll_B = tf.Variable([0], dtype=tf.float64)
    elif init_type == 'random':
      f_ur_A = tf.Variable(tf.truncated_normal([1], stddev=stddev, dtype=tf.float64), dtype=tf.float64)
      f_ur_B = tf.Variable(tf.truncated_normal([1], stddev=stddev, dtype=tf.float64), dtype=tf.float64)
      f_ll_A = tf.Variable(tf.truncated_normal([1], stddev=stddev, dtype=tf.float64), dtype=tf.float64)
      f_ll_B = tf.Variable(tf.truncated_normal([1], stddev=stddev, dtype=tf.float64), dtype=tf.float64)
    else:
      print('init_type not supported: ', init_type)
      assert 0 
  else:
    f_ur_A = tf.constant([1], dtype=tf.float64)
    f_ur_B = tf.constant([-1], dtype=tf.float64)
    f_ll_A = tf.constant([0], dtype=tf.float64)
    f_ll_B = tf.constant([0], dtype=tf.float64)
  return f_ur_A, f_ur_B, f_ll_A, f_ll_B

def get_f(learn_corner, init_type, stddev):
    if learn_corner:
        if init_type == 'toeplitz':
            f_A = tf.Variable([1], dtype=tf.float64)
            f_B = tf.Variable([-1], dtype=tf.float64)
        elif init_type == 'random':
            f_A = tf.Variable(tf.truncated_normal([1], stddev=stddev, dtype=tf.float64), dtype=tf.float64)
            f_B = tf.Variable(tf.truncated_normal([1], stddev=stddev, dtype=tf.float64), dtype=tf.float64)
        else:
            print('init_type not supported: ', init_type)
            assert 0 
    else:
        f_A = tf.constant([1], dtype=tf.float64)
        f_B = tf.constant([-1], dtype=tf.float64)

    return f_A, f_B

def get_subdiag(n_diag_learned, init_type, stddev):
    if n_diag_learned > 0:
        if init_type == 'toeplitz':
            x = tf.Variable(tf.ones(n_diag_learned, dtype=tf.float64))
        elif init_type == 'random':
            x = tf.Variable(tf.truncated_normal([n_diag_learned], stddev=stddev, dtype=tf.float64))
        else:
            print('init_type not supported: ', init_type)
            assert 0 
    return x

# Produce tf.Variable for subdiagonal + corner depending on params (initialization, num learned entries, etc.)
def get_x_f(n, init_type, learn_corner, n_diag_learned, stddev=0.01):
  f_A, f_B = get_f(learn_corner, init_type, stddev)
  x_A = get_subdiag(n_diag_learned, init_type, stddev)
  x_B = get_subdiag(n_diag_learned, init_type, stddev)

  # Pad
  if n_diag_learned < (n-1):
    ones = tf.ones(n-1-n_diag_learned, dtype=tf.float64)
    x_A = tf.concat([x_A, ones], axis=0)
    x_B = tf.concat([x_B, ones], axis=0)

  # Concatenate
  x_f_A = tf.concat([x_A, f_A], axis=0)
  x_f_B = tf.concat([x_B, f_B], axis=0)

  """
  f_x_A = np.ones(n_diag_learned+1)
  f_x_A = tf.convert_to_tensor(f_x_A,dtype=tf.float64)
  f_x_B = np.ones(n_diag_learned+1)
  f_x_B[0] = -1
  f_x_B = tf.convert_to_tensor(f_x_B, dtype=tf.float64)
  """

  return x_f_A, x_f_B

def get_symm_tridiag_vars(n):
  return 0

def get_symm_pos_tridiag_vars(n, init_type, stddev=0.01):
  # Constraint to be positive
  if init_type == 'random':
    diag_A = tf.Variable(tf.truncated_normal([n], stddev=stddev, dtype=tf.float64))
    off_diag_A = tf.get_variable('off_diag_A', initializer=tf.truncated_normal([n-1], stddev=stddev, dtype=tf.float64), 
      constraint=lambda x: tf.clip_by_value(x, 0, np.infty))
    diag_B = tf.Variable(tf.truncated_normal([n], stddev=stddev, dtype=tf.float64))
  elif init_type == 'chebyshev':
    # A: 0 on diagonal
    # A: 1/2 on on sub/super diagonal 
    # B: uniform(-1, 1)  
    print('chebyshev initialization')
    diag_A = tf.Variable(tf.zeros([n], dtype=tf.float64))
    off_diag_A = tf.get_variable('off_diag_A', initializer=0.5*tf.ones([n-1], dtype=tf.float64), 
      constraint=lambda x: tf.clip_by_value(x, 0, np.infty))
    diag_B = tf.Variable(tf.random_uniform([n],minval=-1, maxval=1, dtype=tf.float64))
  else:
    print('init_type not supported: ', init_type)

  return diag_A, off_diag_A, diag_B

def get_tridiag_vars(n, init_type, stddev=0.01):
  if init_type == 'toeplitz':
    subdiag = tf.Variable(tf.ones([n-1], dtype=tf.float64))
    supdiag = tf.Variable(tf.zeros([n-1], dtype=tf.float64))
    diag = tf.Variable(tf.zeros([n], dtype=tf.float64))
  elif init_type == 'random':
    subdiag = tf.Variable(tf.truncated_normal([n-1], stddev=stddev, dtype=tf.float64))
    supdiag = tf.Variable(tf.truncated_normal([n-1], stddev=stddev, dtype=tf.float64))
    diag = tf.Variable(tf.truncated_normal([n], stddev=stddev, dtype=tf.float64))
  else:
    print('init_type not supported: ', init_type)
    assert 0 
  return subdiag, supdiag, diag 

# Returns subdiag, supdiag, diag, f
def get_tridiag_corner_vars(n, init_type, stddev=0.01, learn_corner=True):
  subdiag_A, supdiag_A, diag_A = get_tridiag_vars(n, init_type, stddev)
  subdiag_B, supdiag_B, diag_B = get_tridiag_vars(n, init_type, stddev)
  
  f_A, f_B = get_f(learn_corner, init_type, stddev)
  return subdiag_A, supdiag_A, diag_A, subdiag_B, supdiag_B, diag_B, f_A, f_B

# Returns subdiag, supdiag, diag, f
def get_tridiag_corners_vars(n, init_type, stddev=0.01, learn_corner=True):
  subdiag_A, supdiag_A, diag_A = get_tridiag_vars(n, init_type, stddev)
  subdiag_B, supdiag_B, diag_B = get_tridiag_vars(n, init_type, stddev)
  
  f_ur_A, f_ur_B, f_ll_A, f_ll_B = get_fs(learn_corner, init_type, stddev)
  return subdiag_A, supdiag_A, diag_A, subdiag_B, supdiag_B, diag_B, f_ur_A, f_ur_B, f_ll_A, f_ll_B

# Returns loss, accuracy
def compute_loss_and_accuracy(y, y_, params):
  if params.loss == 'mse':
    mse = tf.reduce_mean(tf.squared_difference(y, y_),name='loss')
    accuracy = tf.constant(0,name='accuracy')
    return mse, accuracy
  elif params.loss == 'cross_entropy':
    cross_entropy = tf.reduce_mean(
     tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y), name='loss')

    # Get prediction
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32),name='accuracy')
    return cross_entropy, accuracy
  else:
    print('Not supported: ', params.loss)
    assert 0

def compute_y_cnn(x, W1, params):
  if params.dataset_name == 'cifar10' and 'grayscale' not in params.transform:
    input_layer = tf.reshape(x, [-1, 32, 32, 3])
  else:
    dim = int(np.sqrt(params.input_size))
    input_layer = tf.reshape(x, [-1, dim, dim, 1]) # Assuming single channel
  input_layer = tf.cast(input_layer, tf.float32)

  print('input ', input_layer)
  
  # Reshape to x to 32x32x3
  # Convolutional Layer #1
  conv1 = tf.layers.conv2d(
  inputs=input_layer,
  filters=params.cnn_params['c1_filters'],
  kernel_size=[params.cnn_params['c1_ksize'], params.cnn_params['c1_ksize']],
  padding="same",
  activation=tf.nn.relu)


  print('conv1', conv1)
  # Pooling Layer #1
  pool1 = tf.layers.max_pooling2d(inputs=conv1, 
    pool_size=[params.cnn_params['p1_size'], params.cnn_params['p1_size']], 
    strides=params.cnn_params['p1_strides'])

  print('pool1 ', pool1)

  if params.num_conv_layers == 1:
    pool2_flat = tf.reshape(pool1, [-1, params.cnn_params['p2_flat_size']])
    pool2_flat = tf.cast(pool2_flat, tf.float64)

    print('pool2_flat: ', pool2_flat)

    dense = tf.nn.relu(tf.matmul(pool2_flat, W1))
  else:
    # Convolutional Layer #2 and Pooling Layer #2
    conv2 = tf.layers.conv2d(
     inputs=pool1,
     filters=params.cnn_params['c2_filters'],
     kernel_size=[params.cnn_params['c2_ksize'], params.cnn_params['c2_ksize']],
     padding="same",
     activation=tf.nn.relu)
    
    print('conv2 ', conv2)

    pool2 = tf.layers.max_pooling2d(inputs=conv2, 
      pool_size=[params.cnn_params['p2_size'], params.cnn_params['p2_size']], 
      strides=params.cnn_params['p2_strides'])
    
    print('pool2 ', pool2)
    # Dense Layer: replace with structured matrix
    pool2_flat = tf.reshape(pool2, [-1, params.cnn_params['p2_flat_size']])
    pool2_flat = tf.cast(pool2_flat, tf.float64)

    print('pool2_flat: ', pool2_flat)

    dense = tf.nn.relu(tf.matmul(pool2_flat, W1))

  print('dense ', dense)
  #dense = tf.layers.dense(inputs=pool3_flat, units=1024, activation=tf.nn.relu)

  # Logits Layer
  logits = tf.layers.dense(inputs=dense, units=params.out_size,name='forward')
  print('dense ', dense)

  
  print('logits', logits)

  return logits

def get_commit_id():
  return subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD'])

def gen_operators(params):
  if params.disp_type == 'sylvester':
    return gen_sylvester_operators(params.class_type, params.layer_size, params.layer_size)
  elif params.disp_type == 'stein':
    return gen_stein_operators(params.class_type, params.layer_size, params.layer_size)
  else:
    print('disp_type not supported: ', params.disp_type)
    assert 0

# Operators for Stein type displacement.
def gen_sylvester_operators(class_type, m, n):
  if class_type.startswith('toeplitz'):
    A = gen_Z_f(m, 1)
    B = gen_Z_f(n, -1)
  elif class_type.startswith('hankel'):
    A = gen_Z_f(m, 1)
    B = gen_Z_f(n, 0).T
  elif class_type.startswith('t+h'):
    A = gen_Z_f(m, 0) + gen_Z_f(m, 0).T
    B = gen_Z_f(n, 0) + gen_Z_f(n, 0).T
  elif class_type.startswith('vandermonde'):
    v = np.random.random(n)
    A = np.diag(v)
    B = gen_Z_f(n, 0)
  elif class_type == 'random':
    A = np.random.random((m, m))
    B = np.random.random((n, n))
  else:
    print('Class ' + prefix + ' not supported')
    assert 0
  return A,B

# Operators for Stein type displacement.
def gen_stein_operators(class_type, m, n):
  if class_type.startswith('toeplitz'):
    A = gen_Z_f(m, 1).T
    B = gen_Z_f(n, -1)
  elif class_type.startswith('hankel'):
    A = gen_Z_f(m, 0)
    B = gen_Z_f(n, 1)
  elif class_type.startswith('vandermonde'):
    v = np.random.random(n)
    A = np.diag(v)
    B = gen_Z_f(n, 0)
  elif class_type == 'random':
    A = np.random.random((m, m))
    B = np.random.random((n, n))
  else:
    print('Class ' + prefix + ' not supported')
    assert 0
  return A,B

# Operators for Stein type displacement.
def gen_stein_operators_tf(init, m, n):
  A,B = gen_stein_operators(init, m, n)
  return tf.Variable(A), tf.Variable(B)

def gen_trid_mask(n):
  ones1 = list(np.ones(n))
  ones2 = list(np.ones(n-1))
  data = [ones1, ones2, ones2]
  positions = [0, 1, -1]

  mask = diags(data, positions, (n, n)).toarray()

  return mask

def gen_trid_corner_mask(n):
  mask = gen_trid_mask(n)
  mask[0, -1] = 1

  return mask

# Circulant sparsity pattern
def gen_Z_f(m, f, v=None):
  if v is not None:
    assert v.size <= m-1
  I_m = np.eye(m-1, m-1)
  Z_f = np.hstack((I_m, np.zeros((m-1, 1))))
  Z_f = np.vstack((np.zeros((1, m)), Z_f)) 
  Z_f[0, -1] = f
  if v is not None:
    for i in range(v.size):
      Z_f[i+1, i] = v[i]

  return Z_f

"""
def gen_Z_f(m, f):
  I_m = np.eye(m-1, m-1)
  Z_f = np.hstack((I_m, np.zeros((m-1, 1))))
  Z_f = np.vstack((np.zeros((1, m)), Z_f)) 
  Z_f[0, -1] = f
  return Z_f
"""

def gen_circ_scaling_mask(n):
  M = np.zeros((n,n))
  for i in range(n-1):
    M[i, -(i+1):] = 1 #Last i+1

  return np.roll(M, 2, 0).astype(np.bool)

# Shift rows circularly by num_shifts shifts.
# Or just multiply by Z_1 - which is faster?
def tf_roll_rows(x, num_shifts):
  if num_shifts == 0:
    return x

  x = tf.transpose(x)
  x_len = x.get_shape().as_list()[1] 
  y = tf.concat([x[:,x_len-num_shifts:], x[:,:x_len-num_shifts]], axis=1)
  return tf.transpose(y)

# Replace all 0's with 1's
def update_mask(scale, mask):
  all_ones = tf.ones(mask.get_shape(), dtype=tf.float64)
  return tf.where(mask, scale*all_ones, all_ones)

def gen_circ_scaling_tf(x, mask, num_learned):
  if x is None:
    return tf.ones(mask.get_shape(), dtype=tf.float64)

  t0 = time.time()
  final_mask = update_mask(x[0], mask)
  print('time of first update_mask call: ', time.time() - t0)

  sum_update = 0
  sum_mult = 0

  for i in np.arange(1, num_learned):
    # Shift mask
    t1 = time.time()
    shifted_mask = update_mask(x[i], tf_roll_rows(mask, i))
    sum_update += (time.time() - t1)
    t2 = time.time()
    final_mask = tf.multiply(final_mask, shifted_mask)
    sum_mult += (time.time() - t2)

  print('time of all update_mask calls: ', sum_update)
  print('time of all tf.multiply calls: ', sum_mult)    
  return final_mask

def gen_f_mask(f, m,n):
  mask = np.ones((m, n))
  f_mask = f*mask

  # Set lower triangular indices to 1: 1 if row>=col
  il1 = np.tril_indices(n=f_mask.shape[0], m=f_mask.shape[1])
  f_mask[il1] = 1 
  f_mask = tf.constant(f_mask, dtype=tf.float64)

  return f_mask

def gen_index_arr(n):
  r = np.arange(0, n)
  a = np.expand_dims(r, -1)
  b = np.expand_dims(-r, 0)

  index = a+b

  # Create mask
  pos = (index < 0)

  antipos = 1-pos

  updated = np.multiply(antipos, index) + np.multiply(pos, index+n)

  return np.expand_dims(updated, -1)

def sylvester(A, B, n, r):
  # Generate random rank r error matrix
  G = np.random.random((n, r))
  H = np.random.random((n, r))
  GH = np.dot(G,H.T)

  # Solve Sylvester equation to recover M
  # Such that AM - MB = GH^T
  M = solve_sylvester(A, -B, GH)

  E = np.dot(A,M) - np.dot(M,B)

  assert np.linalg.norm(E - GH) <= 1e-10

  return M

def gen_matrix(n, prefix, r=2):
  if prefix.startswith('rank'):
    r = int(prefix[-1])
    G = np.random.random((n,r))
    H = np.random.random((n,r))
    return np.dot(G,H.T)
  elif prefix == 'toeplitz':
    c = np.random.random(n)
    r = np.random.random(n)
    return toeplitz(c,r)
  elif prefix == 'hankel':
    c = np.random.random(n)
    r = np.random.random(n)
    return np.flipud(toeplitz(c,r)) 
  elif prefix == 'vandermonde':
    v = np.random.random(n)
    return np.vander(v, n, increasing=True)
  elif prefix == 'cauchy':
    s = np.random.random(n)
    t = np.random.random(n)
    return 1.0 / (s.reshape((-1,1)) - t)   
  elif prefix == 'random':
    return np.random.random((n,n))
  elif prefix == 'toeplitz-like':
    # Generate operators
    A = gen_Z_f(n, 1)
    B = gen_Z_f(n, -1)
    # Generate random rank r error matrix
    # Solve sylvester
    return sylvester(A, B, n, r)
  elif prefix == 'hankel-like':
    # Generate operators
    A = gen_Z_f(n, 1)
    B = gen_Z_f(n, 0).T
    return sylvester(A, B, n, r)
  elif prefix == 'vandermonde-like':
    # Generate operators
    v = np.random.random(n)
    #v = np.linalg.eigvals(Z1)
    V = np.vander(v, increasing=True)

    # Generate operators
    A = np.diag(v)
    B = gen_Z_f(n, 0)

    return sylvester(A, B, n, r)

  elif prefix == 'cauchy-like':
    s = np.random.random(n)
    t = np.random.random(n)

    C = 1.0 / (s.reshape((-1,1)) - t)

    # Generate operators
    A = np.diag(s)
    B = np.diag(t)

    return sylvester(A, B, n, r)

  elif prefix == 'tridiag_corner': # 
    # Generate random tridiagonal+corner operators
    A = np.random.random((n,n))
    B = np.random.random((n,n))
    mask = gen_trid_corner_mask(n)
    A = np.multiply(A, mask)
    B = np.multiply(B, mask)

    return sylvester(A, B, n, r)
  elif prefix == 'circ_sparsity':
    # Generate random circulant sparsity pattern operators
    A = np.random.random((n,n))
    B = np.random.random((n,n))
    mask = gen_Z_f(n, 1)
    A = np.multiply(A, mask)
    B = np.multiply(B, mask)

    return sylvester(A, B, n, r)
  else:
    print('Type ' + prefix + ' not supported')
    assert 0


def gen_batch(A, N, P=None):
  """
  Generates N random x's, computes corresponding y's, such that Ax = y.
  A: the matrix.
  N: number of datapoints.
  P: if not None, then with probability 0.5, return (Px, APx). Otherwise return (x, Ax).
  """

  X = np.random.random((A.shape[1], N))

  if P is not None:
    if np.random.random() >= 0.5:
      X = np.dot(P,X)

  Y = np.dot(A, X)

  assert np.isclose(np.linalg.norm(Y[:, 0] - np.dot(A, X[:, 0])), 0)

  return X.T,Y.T

def get_GH(E):
  disp_rank = np.linalg.matrix_rank(E)

  # SVD
  U, S, V = np.linalg.svd(E, full_matrices=False)

  SV = np.dot(np.diag(S), V)
  G = U[:, 0:disp_rank]
  H = SV[0:disp_rank, :].T

  return G,H, disp_rank

def sylvester_project(M, A, B, r):
  """
  Project via SVD on error matrix + solving the Sylvester equation.
  """
  t = time.time()
  E = np.dot(A, M) - np.dot(M, B)
  G,H,dr = get_GH(E)

  G_r = G[:, 0:r]
  H_r = H[:, 0:r]

  lowrank = np.dot(G_r, H_r.T)

  print('norm(E-lowrank): ', np.linalg.norm(E-lowrank))

  # Sylvester solve
  M_class = solve_sylvester(A, -B, lowrank)

  print('rank(lowrank): ', np.linalg.matrix_rank(lowrank))
  print('rank(A): ', np.linalg.matrix_rank(A))
  print('rank(B): ', np.linalg.matrix_rank(B))
  print('norm(M-M_class): ', np.linalg.norm(M-M_class))

  E_class = np.dot(A, M_class) - np.dot(M_class, B)
  print('rank of E_class',np.linalg.matrix_rank(E_class))
  #print 'eigvals of E_class',np.linalg.eigvals(E_class)

  print('time of sylv project: ', time.time() - t)

  return M_class

def circulant_tf(vec, index_arr, mask=None):
  # Slice 2D
  output = tf.gather_nd(vec, index_arr)

  # Apply mask
  if mask is not None:
    output = tf.multiply(output, mask)

  return output

# Shape of stack_circ: (v.size, n)
def circulant_mn_tf(v, index_arr, n, num_reps, f_mask):
  circ_v = circulant_tf(v, index_arr)

  multiples = tf.constant([1, num_reps]) 

  stack_circ = tf.tile(circ_v, multiples)
  stack_circ = tf.cast(stack_circ[:, 0:n], tf.float64)

  # Element-wise multiplication
  masked = tf.multiply(f_mask, stack_circ)

  return masked 

# K(Z_f^T, g) = scaling_term*J*Z_f(Jg)
# x.size == num_learned
def krylov_circ_transpose(n, x, v, num_learned, f_mask, scaling_mask, index_arr):
  # Generate mask
  t2 = time.time()
  # Get scaling term
  scale_term = gen_circ_scaling_tf(x, scaling_mask, num_learned)
  print('time of gen_circ_scaling_tf', time.time() - t2)

  # Call circulant_tf on flipped v
  t3 = time.time()
  Z = circulant_tf(tf.reverse(v, [0]), index_arr, f_mask)
  print('time of circulant_tf', time.time() - t3)

  # Flip row-wise
  JZ = tf.reverse(Z, [0])

  # Elementwise multiplication by scale_term
  return tf.multiply(scale_term, JZ)

def krylov_tf(A, v, n):
  v_exp = tf.expand_dims(v,1)
  cols = [v_exp]
  this_pow = A

  for i in range(n-1):
    this_col = tf.matmul(this_pow, v_exp)

    this_pow = tf.matmul(A, this_pow)

    cols.append(this_col)

    K = tf.stack(cols)

  return tf.transpose(tf.squeeze(K))

def Ax_circ(f_v, x, n):
  # Circular shift x to the right
  y = tf.concat([x[n-1:], x[:n-1]], axis=0)
  
  # Scale by [f v]
  return tf.multiply(y, f_v)

def krylov_tf_circ(f_x, v, n):
  v_exp = tf.expand_dims(v,1)
  cols = [v_exp]
  this_col = v

  for i in range(n-1):
    this_col = Ax_circ(f_x, this_col, n)

    cols.append(tf.expand_dims(this_col,1))

  K = tf.stack(cols)

  return tf.transpose(tf.squeeze(K))

def V_mn(v, m, n):
  # Stack columns
  # First col: ones
  # Second col: v
  # Subsequent cols: v^{c-1}
  ones = tf.ones([m], dtype=tf.float64)

  cols = [ones, v]

  for i in range(n-2):
    this_col = tf.pow(v, i+2)
    cols.append(this_col)

  V = tf.transpose(tf.stack(cols))

  return tf.cast(V, tf.float64)
