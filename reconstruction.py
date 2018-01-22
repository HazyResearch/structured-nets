import tensorflow as tf
from utils import *

def circ_sparsity_recon(G, H, n, r, learn_corner, n_diag_learned, init_type, stddev):
  if learn_corner:
    if init_type == 'toeplitz':
      f_A = tf.Variable([1], dtype=tf.float64)
      f_B = tf.Variable([-1], dtype=tf.float64)
    elif init_type == 'random':
      f_A = tf.Variable(tf.truncated_normal([1], stddev=stddev, dtype=tf.float64), dtype=tf.float64)
      f_B = tf.Variable(tf.truncated_normal([1], stddev=stddev, dtype=tf.float64), dtype=tf.float64)  
    else:
      print 'init_type not supported: ', init_type
      assert 0   
  else:
    f_A = tf.constant([1], dtype=tf.float64)
    f_B = tf.constant([-1], dtype=tf.float64)

  # diag: first n_learned entries 
  v_A = None
  v_B = None
  if n_diag_learned > 0:
    if init_type == 'toeplitz':
      v_A = tf.Variable(tf.ones(n_diag_learned, dtype=tf.float64))
      v_B = tf.Variable(tf.ones(n_diag_learned, dtype=tf.float64))
    elif init_type == 'random':
      v_A = tf.Variable(tf.truncated_normal([n_diag_learned], stddev=stddev, dtype=tf.float64))
      v_B = tf.Variable(tf.truncated_normal([n_diag_learned], stddev=stddev, dtype=tf.float64))     
    else:
      print 'init_type not supported: ', init_type
      assert 0  

  scaling_mask = tf.constant(gen_circ_scaling_mask(n))

  f_mask_pattern = tf.constant([[True if j > k else False for j in range(n)] for k in range(n)])
  all_ones = tf.ones(f_mask_pattern.get_shape(), dtype=tf.float64)

  f_A_mask = tf.where(f_mask_pattern, f_A*all_ones, all_ones)
  f_B_mask = tf.where(f_mask_pattern, f_B*all_ones, all_ones)

  # Reconstruct W1 from G and H
  index_arr = gen_index_arr(n)

  W1 = tf.zeros([n, n], dtype=tf.float64)
  for i in range(r):
    prod = circ_sparsity_recon_rank1(n, v_A, v_B, G[:, i], H[:, i], f_A_mask, f_B_mask, scaling_mask, index_arr, n_diag_learned)
    W1 = tf.add(W1, prod)

  # Compute a and b
  a = f_A
  b = f_B
  if v_A is not None:
    a *= tf.reduce_prod(v_A)
  if v_B is not None:
    b *= tf.reduce_prod(v_B)

  coeff = 1.0/(1 - a*b)

  #coeff = tf.Print(coeff,[coeff], message="my W1-values:") # <-------- TF PRINT STATMENT

  W1_scaled = tf.scalar_mul(coeff[0], W1)

  return W1_scaled, f_A, f_B, v_A, v_B 

#assumes g and h are vectors.
#K(Z_f^T, g)*K(Z_f^T, h)^T
def circ_sparsity_recon_rank1(n, v_A, v_B, g, h, f_A_mask, f_B_mask, scaling_mask, index_arr, num_learned):
  K1 = krylov_circ_transpose(n, v_A, g, num_learned, f_A_mask, scaling_mask, index_arr)
  K2 = krylov_circ_transpose(n, v_B, h, num_learned, f_B_mask, scaling_mask, index_arr)

  prod = tf.matmul(K1, tf.transpose(K2))

  return prod

# Implements inversion in Theorem 2.2 in NIPS '15 paper.
def general_tf(A, B, G, H, r, m, n):
  M = tf.zeros([m,n], dtype=tf.float64)

  for i in range(r):
    K_A_g = krylov_tf(A, G[:, i], m)
    K_B_h = tf.transpose(krylov_tf(tf.transpose(B), H[:, i], n))

    this_prod = tf.matmul(K_A_g, K_B_h)
    M = tf.add(M, this_prod)

  return 0.5*M

def compute_J_term(m, n, B, e):
  term = np.eye(n) - e*np.linalg.matrix_power(B, m)
  term_inv = np.linalg.inv(term)
  J = np.flipud(np.eye(n))#np.flip(np.eye(n), axis=0)

  # Multiply by J
  return np.dot(J, term_inv)

def rect_recon_tf(G, H, B, m, n, e, f, r):  
  e_mask = tf.constant([[e if j > k else 1 for j in range(m)] for k in range(m)], dtype=tf.float64)
  f_mask = gen_f_mask(f,n,m)
  num_reps = int(np.ceil(float(m)/n))

  # Compute J-term: once
  J_term = compute_J_term(m, n, B, e)

  index_arr_m = gen_index_arr(m)
  index_arr_n = gen_index_arr(n)

  recon_mat_partial = tf.zeros([m, n], dtype=tf.float64)

  Jh = tf.reverse(H, [0])

  for i in range(r):
    Zg_i = circulant_tf(G[:, i], index_arr_m, e_mask)
    Zh_i = circulant_mn_tf(Jh[:, i], index_arr_n, m, num_reps, f_mask)

    this_prod = tf.matmul(Zg_i, tf.transpose(Zh_i))
    recon_mat_partial = tf.add(recon_mat_partial, this_prod)
  
  recon_mat_partial = tf.matmul(recon_mat_partial, J_term)

  return recon_mat_partial


def toeplitz_recon(r, c):
  return 0

def toeplitz_like_recon(G, H, n, r):
  W1 = tf.zeros([n, n], dtype=tf.float64)
  f = 1
  g = -1
  f_mask = tf.constant([[f if j > k else 1 for j in range(n)] for k in range(n)], dtype=tf.float64)
  g_mask = tf.constant([[g if j > k else 1 for j in range(n)] for k in range(n)], dtype=tf.float64)
  index_arr = gen_index_arr(n)

  for i in range(r):
    Z_g_i = circulant_tf(G[:, i], index_arr, f_mask)
    Z_h_i = circulant_tf(tf.reverse(H[:, i], tf.constant([0])), index_arr, g_mask)
    prod = tf.matmul(Z_g_i, Z_h_i)
    W1 = tf.add(W1, prod)

  W1 = tf.scalar_mul(0.5, W1)

  return W1


# Pan's Vandermonde specific reconstruction.
def vand_recon(G, H, v, m, n, f, r):
  # Create vector of fv_i^n
  raised = tf.pow(v, n)
  scaled = tf.cast(tf.scalar_mul(f, raised), dtype=tf.float64)
  denom = tf.subtract(tf.constant(1, dtype=tf.float64), scaled)
  divided = tf.divide(tf.constant(1, dtype=tf.float64), denom)
  D = tf.diag(divided)

  index_arr = gen_index_arr(n)
  f_mask = gen_f_mask(f, n,n)

  recon = tf.zeros([m,n], dtype=tf.float64)

  for i in range(r):
    D_g_i = tf.diag(G[:, i])
    V_v = V_mn(v, m, n)
    Z_h_i = circulant_tf(H[:, i], index_arr, f_mask)
    Z_h_i = tf.transpose(Z_h_i)

    this_prod = tf.matmul(D_g_i, V_v)
    this_prod = tf.matmul(this_prod, Z_h_i)

    recon = tf.add(recon, this_prod)

  recon = tf.matmul(D, recon)

  return recon