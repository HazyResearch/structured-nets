import tensorflow as tf
from utils import *

# Implements inversion in Theorem 2.2 in NIPS '15 paper.
def general_tf(A, B, G, H, r, m, n):
  M = tf.zeros([m,n], dtype=tf.float64)
  Jh = tf.reverse(H, [0])

  for i in range(r):
    K_A_g = krylov_tf(A, G[:, i], m)


    K_B_h = krylov_tf(B, Jh[:, i], n)

    this_prod = tf.matmul(K_A_g, K_B_h)
    M = tf.add(M, tf.transpose(this_prod))

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