import numpy as np
from scipy.sparse import diags
import tensorflow as tf
import functools

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

def tridiag_corner_transpose_mult_fn(subdiag, diag, supdiag, f, x, n):
	sub_result = tf.multiply(x[1:], subdiag)
	sup_result = tf.multiply(x[:n-1], supdiag)

	sup_result = tf.concat([[0], sup_result], axis=0)
	sub_result = tf.concat([sub_result, [0]], axis=0)

	z = tf.zeros(n-1, dtype=tf.float64)
	scaled_f = tf.multiply(f, x[0])

	f_result = tf.concat([z, scaled_f], axis=0)

	return sup_result + sub_result + tf.multiply(x, diag) + f_result

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

if __name__ == '__main__':
	n = 4
	subdiag = np.array([2,3,4])
	supdiag = np.array([4,5,6])
	diag = np.array([1,1,1,1])
	f = 5.0

	A = diags([subdiag, diag, supdiag], [-1, 0, 1], (n,n)).toarray()

	A[0, -1] = f
	print A

	x = np.array([1,2,3,4])

	print np.dot(np.linalg.matrix_power(A.T, 3), x)

	fn = functools.partial(tridiag_corner_transpose_mult_fn, tf.constant(subdiag, dtype=tf.float64), 
		tf.constant(diag, dtype=tf.float64), tf.constant(supdiag, dtype=tf.float64), tf.constant(f, dtype=tf.float64))

	x = tf.constant(x, dtype=tf.float64)
	result = krylov(fn, x, n)

	sess = tf.InteractiveSession()
	tf.initialize_all_variables().run()

	print sess.run(result)
