from scipy.sparse import diags
import numpy as np
import tensorflow as tf
import functools
from reconstruction import *
from utils import *
from krylov import *

def check_rank(sess, x, y_, batch_xs, batch_ys, params, model):
    if not params.check_disp:
        return 

    if params.class_type in ['unconstrained', 'symmetric']:
        if params.class_type == 'symmetric':
            A = sess.run(model['A'], feed_dict={x: batch_xs, y_: batch_ys})
            B = sess.run(model['B'], feed_dict={x: batch_xs, y_: batch_ys})

            print('A: ', A)
            print('B: ', B)

        return
    if params.class_type in ['toeplitz_like', 'hankel_like']:
        A,B = gen_operators(params)
        W = sess.run(model['W'], feed_dict={x: batch_xs, y_: batch_ys})
        print('W: ', W.shape)
        E = compute_disp(params.disp_type, W, A, B)

        dr = np.linalg.matrix_rank(E)
        norm_res = np.linalg.norm(E)
        norm_W = np.linalg.norm(W)

    elif params.class_type == 'symm_tridiag_pan':
        return
    elif params.class_type == 'symm_tridiag_krylov':
        return
    elif params.class_type == 'tridiagonal_corner':
        # Construct A
        supdiag_A = sess.run(model['supdiag_A'], feed_dict={x: batch_xs, y_: batch_ys})
        diag_A = sess.run(model['diag_A'], feed_dict={x: batch_xs, y_: batch_ys})
        subdiag_A = sess.run(model['subdiag_A'], feed_dict={x: batch_xs, y_: batch_ys})
        f_A = sess.run(model['f_A'], feed_dict={x: batch_xs, y_: batch_ys})
        # Construct B
        supdiag_B = sess.run(model['supdiag_B'], feed_dict={x: batch_xs, y_: batch_ys})
        diag_B = sess.run(model['diag_B'], feed_dict={x: batch_xs, y_: batch_ys})
        subdiag_B = sess.run(model['subdiag_B'], feed_dict={x: batch_xs, y_: batch_ys})
        f_B = sess.run(model['f_B'], feed_dict={x: batch_xs, y_: batch_ys})

        #print 'subdiag_A: ', subdiag_A
        #print 'supdiag_A: ', supdiag_A
        #print 'diag_A: ', diag_A

        A = gen_tridiag_corner(subdiag_A, supdiag_A, diag_A, f_A).T
        B = gen_tridiag_corner(subdiag_B, supdiag_B, diag_B, f_B)

        W = sess.run(model['W'], feed_dict={x: batch_xs, y_: batch_ys})
        E = compute_disp(params.disp_type, W, A, B)

        dr = np.linalg.matrix_rank(E)
        norm_res = np.linalg.norm(E)
        norm_W = np.linalg.norm(W)

    elif params.class_type == 'circulant_sparsity':
        # Construct A
        x_f_A = sess.run(model['x_f_A'], feed_dict={x: batch_xs, y_: batch_ys})
        # Construct B
        x_f_B = sess.run(model['x_f_B'], feed_dict={x: batch_xs, y_: batch_ys})

        if params.fix_A_identity:
            A = np.eye(params.layer_size)
        else:
          A = gen_Z_f(params.layer_size, x_f_A[-1], x_f_A[:-1]).T

        B = gen_Z_f(params.layer_size, x_f_B[-1], x_f_B[:-1])

        W = sess.run(model['W'], feed_dict={x: batch_xs, y_: batch_ys})
        E = compute_disp(params.disp_type, W, A, B)
        dr = np.linalg.matrix_rank(E)
        norm_res = np.linalg.norm(E)
        norm_W = np.linalg.norm(W)

    elif params.class_type == 'tridiagonal_corners':
        # Construct A
        subdiag_A, supdiag_A, diag_A, f_ur_A, f_ll_A = sess.run([model['subdiag_A'], model['supdiag_A'], model['diag_A'], model['f_ur_A'], model['f_ll_A']], feed_dict={x: batch_xs, y_: batch_ys})
        A = gen_tridiag_corners(subdiag_A, supdiag_A, diag_A, f_ur_A, f_ll_A).T
        # Construct B
        subdiag_B, supdiag_B, diag_B, f_ur_B, f_ll_B = sess.run([model['subdiag_B'], model['supdiag_B'], model['diag_B'], model['f_ur_B'], model['f_ll_B']], 
            feed_dict={x: batch_xs, y_: batch_ys})     
        B = gen_tridiag_corners(subdiag_B, supdiag_B, diag_B, f_ur_B, f_ll_B)
        W = sess.run(model['W'], feed_dict={x: batch_xs, y_: batch_ys})
        E = compute_disp(params.disp_type, W, A, B)
        dr = np.linalg.matrix_rank(E)
        norm_res = np.linalg.norm(E)
        norm_W = np.linalg.norm(W)

    elif params.class_type == 'low_rank':
        E = sess.run(model['W'], feed_dict={x: batch_xs, y_: batch_ys})

        dr = np.linalg.matrix_rank(E)
        norm_res = 0
        norm_W = np.linalg.norm(E)
    elif params.class_type == 'vandermonde_like':
        v, W = sess.run([model['v'], model['W']], feed_dict={x: batch_xs, y_: batch_ys})
        A = np.diag(v)
        B = gen_Z_f(params.layer_size, 0).T
        E = compute_disp(params.disp_type, W, A, B)
        dr = np.linalg.matrix_rank(E)
        norm_res = np.linalg.norm(E)
        norm_W = np.linalg.norm(W)
    else:
        print('class_type not supported: ', params.class_type)
        assert 0 
    ratio = norm_res/norm_W
    print(E.shape)
    print(('(Displacement) Rank: ', np.linalg.matrix_rank(E)))
    print(('||E||/||W||: ', ratio))
    eigvals_E = np.abs(np.linalg.eigvals(E))
    eigvals_W = np.abs(np.linalg.eigvals(W))
    eigvals_A = np.abs(np.linalg.eigvals(A))
    eigvals_B = np.abs(np.linalg.eigvals(B))
    #print('eigvals_E: ', eigvals_E)
    #print('eigvals_W: ', eigvals_W)
    #print('eigvals_A: ', eigvals_A)
    #print('eigvals_B: ', eigvals_B)
    return dr, norm_res, norm_W, eigvals_E, eigvals_W, eigvals_A, eigvals_B, E, W, A, B

def get_structured_W(params):
    model = {}
    if params.class_type == 'unconstrained':
        W = tf.Variable(tf.truncated_normal([params.layer_size, params.layer_size], stddev=params.init_stddev, dtype=tf.float64))
        if params.check_disp or params.viz_freq > 0:
            model['W'] = W
        return W, model
    elif params.class_type in ['low_rank', 'symm_tridiag_corner_pan', 'symm_tridiag_corner_krylov','symmetric', 'toeplitz_like', 
        'vandermonde_like', 'hankel_like', 'circulant_sparsity', 'tridiagonal_corner', 'tridiagonal_corners']:
        G = tf.Variable(tf.truncated_normal([params.layer_size, params.r], stddev=params.init_stddev, dtype=tf.float64))
        H = tf.Variable(tf.truncated_normal([params.layer_size, params.r], stddev=params.init_stddev, dtype=tf.float64))
        model['G'] = G
        model['H'] = H
        if params.class_type == 'low_rank':
            W = tf.matmul(G, tf.transpose(H))
        elif params.class_type == 'symm_tridiag_corner_pan':
            mask = symm_tridiag_corner_mask(n)
            A = tf.Variable(tf.truncated_normal([params.layer_size, params.layer_size], stddev=params.init_stddev, dtype=tf.float64))
            B = tf.Variable(tf.truncated_normal([params.layer_size, params.layer_size], stddev=params.init_stddev, dtype=tf.float64))

            A = tf.multiply(A, mask)
            B = tf.multiply(B, mask)

            W = general_recon(G, H, A, B)
            if params.check_disp or params.viz_freq > 0:
                model['A'] = A
                model['B'] = B
        elif params.class_type == 'symm_tridiag_corner_krylov':
            diag_A = tf.Variable(tf.truncated_normal([params.layer_size], stddev=params.init_stddev, dtype=tf.float64))
            off_diag_A = tf.Variable(tf.truncated_normal([params.layer_size-1], stddev=params.init_stddev, dtype=tf.float64))
            diag_B = tf.Variable(tf.truncated_normal([params.layer_size], stddev=params.init_stddev, dtype=tf.float64))
            off_diag_B = tf.Variable(tf.truncated_normal([params.layer_size-1], stddev=params.init_stddev, dtype=tf.float64))
            f_A = tf.Variable(tf.truncated_normal([1], stddev=params.init_stddev, dtype=tf.float64))
            f_B = tf.Variable(tf.truncated_normal([1], stddev=params.init_stddev, dtype=tf.float64))
            if params.check_disp or params.viz_freq > 0:
                model['diag_A'] = diag_A
                model['off_diag_A'] = off_diag_A
                model['f_A'] = f_A
                model['diag_B'] = diag_B
                model['off_diag_B'] = off_diag_B
                model['f_B'] = f_B

            fn_A = functools.partial(tridiag_corners_transpose_mult_fn, off_diag_A, diag_A, off_diag_A, f_A, f_A)
            fn_B = functools.partial(tridiag_corners_transpose_mult_fn, off_diag_B, diag_B, off_diag_B, f_B, f_B)
            W = krylov_recon(params, G, H, fn_A, fn_B)
            # Compute a and b
            a = tf.multiply(f_A, tf.reduce_prod(subdiag_A))
            b = tf.multiply(f_B, tf.reduce_prod(subdiag_B))

            coeff = 1.0/(1 - a*b)

            W = tf.multiply(coeff, W)

        elif params.class_type == 'symmetric':
            # Initialization with T+H operators
            Z1 = gen_Z_f(params.layer_size, 1)
            Z1m = gen_Z_f(params.layer_size, -1)

            op_A = Z1 + Z1.T
            op_B = Z1m + Z1m.T

            #print 'op_A: ', op_A
            #print 'op_B: ', op_B

            op_A = np.random.random((params.layer_size, params.layer_size))
            op_B = np.random.random((params.layer_size, params.layer_size))

            A = tf.Variable(op_A)
            A_upper = tf.matrix_band_part(A, 0, -1)
            A_symm = 0.5 * (A_upper + tf.transpose(A_upper))

            B = tf.Variable(op_B)
            B_upper = tf.matrix_band_part(B, 0, -1)
            B_symm = 0.5 * (B_upper + tf.transpose(B_upper))

            W = general_recon(G, H, A_symm, B_symm)
            if params.check_disp or params.viz_freq > 0:
                model['A'] = A_symm
                model['B'] = B_symm
        elif params.class_type == 'toeplitz_like':
            W = toeplitz_like_recon(G, H, params.layer_size, params.r)
        elif params.class_type == 'hankel_like':
            f = 0
            g = 1
            B = gen_Z_f(params.layer_size, g)
            W = rect_recon_tf(G, H, B, params.layer_size, params.layer_size, f, g, params.r)
        elif params.class_type == 'vandermonde_like':
            f_V = 0
            v = tf.Variable(tf.truncated_normal([params.layer_size], stddev=params.init_stddev, dtype=tf.float64))
            model['v'] = v
            W = vand_recon(G, H, v, params.layer_size, params.layer_size, f_V, params.r)
        elif params.class_type == 'circulant_sparsity':
            x_f_A, x_f_B = get_x_f(params.layer_size, params.init_type, params.learn_corner, params.n_diag_learned, params.init_stddev)

            if params.learn_diagonal:
                diag_A = tf.Variable(tf.zeros(params.layer_size, dtype=tf.float64))
                diag_B = tf.Variable(tf.zeros(params.layer_size, dtype=tf.float64))
                fn_A = functools.partial(circ_diag_transpose_mult_fn, x_f_A, diag_A)
                fn_B = functools.partial(circ_diag_transpose_mult_fn, x_f_B, diag_B)

            elif params.fix_A_identity:
                # f_x_A is unused
                print('fixing A to be identity')
                fn_A = identity_mult_fn
                fn_B = functools.partial(circ_transpose_mult_fn, x_f_B)

            else:
                fn_A = functools.partial(circ_transpose_mult_fn, x_f_A)
                fn_B = functools.partial(circ_transpose_mult_fn, x_f_B)

            W = krylov_recon(params, G, H, fn_A, fn_B)

            # Compute a and b
            a = tf.reduce_prod(x_f_A)
            b = tf.reduce_prod(x_f_B)

            coeff = 1.0/(1 - a*b)

            W = tf.scalar_mul(coeff, W)
            if params.check_disp or params.viz_freq > 0:
                model['x_f_A'] = x_f_A
                model['x_f_B'] = x_f_B
        elif params.class_type == 'tridiagonal_corners':
            subdiag_A, supdiag_A, diag_A, subdiag_B, supdiag_B, diag_B, f_ur_A, f_ur_B, f_ll_A, f_ll_B = get_tridiag_corners_vars(params.layer_size, params.init_type, params.init_stddev, params.learn_corner)
            if params.check_disp or params.viz_freq > 0:
                model['subdiag_A'] = subdiag_A
                model['supdiag_A'] = supdiag_A
                model['diag_A'] = diag_A
                model['f_ur_A'] = f_ur_A
                model['f_ur_B'] = f_ur_B
                model['f_ll_A'] = f_ll_A
                model['f_ll_B'] = f_ll_B
                model['subdiag_B'] = subdiag_B
                model['supdiag_B'] = supdiag_B
                model['diag_B'] = diag_B

            fn_A = functools.partial(tridiag_corners_transpose_mult_fn, subdiag_A, diag_A, supdiag_A, f_ur_A, f_ll_A)
            fn_B = functools.partial(tridiag_corners_transpose_mult_fn, subdiag_B, diag_B, supdiag_B, f_ur_B, f_ll_B)
            W = krylov_recon(params, G, H, fn_A, fn_B)
            # Compute a and b
            a = tf.multiply(f_ur_A, tf.reduce_prod(subdiag_A))
            b = tf.multiply(f_ur_B, tf.reduce_prod(subdiag_B))

            coeff = 1.0/(1 - a*b)

            W = tf.multiply(coeff, W)
        elif params.class_type == 'tridiagonal_corner':
            subdiag_A, supdiag_A, diag_A, subdiag_B, supdiag_B, diag_B, f_A, f_B = get_tridiag_corner_vars(params.layer_size, params.init_type, params.init_stddev, params.learn_corner)
            if params.check_disp or params.viz_freq > 0:
                model['subdiag_A'] = subdiag_A
                model['supdiag_A'] = supdiag_A
                model['diag_A'] = diag_A
                model['f_A'] = f_A
                model['f_B'] = f_B
                model['subdiag_B'] = subdiag_B
                model['supdiag_B'] = supdiag_B
                model['diag_B'] = diag_B
            
            fn_A = functools.partial(tridiag_corner_transpose_mult_fn, subdiag_A, diag_A, supdiag_A, f_A)
            fn_B = functools.partial(tridiag_corner_transpose_mult_fn, subdiag_B, diag_B, supdiag_B, f_B)
            W = krylov_recon(params, G, H, fn_A, fn_B)
            # Compute a and b
            a = tf.multiply(f_A, tf.reduce_prod(subdiag_A))
            b = tf.multiply(f_B, tf.reduce_prod(subdiag_B))

            coeff = 1.0/(1 - a*b)

            W = tf.multiply(coeff, W)
        if params.check_disp or params.viz_freq > 0:
            model['W'] = W
        return W, model


    else:
        print('Not supported: ', params.class_type)   
        assert 0    

def forward(x, params):
    W, model = get_structured_W(params)
    y = compute_y(x, W, params)
    return y, model

def compute_y(x, W1, params):
    if 'cnn' in params.transform:
        return compute_y_cnn(x, W1, params)
    elif params.num_layers==0:
        y = tf.matmul(x, W1,name='forward')
        return y
    elif params.num_layers==1:
        b1 = tf.Variable(tf.truncated_normal([params.layer_size], stddev=params.init_stddev, dtype=tf.float64))
        W2 = tf.Variable(tf.truncated_normal([params.layer_size, params.out_size], stddev=params.init_stddev, dtype=tf.float64))
        b2 = tf.Variable(tf.truncated_normal([params.out_size], stddev=params.init_stddev, dtype=tf.float64))
        xW = tf.matmul(x, W1)

        h = tf.nn.relu(xW + b1)
        prod = tf.matmul(h, W2)
        y = tf.add(prod, b2,name='forward')
        return y
    else:
        print('Not supported: ', params.num_layers)
        assert 0
