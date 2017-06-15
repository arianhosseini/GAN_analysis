import numpy as np
import theano
import theano.tensor as T
from theano.compile import ViewOp
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from plot_svds import plot_svd
# np.random.seed(1991)

def theano_print(variable, name ,dev = True):
    if dev:
        return theano.printing.Print(name)(variable)
    else:
        return variable

cmap = LinearSegmentedColormap.from_list(
    'mycmap', [(0.0, '#f6b871'),
               (0.5, '#ffffff'),
               (1.0, '#61a7d3')])
batch_size = 30

w_01_u_reg_term = 0.1
w_01_v_reg_term = 0.0001
w_12_u_reg_term = 0.1
w_12_v_reg_term = 0.1
sv1_reg_term = 0.001
sv2_reg_term = 0.001
reg_term_6 = 0.01

decomposition_freq = 5

def get_data():
    N = 10 * batch_size
    D = 2
    K = 1
    X = np.zeros((N * K, D))
    y = np.zeros((N * K, K), dtype='uint8')
    for j in xrange(K):
        ix = range(N * j, N * (j + 1))
        r = np.linspace(0.0, 1, N)
        t = np.linspace(j * 4, (j + 1) * 4, N) + np.random.randn(N) * 0.2
        X[ix] = np.c_[r * np.sin(t), r * np.cos(t)]
        y[ix, j] = 1
    order = range(X.shape[0])
    np.random.shuffle(order)
    X = X[order]
    y = y[order]

    X = X.reshape((10, batch_size, 2))

    return X

def plot(real_, fake_, f_dis, name):
    fig = plt.figure(figsize=np.array([24, 6]))
    ax_0 = fig.add_subplot(1, 3, 1)
    ax_0.scatter(real_[:, 0], real_[:, 1], c='#f5972a', s=40)
    ax_0.set_xlim([-1, 1])
    ax_0.set_ylim([-1, 1])

    ax_1 = fig.add_subplot(1, 3, 2)
    x_0 = np.linspace(-1, 1, 100)
    y_0 = np.linspace(-1, 1, 100)
    X_0, Y_0 = np.meshgrid(x_0, y_0)
    Z = np.zeros(X_0.shape)
    for i in range(X_0.shape[0]):
        for j in range(Y_0.shape[1]):
            res = f_dis(np.array([[X_0[0, i], Y_0[j, 0]]]))
            Z[i, j] = res[0, 0]
    ax_1.pcolor(X_0, Y_0, Z.T, cmap=cmap)
    ax_1.scatter(real_[:, 0], real_[:, 1], c='#f5972a', s=10)
    ax_1.scatter(fake_[:, 0], fake_[:, 1], c='#61a7d3', s=10)
    ax_1.set_xlim([-1, 1])
    ax_1.set_ylim([-1, 1])

    ax_2 = fig.add_subplot(1, 3, 3)
    ax_2.scatter(fake_[:, 0], fake_[:, 1], c='#61a7d3', s=10)
    ax_2.set_xlim([-1, 1])
    ax_2.set_ylim([-1, 1])

    # plt.show()
    plt.savefig('res/' + name + '.png')
    plt.close()


def generator():
    noise = T.matrix('noise')
    w_01 = theano.shared(np.random.randn(150, 128) * 0.01, 'w_01')
    w_12 = theano.shared(np.random.randn(128, 128) * 0.01, 'w_12')
    w_23 = theano.shared(np.random.randn(128, 2) * 0.01, 'w_23')
    b_1 = theano.shared(np.zeros((128)), 'b_1')
    b_2 = theano.shared(np.zeros((128)), 'b_2')
    b_3 = theano.shared(np.zeros((2)), 'b_3')
    params = [w_01, b_1, w_12, b_2, w_23, b_3]

    l_1 = T.maximum(0, T.dot(noise, w_01) + b_1.dimshuffle('x', 0))
    l_2 = T.maximum(0, T.dot(l_1, w_12) + b_2.dimshuffle('x', 0))
    fake = T.tanh(T.dot(l_2, w_23) + b_3.dimshuffle('x', 0))

    return noise, fake, params

def discriminator(real, fake, x):
    w_01_u, w_01_s, w_01_v = get_svd_layer(2, 128, 'w_01')
    w_12_u, w_12_s, w_12_v = get_svd_layer(128, 1, 'w_12')

    b_1 = theano.shared(np.zeros((128)), 'b_1')
    b_2 = theano.shared(np.zeros((1)), 'b_2')

    params = [w_01_u, w_01_s, w_01_v, b_1, w_12_u, w_12_s, w_12_v, b_2]

    w_01 = T.dot(w_01_u, T.dot(w_01_s, w_01_v))
    w_12 = T.dot(w_12_u, T.dot(w_12_s, w_12_v))
    l_1_real = T.maximum(0, T.dot(real, w_01) + b_1.dimshuffle('x', 0))
    pred_real = T.nnet.sigmoid(T.dot(l_1_real, w_12) + b_2.dimshuffle('x', 0))

    l_1_fake = T.maximum(0, T.dot(fake, w_01) + b_1.dimshuffle('x', 0))
    pred_fake = T.nnet.sigmoid(T.dot(l_1_fake, w_12) + b_2.dimshuffle('x', 0))

    l_1_x = T.maximum(0, T.dot(x, w_01) + b_1.dimshuffle('x', 0))
    pred_x = T.nnet.sigmoid(T.dot(l_1_x, w_12) + b_2.dimshuffle('x', 0))

    return pred_real, pred_fake, pred_x, params

def gan():
    noise, fake, p_g = generator()
    real = T.matrix('real')
    x = T.matrix('x')

    pred_real, pred_fake, pred_x, p_d = discriminator(real, fake, x)

    d_cost = (
        T.mean(T.nnet.binary_crossentropy(
            pred_real, T.alloc(1, batch_size, 1))) +
        T.mean(T.nnet.binary_crossentropy(
            pred_fake, T.alloc(0, batch_size, 1))))

    # w_01_uTu = T.dot(p_d[0].T,p_d[0])
    # w_01_vTv = T.dot(p_d[2].T,p_d[2])
    # d_cost += theano_print(w_01_u_reg_term * (T.sum( (w_01_uTu - T.identity_like(w_01_uTu)) **2 )), 'w_01_u_ortho', True)
    # d_cost += theano_print(w_01_v_reg_term * (T.sum( (w_01_vTv - T.identity_like(w_01_vTv)) **2 )), 'w_01_v_ortho', True)
    #
    # w_12_uTu = T.dot(p_d[4].T,p_d[4])
    # w_12_vTv = T.dot(p_d[6].T,p_d[6])
    # d_cost += theano_print(w_12_u_reg_term * (T.sum((w_12_uTu - T.identity_like(w_12_uTu)) ** 2 )), 'w_12_u_ortho')
    # d_cost += theano_print(w_12_v_reg_term * (T.sum((w_12_vTv - T.identity_like(w_12_vTv)) ** 2 )), 'w_12_v_ortho')
    #
    # d_cost += theano_print((sv1_reg_term * T.sum((p_d[1] ** 2))), 'sv1')
    #
    # d_cost += theano_print((sv2_reg_term * T.sum((p_d[5]))), 'sv2')

    g_cost = T.mean(T.nnet.binary_crossentropy(
        pred_fake, T.alloc(1, batch_size, 1)))

    return noise, fake, real, p_g, p_d, g_cost, d_cost, x, pred_x


def get_svd_layer(in_dim, out_dim, name):
    u,s,v = np.linalg.svd(np.random.randn(in_dim, out_dim)* 0.01, full_matrices=0)
    w_u = theano.shared(u, name+'_u')
    w_s = theano.shared(np.diag(s), name+'_s')
    w_v = theano.shared(v, name+'_v')
    return w_u, w_s, w_v

def orthogonalize(p_d):
    w_01 = p_d[0].get_value().dot(p_d[1].get_value().dot(p_d[2].get_value()))
    u_hat, s_hat, v_hat = np.linalg.svd(w_01, full_matrices=0)
    p_d[0].set_value(u_hat)
    p_d[1].set_value(np.diag(s_hat))
    p_d[2].set_value(v_hat)

    w_12 = p_d[4].get_value().dot(p_d[5].get_value().dot(p_d[6].get_value()))
    u_hat, s_hat, v_hat = np.linalg.svd(w_12, full_matrices=0)
    p_d[4].set_value(u_hat)
    p_d[5].set_value(np.diag(s_hat))
    p_d[6].set_value(v_hat)

def test_gan():
    noise, fake, real, p_g, p_d, g_cost, d_cost, x, pred_x = gan()
    weight = T.matrix('weight')

    g_grads = T.grad(g_cost, p_g)
    g_updates = []
    for p, g in zip(p_g, g_grads):
        g_updates.append((p, p - 1 * g))

    d_grads = T.grad(d_cost, p_d)
    d_updates = []
    for p, g in zip(p_d, d_grads):
        d_updates.append((p, p - 1 * g))


    w_01_u, w_01_s, w_01_v = T.nlinalg.svd(T.dot(p_d[0],T.dot(p_d[1],p_d[2])), full_matrices=0)
    w_12_u, w_12_s, w_12_v = T.nlinalg.svd(T.dot(p_d[4],T.dot(p_d[5],p_d[6])), full_matrices=0)
    decomposition_updates = [(p_d[0], w_01_u),
                             (p_d[1], T.nlinalg.diag(w_01_s)),
                             (p_d[2], w_01_v),
                             (p_d[4], w_12_u),
                             (p_d[5], T.nlinalg.diag(w_12_s)),
                             (p_d[6], w_12_v)
                            ]

    u,s,v = T.nlinalg.svd(weight, full_matrices=0)

    f_eval = theano.function([noise], fake)
    f_dis = theano.function([x], pred_x)
    train_g = theano.function([noise], g_cost, updates=g_updates)
    train_d = theano.function([noise, real], [d_cost] + d_grads, updates=d_updates)
    decompose_d_weights = theano.function([],[], updates=decomposition_updates)
    f_svd = theano.function([weight],s)

    g_sv_w01 = []
    g_sv_w12 = []
    g_sv_w23 = []

    d_sv_w01 = []
    d_sv_w12 = []
    d_sv_w23 = []

    for epoch in range(1501):
        real_ = get_data()
        g_costs = []
        d_costs = []
        for i in range(real_.shape[0]):
            noise_ = (np.random.rand(batch_size, 150) - 0.5) * 10
            res = train_d(noise_, real_[i])
            d_costs += [res[0]]
            grads_ = res[1:]
            grads_norm = [np.sqrt(np.sum(np.array(g) ** 2)) for g in grads_]

            # print "d_cost: ", d_costs[-1]
            # print "grads norm: ", grads_norm, "\n\n"

            for j in range(5):
                g_costs += [train_g(noise_)]
            if (i + 1) % decomposition_freq == 0:
                orthogonalize(p_d)


        if epoch % 1 == 0:
            noise_ = (np.random.rand(1000, 150) - 0.5) * 10
            fake_ = f_eval(noise_)
            plot(real_.reshape((-1, 2)), fake_, f_dis, str(epoch))

            d_sv_w01.append(np.diag(p_d[1].get_value()))
            d_sv_w12.append(np.diag(p_d[5].get_value()))
            d_sv_w23.append(np.diag(p_d[5].get_value()))

            g_sv_w01.append(np.linalg.svd(p_g[0].get_value(), full_matrices=0)[1])
            g_sv_w12.append(np.linalg.svd(p_g[2].get_value(), full_matrices=0)[1])
            g_sv_w23.append(np.linalg.svd(p_g[4].get_value(), full_matrices=0)[1])


        if epoch % 10 == 0:
            print ('epoch: ' + str(epoch) + '\t'
                   'd_cost: ' + str(np.mean(np.array(d_costs))) + '\t'
                   'g_cost: ' + str(np.mean(np.array(g_costs))))

            np.save('res/svds/svds.npy',[g_sv_w01,
                                    g_sv_w12,
                                    g_sv_w23,
                                    d_sv_w01,
                                    d_sv_w12,
                                    d_sv_w23
                                    ])
            plot_svd(epoch)


test_gan()
