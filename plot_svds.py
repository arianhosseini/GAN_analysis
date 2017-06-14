import numpy as np
import matplotlib.pyplot as plt

def plot_svd(iteration, path = 'res/svds/'):

    fig = plt.figure(figsize=np.array([12, 8]))
    axes = [fig.add_subplot(6, 1, 1),
            fig.add_subplot(6, 1, 2),
            fig.add_subplot(6, 1, 3),
            fig.add_subplot(6, 1, 4),
            fig.add_subplot(6, 1, 5),
            fig.add_subplot(6, 1, 6)]

    svds = np.load(path + 'svds.npy').tolist()
    names = ['G_w_01', 'G_w_12', 'G_w_23',
             'D_w_01', 'D_w_12', 'D_w_23']

    for i, name in enumerate(names):
        ax = axes[i]
        ax.set_title(name)
        to_show = np.asarray(svds[i])
        to_show = to_show[:400]
        if to_show.shape[1] > 10:
            to_show = to_show[:, :10]
        # import ipdb; ipdb.set_trace()
        ax.imshow(to_show.T,
                  aspect='auto',
                  interpolation='nearest',
                  cmap=plt.cm.viridis)
        ax.tick_params(labelsize=0.05, direction='out')
        ax.set_xticks(range(to_show.shape[0]))
        ax.set_yticks(range(to_show.shape[1]))
        ax.grid(color='gray', linestyle='-', linewidth=0.1)

    plt.savefig(path + 'svds_'+str(iteration)+'.png', dpi=500)
    # import ipdb; ipdb.set_trace()

if __name__ == '__main__':
    plot_svd(0)
