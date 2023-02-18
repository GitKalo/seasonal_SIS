import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np

sns.set_theme(style='white', font_scale=1.2)

def report_plot(res_mean, res_std, res_pnt, N, t_max, K, bin_err=None) :
    fig = plt.figure(figsize=(8, 6), layout='constrained')

    gs = fig.add_gridspec(1, 2,  width_ratios=(4, 1),
                        left=0.1, right=0.9, bottom=0.1, top=0.9,
                        wspace=0.1, hspace=0.05)

    ax = fig.add_subplot(gs[0, 0])
    
    if bin_err is None :
        bin_err = np.zeros_like(res_std)
    x, y, err_m, err_k = np.arange(K)*(t_max/K), res_mean, res_std, bin_err
    
    ax.plot(x, y)
    ax.fill_between(x, y-err_m, y+err_m, alpha=0.2)
    ax.fill_between(x, y+err_m, y+err_m+err_k, alpha=0.2, color='green')
    ax.fill_between(x, y-err_m, y-err_m-err_k, alpha=0.2, color='green')
    
    ax.set_xlabel("$t$")
    ax.set_xlim(0,t_max)
    ax.set_ylabel("$n(t)$")
    ax.set_ylim(0,N)

    ax_y = fig.add_subplot(gs[0, 1], sharey=ax)
    # ax_y.set_title("PDF")
    ax_y.set_xlabel("$p(n;t=t_{max})$")
    ax_y.tick_params(axis="y", labelleft=False)
    ax_y.plot(res_pnt[:, -1], range(res_pnt[:, -1].size))

    return fig

if __name__ == '__main__' :
    res_pnt = np.loadtxt('res_sis_periodic_pnt_200.txt')
    res_pnt = res_pnt[:, 60:]

    ### Plot PDF over time in 3D
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    X, Y = np.meshgrid(np.arange(res_pnt.shape[1]), np.arange(res_pnt.shape[0]))
    surf = ax.plot_surface(X, Y, res_pnt, rcount=1000, ccount=1000, cmap=plt.cm.cividis, linewidth=0, antialiased=False)
    ax.view_init(elev=35, azim=-40)     # Rotate image
    ax.set_xlabel('$t$')
    ax.set_ylabel('$n$')
    ax.set_zlabel('$p(n;t)$')
    ax.set_xticklabels([str(int(t)//4 + 15) for t in ax.get_xticks()])
    fig.colorbar(surf, shrink=0.5, aspect=10, pad=0.2)
    fig.tight_layout()
    
    fname = 'sis_pnt_3d.pdf'
    plt.savefig(fname, dpi=300)
    print(f"Saved 3D plot to '{fname}'.")
    # plt.show()

    ### Plot contour plot of PDF over time
    fig = plt.figure(figsize=(5, 4))
    ax = plt.axes()
    X, Y = np.meshgrid(np.arange(res_pnt.shape[1]), np.arange(res_pnt.shape[0]))
    cs = ax.contourf(X, Y, res_pnt, cmap=plt.cm.cividis)
    ax.set_xlabel('$t$')
    ax.set_ylabel('$n$')
    ax.set_xticklabels([str(int(t)//4 + 15) for t in ax.get_xticks()])
    # fig.colorbar(cs, shrink=0.5, aspect=10, pad=0.1)
    fig.tight_layout()

    fname = 'sis_pnt_contour.pdf'
    plt.savefig(fname, dpi=300)
    print(f"Saved contour plot to '{fname}'.")
    # plt.show()

    ### Plot standard deviation over time
    fig = plt.figure()
    ax = plt.gca()
    # res_std = np.std(res_pnt, axis=0)
    res_std = np.loadtxt('res_sis_periodic_200.txt')[1, :]
    plt.plot(res_std)
    ax.set_xlabel('$t$')
    ax.set_ylabel('$\hat{\sigma}_M$')
    ax.set_xticklabels([str(int(t)//4) for t in ax.get_xticks()])
    fig.tight_layout()

    fname = 'sis_std_t.pdf'
    plt.savefig(fname, dpi=300)
    print(f"Saved std plot to '{fname}'.")
    plt.show()