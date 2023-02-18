import math, time
import numpy as np
import matplotlib.pyplot as plt

from plot_3d import report_plot

def run() :
    """
    Do a single realization of the simulation and return results as list of (t, n).
    """
    t = t0
    n = n0
    res = []
    while t < t_max :
        if n == 0 :     # If in absorbing state, do nothing
            pass
        elif n == N :   # Restrict n to (0, N)
            dt = - math.log(np.random.random_sample()) / Wm(n)
            n -= 1
        else :
            tp = - math.log(np.random.random_sample()) / Wp(n)
            tm = - math.log(np.random.random_sample()) / Wm(n)
            if tp < tm :    # Increase (infection) 
                dt = tp
                n += 1
            else :          # Decrease (recovery)
                dt = tm
                n -= 1

        # Increment time and record results
        t = t + dt
        res.append((t,n))
    
    return res
    
def get_binned(res, K=100) :
    """
    Reduce a series of 'continuous-time' results into discrete time by a weighted average into K bins.
    """
    t_max = res[-1][0]      # Bin until the end of simulation results
    h = t_max / K           # Step size

    x = np.zeros(K+2)       # Aggregate vector for first moment
    xm2 = np.zeros(K+2)     # Aggregate vector for second moment

    t0 = res[0][0]
    for t, n in res[1:] :                   # Go through each result
        t1 = t
        if t1 > t_max : t1 = t_max          # Restrict to t_max
        k0 = int(t0/h) + 1                  # Bin of last time step
        k1 = int(t1/h) + 1                  # Bin of current time step

        if k1 == k0 :                       # If times are in the same bin
            x[k0] += n * (t1-t0)            # Average over time in bin
            xm2[k0] += n**2 * (t1-t0)
        else :
            x[k0] += n * (k0*h-t0)          # Average over time in first bin
            xm2[k0] += n**2 * (k0*h-t0)
            for i in range(k1-k0-1) :       # Average over time in intermediate bins
                x[k0+i+1] += n * h
                xm2[k0+i+1] += n**2 * h
            x[k1] += n * (t1-h*(k1-1))      # Average over time in last bin
            xm2[k1] += n**2 * (t1-h*(k1-1))
        
        t0 = t1

    x = x[1:-1]     # Harmless hack due to implementation bug
    x /= h
    xm2 = xm2[1:-1]
    xm2 /= h

    return x, xm2

def get_pnt(res, N, K=100) :
    """
    Binning algorithms similar to the one in `get_binned()`, but for the full probability distribution p(n;t).
    """
    t_max = res[-1][0]
    h = t_max / K

    x = np.zeros((N+1, K+2))

    t0 = res[0][0]
    for t, n in res[1:] :
        t1 = t
        if t1 > t_max : t1 = t_max
        k0 = int(t0/h) + 1
        k1 = int(t1/h) + 1

        if k1 == k0 :
            x[n, k0] += (t1-t0)
        else :
            x[n, k0] += (k0*h-t0)
            for i in range(k1-k0-1) :
                x[n, k0+i+1] += h
            x[n, k1] += (t1-h*(k1-1))
        
        t0 = t1

    x = x[:, 1:-1]     # Harmless hack due to implementation bug
    x /= h

    return x

### Simulation parameters
t0, t_max = 0, 200
N = 1000
# n0 = np.random.randint(1, N+1)
n0 = N//20  # 5% infected

### Rates
beta = 1
gamma = 1

print(f"R0 = beta/gamma = {beta/gamma}")

Wp = lambda n : beta * n * (N-n) / N
Wm = lambda n : gamma * n

### Trajectory averaging parameters
M = 200     # Number of realizations for average
K = 1000     # Number of bins for averaging

### Run and take measurements over M trajectories
tic = time.process_time()
res_agg = np.zeros(K)
res_agg_m2 = np.zeros(K)
res_agg_pnt = np.zeros((N+1, K))
bin_err = np.zeros(K)
for im in range(M) :
    res = run()
    avg, avg_sq = get_binned(res, K)
    bin_err += avg_sq - avg**2
    res_agg += avg
    res_agg_m2 += avg**2
    res_agg_pnt += get_pnt(res, N, K)
toc = time.process_time()
print(f"Runtime (process): {toc-tic:.2f} s")

bin_err = np.sqrt(bin_err / M)
res_mean = res_agg / M
res_std = np.sqrt(res_agg_m2 / M - res_mean**2)
res_pnt_norm = res_agg_pnt / M

### Plot
fname = 'sis_fr.pdf'
fig = report_plot(res_mean, res_std, res_pnt_norm, N, t_max, K, bin_err=bin_err)
fig.suptitle("Basic model")
plt.savefig(fname, dpi=300, bbox_inches='tight')
plt.show()

### Save results
with open('res_sis_fr.txt', 'w') as f :
    np.savetxt(f, np.array((res_mean, res_std, bin_err)))
with open('res_sis_fr_pnt.txt', 'w') as f :
    np.savetxt(f, res_pnt_norm)