import numpy as np
import matplotlib.pyplot as plt
import random

# Plot performance results.
def plotResults(Jgd, J, A1, A2, a_opt, a_hist, iteration, maxX, maxY):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10,5))

    cp = ax1.contour(A1, A2, J)
    ax1.clabel(cp, inline=1, fontsize=10)
    ax1.set_xlabel('$a_1$', fontsize=14)
    ax1.set_ylabel('$a_2$', fontsize=14)
    ax1.set_title('Cost-function\'s Contour')
    ax1.plot(a_opt[0], a_opt[1], c='r', marker='*', markersize=14)
    ax1.plot(a_hist[0, 0:iteration], a_hist[1, 0:iteration], 'k--')
    ax1.plot(a_hist[0, 0:iteration], a_hist[1, 0:iteration], 'kx')
    ax1.set_xticks(np.arange(-20, 24, step=4.0))
    ax1.set_yticks(np.arange(-20, 24, step=4.0))

    ax2.plot(np.arange(0, iteration), Jgd[0:iteration])
    ax2.set_yscale('log')
    ax2.set_xlabel('Iteration', fontsize=14)
    ax2.set_ylabel('$J_e$', fontsize=14)
    ax2.set_title('Error vs. Epoch number')
    ax2.set_xlim((0, iteration-1))

    left, bottom, width, height = [0.65, 0.5, 0.23, 0.3]
    ax3 = fig.add_axes([left, bottom, width, height])
    ax3.plot(np.arange(0, maxX), Jgd[0:maxX])
    ax3.set_ylim(0, maxY)

    plt.show()
    
# Plot performance results.
def plotResults2(Jgd, J, A1, A2, a_opt, a_hist, alpha_hist, iteration, maxX, maxY):
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15,5))

    cp = ax1.contour(A1, A2, J)
    ax1.clabel(cp, inline=1, fontsize=10)
    ax1.set_xlabel('$a_1$', fontsize=14)
    ax1.set_ylabel('$a_2$', fontsize=14)
    ax1.set_title('Cost-function\'s Contour')
    ax1.plot(a_opt[0], a_opt[1], c='r', marker='*', markersize=14)
    ax1.plot(a_hist[0, 0:iteration], a_hist[1, 0:iteration], 'k--')
    ax1.plot(a_hist[0, 0:iteration], a_hist[1, 0:iteration], 'kx')
    ax1.set_xticks(np.arange(-20, 24, step=4.0))
    ax1.set_yticks(np.arange(-20, 24, step=4.0))

    ax2.plot(np.arange(0, iteration), Jgd[0:iteration])
    ax2.set_yscale('log')
    ax2.set_xlabel('Iteration', fontsize=14)
    ax2.set_ylabel('$J_e$', fontsize=14)
    ax2.set_title('Error vs. Epoch number')
    ax2.set_xlim((0, iteration-1))

    left, bottom, width, height = [0.45, 0.5, 0.16, 0.3]
    ax4 = fig.add_axes([left, bottom, width, height])
    ax4.plot(np.arange(0, maxX), Jgd[0:maxX])
    ax4.set_ylim(0, maxY)
    
    ax3.plot(np.arange(0, iteration), alpha_hist[0:iteration])
    ax3.set_xlabel('Iteration', fontsize=14)
    ax3.set_ylabel('$\\alpha$', fontsize=14)
    ax3.set_title('Alpha variation')
    ax3.set_xlim((0, iteration-1))
    ax3.set_ylim((0.09, 0.11))
    ax3.set_yticks([0.09, 0.1, 0.11])

    plt.show()
    
# Plot performance results.
def plotResults3(Jgd, J, A1, A2, a_opt, a_hist, alpha_hist, iteration, maxX, maxY):
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(17,5))

    cp = ax1.contour(A1, A2, J)
    ax1.clabel(cp, inline=1, fontsize=10)
    ax1.set_xlabel('$a_1$', fontsize=14)
    ax1.set_ylabel('$a_2$', fontsize=14)
    ax1.set_title('Cost-function\'s Contour')
    ax1.plot(a_opt[0], a_opt[1], c='r', marker='*', markersize=14)
    ax1.plot(a_hist[0, 0:iteration], a_hist[1, 0:iteration], 'k--')
    ax1.plot(a_hist[0, 0:iteration], a_hist[1, 0:iteration], 'kx')
    ax1.set_xticks(np.arange(-20, 24, step=4.0))
    ax1.set_yticks(np.arange(-20, 24, step=4.0))

    ax2.plot(np.arange(0, iteration), Jgd[0:iteration])
    ax2.set_yscale('log')
    ax2.set_xlabel('Iteration', fontsize=14)
    ax2.set_ylabel('$J_e$', fontsize=14)
    ax2.set_title('Error vs. Epoch number')
    ax2.set_xlim((0, iteration-1))

    left, bottom, width, height = [0.45, 0.5, 0.16, 0.3]
    ax4 = fig.add_axes([left, bottom, width, height])
    ax4.plot(np.arange(0, maxX), Jgd[0:maxX])
    ax4.set_ylim(0, maxY)
    
    ax3.plot(np.arange(0, iteration), alpha_hist[0:iteration])
    ax3.set_yscale('log')
    ax3.set_xlabel('Iteration', fontsize=14)
    ax3.set_ylabel('$\\alpha$', fontsize=14)
    ax3.set_title('Alpha variation')
    ax3.set_xlim((0, iteration-1))
    
    left, bottom, width, height = [0.705, 0.2, 0.08, 0.3]
    ax5 = fig.add_axes([left, bottom, width, height])
    ax5.plot(np.arange(0, iteration), alpha_hist[0:iteration])
    ax5.set_xlim((0, 31))
    ax5.set_yticks([0, 0.025, 0.05, 0.1])
    ax5.set_xticks([0, 14, 29])
    ax5.grid()

    plt.show()