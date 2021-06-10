from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import random

def plotHoldOutResults(poly_orders, mse_train_vec, mse_val_vec):
    # Plot results.
    fig = plt.figure()
    plt.plot(poly_orders, mse_train_vec,  label='Treinamento')
    plt.plot(poly_orders, mse_val_vec,  label='Validação')
    plt.xlabel('Ordem do polinômio', fontsize=14)
    plt.ylabel('Erro quadrático Médio', fontsize=14)
    plt.xticks(range(0, max(poly_orders)+1, 2))
    plt.xlim([1, max(poly_orders)])
    plt.yscale('log')
    plt.title('Holdout')
    plt.legend()
    plt.grid()

    left, bottom, width, height = [0.57, 0.4, 0.3, 0.3]
    ax1 = fig.add_axes([left, bottom, width, height])
    ax1.plot(poly_orders, mse_train_vec,  label='Treinamento')
    ax1.plot(poly_orders, mse_val_vec,  label='Validação')
    ax1.set_xlim(16, 30)
    ax1.set_ylim(8, 9.5)
    ax1.set_xticks(range(16,31,2))
    ax1.grid()

    #Show the plot.
    plt.show()

def plotKFoldResults(poly_orders, kfold_mean_vec, kfold_std_vec):
    # Plot results.
    fig = plt.figure(figsize=(15, 5))
    ax = plt.subplot(1, 2, 1)
    plt.plot(poly_orders, kfold_mean_vec, label='Erro quadrático médio')
    plt.yscale('log')
    plt.xlabel('Ordem do polinômio', fontsize=14)
    plt.ylabel('Média do erro quadrático médio', fontsize=14)
    plt.xticks(range(0, max(poly_orders)+1, 2))
    plt.xlim([1, max(poly_orders)])
    plt.grid()
    plt.title('k-Fold')

    left, bottom, width, height = [0.31, 0.6, 0.15, 0.2]
    ax1 = fig.add_axes([left, bottom, width, height])
    ax1.plot(poly_orders, kfold_mean_vec)
    ax1.set_xlim(16, 30)
    ax1.set_ylim(8.6, 9.2)
    ax1.set_xticks(range(16,31,2))
    ax1.grid()

    ax = plt.subplot(1, 2, 2)
    plt.title('k-Fold')
    plt.plot(poly_orders, kfold_std_vec, label='Desvio padrão do erro')
    plt.yscale('log')
    plt.xlabel('Ordem do polinômio', fontsize=14)
    plt.ylabel('Desvio padrão do erro quadrático médio', fontsize=14)
    plt.xticks(range(0, max(poly_orders)+1, 2))
    plt.xlim([1, max(poly_orders)])
    plt.grid()

    left, bottom, width, height = [0.73, 0.6, 0.15, 0.2]
    ax2 = fig.add_axes([left, bottom, width, height])
    ax2.plot(poly_orders, kfold_std_vec)
    ax2.set_xlim(16, 30)
    ax2.set_ylim(0.2, 0.5)
    ax2.set_xticks(range(16,31,2))
    ax2.grid()

    #Show the plot.
    plt.show()