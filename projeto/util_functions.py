from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import random

def plotHoldOutResults(poly_orders, mse_train_vec, mse_val_vec):
    # Plot results.
    min_train = min(mse_train_vec)
    min_val = min(mse_val_vec)
    min_order_train = np.argwhere(mse_train_vec == min_train)[0][0]+1
    min_order_val = np.argwhere(mse_val_vec == min_val)[0][0]+1
    min_order = min([min_order_train, min_order_val])
    min_value = min([min_train,min_val])
    
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
    ax1.set_xlim(min_order-4, poly_orders[len(poly_orders)-1])
    ylimlower = min_value-(min_value/10)
    if(ylimlower < 0): 
        ylimlower = 0
    ax1.set_ylim(ylimlower, min_value+(min_value/5))
    ax1.set_xticks(range(min_order-4, poly_orders[len(poly_orders)-1]+1,2))
    ax1.grid()

    #Show the plot.
    plt.show()

    print('Holdout - Min index train MSE:', min_order_train)
    print('Holdout - Min index val. MSE:', min_order_val)      
    
def plotKFoldResults(poly_orders, kfold_mean_vec, kfold_std_vec):
    # Plot results.
    min_mse = min(kfold_mean_vec)
    min_std = min(kfold_std_vec)
    min_order_mse = np.argwhere(kfold_mean_vec == min_mse)[0][0]+1
    min_order_std = np.argwhere(kfold_std_vec == min_std)[0][0]+1
    min_order = min([min_order_mse, min_order_std])
    
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
    ax1.set_xlim(min_order-4, poly_orders[len(poly_orders)-1])
    ylimlower = min_mse-(min_mse/10)
    if(ylimlower < 0): 
        ylimlower = 0
    ax1.set_ylim(ylimlower, min_mse+(min_mse/5))
    ax1.set_xticks(range(min_order-4, poly_orders[len(poly_orders)-1]+1,2))
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
    ax2.set_xlim(min_order-4, poly_orders[len(poly_orders)-1])
    ylimlower = min_std-(min_std/10)
    if(ylimlower < 0): 
        ylimlower = 0
    ax2.set_ylim(ylimlower, min_std+(min_std/5))
    ax2.set_xticks(range(min_order-4, poly_orders[len(poly_orders)-1]+1,2))
    ax2.grid()

    #Show the plot.
    plt.show()
    
    print('kFold - Min index MSE:', np.argwhere(kfold_mean_vec == min(kfold_mean_vec))[0][0]+1)
    print('kFold - Min index STD:', np.argwhere(kfold_std_vec == min(kfold_std_vec))[0][0]+1)    
    
def plotLeavePOutResults(poly_orders, lpo_mean_vec, lpo_std_vec):
    # Plot results.
    min_mse = min(lpo_mean_vec)
    min_std = min(lpo_std_vec)
    min_order_mse = np.argwhere(lpo_mean_vec == min_mse)[0][0]+1
    min_order_std = np.argwhere(lpo_std_vec == min_std)[0][0]+1
    min_order = min([min_order_mse, min_order_std])    
    
    fig = plt.figure(figsize=(15, 5))
    ax = plt.subplot(1, 2, 1)
    plt.plot(poly_orders, lpo_mean_vec,  label='Erro quadrático médio')
    plt.xlabel('Ordem do polinômio', fontsize=14)
    plt.ylabel('Média do erro quadrático médio', fontsize=14)
    plt.yscale('log')
    plt.xticks(range(0, max(poly_orders)+1, 2))
    plt.xlim([1, max(poly_orders)])
    plt.grid()
    plt.title('leave-p-out')

    left, bottom, width, height = [0.31, 0.6, 0.15, 0.2]
    ax1 = fig.add_axes([left, bottom, width, height])
    ax1.plot(poly_orders, lpo_mean_vec)
    ax1.set_xlim(min_order-4, poly_orders[len(poly_orders)-1])
    ylimlower = min_mse-(min_mse/10)
    if(ylimlower < 0): 
        ylimlower = 0
    ax1.set_ylim(ylimlower, min_mse+(min_mse/5))
    ax1.set_xticks(range(min_order-4, poly_orders[len(poly_orders)-1]+1,2))
    ax1.grid()

    ax = plt.subplot(1, 2, 2)
    plt.plot(poly_orders, lpo_std_vec,  label='Desvio padrão do erro')
    plt.xlabel('Ordem do polinômio', fontsize=14)
    plt.ylabel('Desvio padrão do erro quadrático médio', fontsize=14)
    plt.yscale('log')
    plt.xticks(range(0, max(poly_orders)+1, 2))
    plt.xlim([1, max(poly_orders)])
    plt.grid()
    plt.title('leave-p-out')

    left, bottom, width, height = [0.73, 0.6, 0.15, 0.2]
    ax2 = fig.add_axes([left, bottom, width, height])
    ax2.plot(poly_orders, lpo_std_vec)
    ax2.set_xlim(min_order-4, poly_orders[len(poly_orders)-1])
    ylimlower = min_std-(min_std/10)
    if(ylimlower < 0): 
        ylimlower = 0
    ax2.set_ylim(ylimlower, min_std+(min_std/5))
    ax2.set_xticks(range(min_order-4, poly_orders[len(poly_orders)-1]+1,2))
    ax2.grid()

    #Show the plot.
    plt.show()
    
    print('LPO - Min index MSE:', np.argwhere(lpo_mean_vec == min(lpo_mean_vec))[0][0]+1)
    print('LPO - Min index STD:', np.argwhere(lpo_std_vec == min(lpo_std_vec))[0][0]+1)     

def generateDataSet(groupNumber, N, debug=False):
    
    if(groupNumber <= 0 or groupNumber > 15):
        print('Erro: Número do grupo deve ser maior do que 0 e menor ou igual a 15!!!!')
        return None, None, None
    
    listofSeeds = [11, 1, 18, 42, 20, 3, 4, 13, 14, 17, 19, 28, 31, 34, 67]
    
    np.random.seed(listofSeeds[groupNumber-1])
    
    # Draw the number of terms.
    numOfTerms = np.random.randint(3, 7)
       
    # Define weights.
    a0 = np.random.randn()
    a1 = np.random.randn()
    a2 = np.random.randn()
    a3 = np.random.randn()
    a4 = np.random.randn()    
    a5 = np.random.randn()    
    b = np.random.randn()
    c = np.random.randn()
    
    # Define frequencies.
    f0 = np.random.randint(1, 4)
    f1 = np.random.randint(1, 4)
    
    # Define attribute.
    x = np.linspace(0, 1, N).reshape(N, 1)

    y = a0 + a1*x**2
    
    if(numOfTerms >= 3):
        y += a2*np.cos(2*np.pi*f0*x)
    
    if(numOfTerms >= 4):
        y += a3*np.sin(2*np.pi*f1*x)
    
    if(numOfTerms >= 5):
        y += a4*np.exp(b*x)
        
    if(numOfTerms >= 6):
        y += a5*np.tan(c*x)        
        
    maxy = max(y)[0]
    miny = min(y)[0]
    rangee = maxy - miny
        
    # Noise.
    w = np.sqrt(rangee/70)*np.random.randn(N,1)
    
    y_noisy = y + w
    
    if(debug==True): 
        print('Número de termos:', numOfTerms)
        print('a0:',a0)
        print('a1:',a1)
        print('a2:',a2)
        print('a2:',a2)
        print('a4:',a4)
        print('a5:',a5)
        print('b:',b)
        print('c:',c)
        print('f0:',f0)
        print('f1:',f1)
        print('max:',maxy)
        print('min:',miny)
        print('range:',rangee)    
    
    return x, y, y_noisy

def generateDatasetsv2(groupNumber, N):
    np.random.seed(groupNumber)
    random.seed(groupNumber)
    
    # Generate random wights.
    a0 = 20*np.random.rand() - 10
    a1 = 2*np.random.rand() - 1
    a2 = 2*np.random.rand() - 1
    a3 = 2*np.random.rand() - 1
    a4 = 2*np.random.rand() - 1
    
    # Generate degrees.
    degrees = random.sample(range(1, 8), 4)
    
    # Generate train dataset.
    x = np.sort(2*np.random.rand(N, 1) - 1.0, axis=0)
    y = a0 + a1*(x**degrees[0]) + a2*(x**degrees[1]) + a3*(x**degrees[2]) + a4*(x**degrees[3])
    
    noise_var = (max(y) - min(y))/50.0
    
    w = np.sqrt(noise_var)*np.random.randn(N,1)
    y_noisy = y + w
    
    return x, y, y_noisy

def generateDatasetsv3(groupNumber, N, mult1=13, mult2=22):
    np.random.seed(groupNumber*mult1)
    random.seed(groupNumber*mult2)
    
    # Generate random wights.
    a0 = 20*np.random.rand() - 10
    a1 = 2*np.random.rand() - 1
    a2 = 2*np.random.rand() - 1
    a3 = 2*np.random.rand() - 1
    a4 = 2*np.random.rand() - 1
    
    # Generate degrees.
    degrees = random.sample(range(1, 8), 4)
    
    # Generate train dataset.
    x = np.sort(2*np.random.rand(N, 1) - 1.0, axis=0)
    y = a0 + a1*(x**degrees[0]) + a2*(x**degrees[1]) + a3*(x**degrees[2]) + a4*(x**degrees[3])
    
    noise_var = (max(y) - min(y))/50.0
    
    w = np.sqrt(noise_var)*np.random.randn(N,1)
    y_noisy = y + w
    
    return x, y, y_noisy

def plotHoldOutResultsv2(poly_orders, mse_train_vec, mse_val_vec):
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

    #Show the plot.
    plt.show() 
    
def plotKFoldResultsv2(poly_orders, kfold_mean_vec, kfold_std_vec):
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

    ax = plt.subplot(1, 2, 2)
    plt.title('k-Fold')
    plt.plot(poly_orders, kfold_std_vec, label='Desvio padrão do erro')
    plt.yscale('log')
    plt.xlabel('Ordem do polinômio', fontsize=14)
    plt.ylabel('Desvio padrão do erro quadrático médio', fontsize=14)
    plt.xticks(range(0, max(poly_orders)+1, 2))
    plt.xlim([1, max(poly_orders)])
    plt.grid()

    #Show the plot.
    plt.show() 