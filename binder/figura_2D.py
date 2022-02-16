# Importando as bibliotecas necessárias.
import numpy as np
import matplotlib.pyplot as plt

# Reseta o gerador de sequências pseudo-aleatórias.
np.random.seed(42)

# Define o número de exemplos.
N = 1000

# Vetor coluna com dimensão Nx1, com valores linearmente espaçados entre -1 e 1.
x = np.linspace(-1, 1, N).reshape(N,1)

# Vetor ruído com dimensão Nx1 e variância igual a 0.1.
w = np.sqrt(0.1)*np.random.randn(N,1)

# Função original.
y = -1 + 2*x

# Versão ruidosa de y.
y_noisy = y + w

plt.plot(x, y_noisy, '.b', label='Função ruidosa')
plt.plot(x, y, 'k', label='Função original', linewidth=4)
plt.xlabel('x', fontsize=14)
plt.ylabel('y', fontsize=14)
plt.legend()
plt.grid()

# salva figura em arquivo
plt.savefig('figura_2D.png') 

# Mostra a figura.
plt.show()