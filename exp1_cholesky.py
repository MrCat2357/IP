#Resolução Por Cholesky
import numpy as np
import time
def cholesky_solve(A, B):
# Realizando a decomposição de Cholesky
    L = np.linalg.cholesky(A)
# Resolvendo o sistema Ly = B (substituição para frente)
    y = np.linalg.solve(L, B)
# Resolvendo o sistema L.T x = y (substituição para trás)
    x = np.linalg.solve(L.T, y)
    return x
# Matriz simétrica positiva definida (necessária para Cholesky)
A = np.array([[1, 1, 0],
[1, 2, -1],
[0, -1, 3]])
# Vetor de constantes
B = np.array([1, 1, 2])
# Captura do tempo inicial e uso da CPU
start_time = time.time()
# Resolvendo o sistema usando o método de Cholesky
solution = cholesky_solve(A, B)
# Captura do tempo final e uso da CPU
end_time = time.time()
# Cálculo do tempo de execução e uso de CPU
execution_time = end_time - start_time
# Exibindo o resultado e métricas de desempenho
print("Soluções do sistema:")
print(f"x = {solution[0]}")
print(f"y = {solution[1]}")
print(f"z = {solution[2]}")
print(f"\nTempo de execução: {execution_time:.6f} segundos")