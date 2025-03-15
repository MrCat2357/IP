#Solução Pela Matriz Inversa
import numpy as np
import time
def normal_equation_solve(A, B):
# Calculando A^T A
    AtA = np.dot(A.T, A)
# Calculando A^T B
    AtB = np.dot(A.T, B)
# Calculando a inversa de A^T A
    AtA_inv = np.linalg.inv(AtA)
# Resolvendo o sistema x = (A^T A)^-1 * (A^T B)
    x = np.dot(AtA_inv, AtB)
    return x
# Matriz A (geralmente não precisa ser simétrica)
A = np.array([[1, 1, 0],
[1, 2, -1],
[0, -1, 3]])
# Vetor de constantes
B = np.array([1, 1, 2])
# Captura do tempo inicial e uso da CPU
start_time = time.time()
# Resolvendo o sistema usando as equações normais
solution = normal_equation_solve(A, B)
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