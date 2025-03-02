import numpy as np
from scipy.optimize import minimize
import time  # Importando a biblioteca time
import tracemalloc  # Importando o módulo tracemalloc para rastrear o uso de memória

# Definindo as medições observadas de diferenças de altura
d1 = 39.767
d2 = 25.372
d3 = -5.761
d4 = 9.981
d5 = -15.754
d6 = 31.130
d7 = -12.829
d8 = -15.771
d9 = 34.082
d10 = 52.316

# Matriz A que relaciona as altitudes dos pontos (coeficientes das equações de diferença de altura)
A = np.array([[1, 0, 0, 0, 0, 0],
              [-1, 1, 0, 0, 0, 0],
              [-1, 0, 1, 0, 0, 0],
              [-1, 0, 0, 1, 0, 0],    
              [0, 0, 1, -1, 0, 0],
              [0, 1, -1, 0, 0, 0],
              [0, -1, 0, 0, 1, 0],
              [0, 0, -1, 0, 0, 1],
              [0, 0, 0, 0, 1, -1],
              [0, 0, 0, 0, 1, 0]])   

# Vetor b com as observações
b = np.array([d1, d2, d3, d4, d5, d6, d7, d8, d9, d10])

# Função objetivo a ser minimizada (soma dos quadrados dos resíduos)
def objective(x):
    return np.sum((np.dot(A, x) - b) ** 2)

# Função que resolve o problema de otimização usando o método interior-point
def ajuste_rede_nivelamento():
    # Definindo um ponto inicial para as altitudes (arbitrário)
    x0 = np.array([0, 0, 0, 0, 0, 0])

    # Inicia o rastreamento de memória
    tracemalloc.start()

    # Marcando o tempo de início
    start_time = time.time()

    resultado = minimize(objective, x0, method='trust-constr')

    # Marcando o tempo de término
    end_time = time.time()

    # Calculando o tempo de execução
    execution_time = end_time - start_time
    print(f"Tempo de execução: {execution_time:.4f} segundos")

    # Obtendo o uso de memória atual e o pico de memória
    current, peak = tracemalloc.get_traced_memory()
    print(f"Uso de memória: {current / (1024):.2f} KB; Pico de memória: {peak / (1024):.2f} KB")

    # Finaliza o rastreamento de memória
    tracemalloc.stop()

    return resultado.x

# Realizando o ajuste da rede de nivelamento
altitudes_ajustadas = ajuste_rede_nivelamento()

# Exibindo os resultados
print("Altitudes ajustadas: P1 = {:.4f}, P2 = {:.4f}, P3 = {:.4f}, P4 = {:.4f}, P5 = {:.4f}, P6 = {:.4f}".format(
    altitudes_ajustadas[0], altitudes_ajustadas[1], altitudes_ajustadas[2], altitudes_ajustadas[3], 
    altitudes_ajustadas[4], altitudes_ajustadas[5]
))
