import numpy as np
from scipy.optimize import minimize
import time  # Importando a biblioteca time
import tracemalloc  # Importando o módulo tracemalloc para rastrear o uso de memória
import matplotlib.pyplot as plt  # Importando a biblioteca para plotar gráficos

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

    # Realizando a minimização com o método trust-constr
    resultado = minimize(objective, x0, method='trust-constr')

    # Marcando o tempo de término
    end_time = time.time()

    # Calculando o tempo de execução
    execution_time = end_time - start_time
    print(f"Tempo de execução: {execution_time:.4f} segundos")

    # Obter o consumo total e o pico de memória
    current, peak = tracemalloc.get_traced_memory()
    print(f"Consumo total de memória: {current / 1024:.2f} KB")
    print(f"Pico de memória: {peak / 1024:.2f} KB")

    # Finaliza o rastreamento de memória
    tracemalloc.stop()

    return resultado.x, current, peak

# Realizando o ajuste da rede de nivelamento
altitudes_ajustadas, total_memory, peak_memory = ajuste_rede_nivelamento()

# Exibindo os resultados
print("Altitudes ajustadas: P1 = {:.4f}, P2 = {:.4f}, P3 = {:.4f}, P4 = {:.4f}, P5 = {:.4f}, P6 = {:.4f}".format(
    altitudes_ajustadas[0], altitudes_ajustadas[1], altitudes_ajustadas[2], altitudes_ajustadas[3], 
    altitudes_ajustadas[4], altitudes_ajustadas[5]
))

# Gráfico do consumo de memória ao longo do tempo
times = np.linspace(0, 1, 100)  # Simulando o tempo de execução para plotar
memory_usage = np.linspace(0, total_memory, 100)  # Simulando o uso de memória

plt.figure(figsize=(10, 6))
plt.plot(times, memory_usage, label='Uso de memória (KB)')
plt.xlabel('Tempo (segundos)')
plt.ylabel('Uso de memória (KB)')
plt.title('Uso de memória durante a execução da otimização')
plt.grid(True)
plt.legend()
plt.show()
