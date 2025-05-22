import numpy as np
from scipy.optimize import minimize
import time

# Tentar importar psutil (opcional)
try:
    import psutil
    psutil_disponivel = True
except ImportError:
    print("Aviso: psutil não está instalado. Uso de CPU/memória não será medido.")
    psutil_disponivel = False

# Início das medições
tempo_inicio = time.time()
if psutil_disponivel:
    processo = psutil.Process()
    cpu_inicio = psutil.cpu_percent(interval=None)
    mem_inicio = processo.memory_info().rss / 1024 / 1024  # MB

# Distâncias observadas entre pares de pontos
medidas_distancias = {
    (1, 2): 104.4226,
    (1, 3): 141.4264,
    (1, 4): 100.0186,
    (1, 5): 206.1519,
    (1, 6): 70.6993,
    (2, 3): 130.0119,
    (2, 4): 164.0010,
    (2, 5): 128.0688,
    (2, 6): 169.9940,
    (3, 4): 100.0009,
    (3, 5): 111.7834,
    (3, 6): 158.1090,
    (4, 5): 206.1490,
    (4, 6): 70.6874,
    (5, 6): 250.0094,
}

# Coordenadas conhecidas do ponto fixo P1
P1 = (100, 100)

# Chute inicial para os pontos P2 a P6
chute_inicial = np.array([
    200, 65, 200, 200, 100, 200, 300, 150, 50, 150
])

# Função de distância euclidiana
def calc_dist(p1, p2):
    return np.hypot(p2[0] - p1[0], p2[1] - p1[1])

# Função de cálculo de resíduos (vetor)
def calcular_residuos(parametros):
    pontos = {
        1: P1,
        2: (parametros[0], parametros[1]),
        3: (parametros[2], parametros[3]),
        4: (parametros[4], parametros[5]),
        5: (parametros[6], parametros[7]),
        6: (parametros[8], parametros[9]),
    }

    residuos = []
    for (i, j), d_obs in medidas_distancias.items():
        d_calc = calc_dist(pontos[i], pontos[j])
        residuos.append(d_calc - d_obs)
    return np.array(residuos)

# Função objetivo escalar (para minimize)
def func_objetivo(params):
    r = calcular_residuos(params)
    return np.sum(r**2)

# Otimização com método trust-constr
resultado = minimize(func_objetivo, chute_inicial, method='trust-constr', options={'verbose': 1})

# Fim das medições
tempo_fim = time.time()
tempo_execucao = tempo_fim - tempo_inicio

if psutil_disponivel:
    cpu_fim = psutil.cpu_percent(interval=None)
    mem_fim = processo.memory_info().rss / 1024 / 1024
    uso_memoria = mem_fim - mem_inicio

# Parâmetros ajustados
ajustado = resultado.x
coordenadas = {
    'P1': P1,
    'P2': (ajustado[0], ajustado[1]),
    'P3': (ajustado[2], ajustado[3]),
    'P4': (ajustado[4], ajustado[5]),
    'P5': (ajustado[6], ajustado[7]),
    'P6': (ajustado[8], ajustado[9]),
}

# Impressão dos resultados
print("\nCoordenadas estimadas:")
for nome, (x, y) in coordenadas.items():
    print(f"{nome}: ({x:.4f}, {y:.4f})")

# Resíduos finais
res_final = calcular_residuos(ajustado)
print("\nResíduos finais (diferença entre calculado e medido):")
for i, ((de, para), d_obs) in enumerate(medidas_distancias.items()):
    print(f"De {de} para {para}: {res_final[i]:.4f} m")

# Fator de variância a posteriori
n_obs = len(medidas_distancias)
n_par = len(ajustado)
gl = n_obs - n_par
s0_ap_2 = np.sum(res_final**2) / gl
s0_ap = np.sqrt(s0_ap_2)
print(f"\nFator de variância a posteriori: {s0_ap_2:.6f}")
print(f"Desvio padrão a posteriori: {s0_ap:.6f} m")

# Relatório de desempenho
print(f"\n⏱ Tempo de execução: {tempo_execucao:.4f} segundos")
if psutil_disponivel:
    print(f"🧠 Uso de memória (MB): {uso_memoria:.4f}")
    print(f"⚙️  CPU (%): {cpu_fim}")
