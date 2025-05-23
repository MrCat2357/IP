import numpy as np
from scipy.optimize import minimize
import time
import pandas as pd

# Tentar importar psutil, com tratamento de erro
try:
    import psutil
    psutil_disponivel = True
    processo = psutil.Process()
    memoria_inicial = processo.memory_info().rss / 1024 / 1024  # em MB
except ImportError:
    print("Aviso: psutil não está instalado. Uso de CPU e memória não será medido.")
    psutil_disponivel = False
    memoria_inicial = 0

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

# Listas para armazenar resultados
coordenadas_resultados = []
tempos_execucao = []
usos_memoria = []

num_execucoes = 100

# Loop de execuções
for i in range(num_execucoes):
    tempo_inicio = time.time()

    resultado = minimize(func_objetivo, chute_inicial, method='trust-constr', options={'disp': False})
    
    tempo_fim = time.time()
    tempo_execucao = tempo_fim - tempo_inicio
    tempos_execucao.append(tempo_execucao)

    if psutil_disponivel:
        mem_uso = processo.memory_info().rss / 1024 / 1024
        usos_memoria.append(mem_uso - memoria_inicial)
    else:
        usos_memoria.append(np.nan)

    coordenadas_resultados.append(resultado.x)
#    time.sleep(1)  # opcional, para espaçar execuções

# Estatísticas
coordenadas_array = np.array(coordenadas_resultados)
tempos_array = np.array(tempos_execucao)
memoria_array = np.array(usos_memoria)

media_coord = np.mean(coordenadas_array, axis=0)
std_coord = np.std(coordenadas_array, axis=0)
media_tempo = np.mean(tempos_array)
std_tempo = np.std(tempos_array)
media_mem = np.mean(memoria_array)
std_mem = np.std(memoria_array)

# 📊 Tabela com os resultados de cada execução
df_resultados = pd.DataFrame({
    'Execução': np.arange(1, num_execucoes + 1),
    'Tempo de Execução (s)': tempos_array,
    'Uso de Memória (MB)': memoria_array
})

print("\n📋 Tabela das 100 execuções:")
print(df_resultados.to_string(index=False))

# 📍 Média e desvio padrão das coordenadas
nomes_parametros = ['x2', 'y2', 'x3', 'y3', 'x4', 'y4', 'x5', 'y5', 'x6', 'y6']
print("\n📍 Média e desvio padrão das coordenadas:")
for nome, media, std in zip(nomes_parametros, media_coord, std_coord):
    print(f"{nome}: média = {media:.4f}, desvio padrão = {std:.4f}")

# ⏱ Tempo médio e 🧠 memória média
print(f"\n⏱ Tempo de execução médio: {media_tempo:.4f} s (± {std_tempo:.4f})")
print(f"🧠 Uso de memória médio: {media_mem:.4f} MB (± {std_mem:.4f})")
