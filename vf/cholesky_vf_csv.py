import numpy as np
import pandas as pd
import time

# Para caixa de diálogo de salvamento
import tkinter as tk
from tkinter import filedialog

# Desabilita a janela principal do tkinter
root = tk.Tk()
root.withdraw()

# Tentar importar psutil
try:
    import psutil
    psutil_disponivel = True
    processo = psutil.Process()
    memoria_inicial = processo.memory_info().rss / 1024 / 1024  # em MB
except ImportError:
    print("Aviso: psutil não está instalado. Uso de memória não será medido.")
    psutil_disponivel = False
    memoria_inicial = 0

from scipy.linalg import cho_factor, cho_solve

x1, y1 = 100, 100

def f(x):
    x2, y2, x3, y3, x4, y4, x5, y5, x6, y6 = x  
    return np.array([
        np.sqrt((x1 - x2)**2 + (y1 - y2)**2) - 104.4226,
        np.sqrt((x1 - x3)**2 + (y1 - y3)**2) - 141.4264,
        np.sqrt((x1 - x4)**2 + (y1 - y4)**2) - 100.0186,
        np.sqrt((x1 - x5)**2 + (y1 - y5)**2) - 206.1519,
        np.sqrt((x1 - x6)**2 + (y1 - y6)**2) - 70.6993,
        np.sqrt((x2 - x3)**2 + (y2 - y3)**2) - 130.0119,
        np.sqrt((x2 - x4)**2 + (y2 - y4)**2) - 164.0010,
        np.sqrt((x2 - x5)**2 + (y2 - y5)**2) - 128.0688,
        np.sqrt((x2 - x6)**2 + (y2 - y6)**2) - 169.9940,
        np.sqrt((x3 - x4)**2 + (y3 - y4)**2) - 100.0009,
        np.sqrt((x3 - x5)**2 + (y3 - y5)**2) - 111.7834,
        np.sqrt((x3 - x6)**2 + (y3 - y6)**2) - 158.1090,
        np.sqrt((x4 - x5)**2 + (y4 - y5)**2) - 206.1490,
        np.sqrt((x4 - x6)**2 + (y4 - y6)**2) - 70.6874,
        np.sqrt((x5 - x6)**2 + (y5 - y6)**2) - 250.0094
    ])

def jacobian(x):
    x2, y2, x3, y3, x4, y4, x5, y5, x6, y6 = x
    def dist_grad(xa, ya, xb, yb):
        d = np.sqrt((xa - xb)**2 + (ya - yb)**2)
        return (xa - xb)/d, (ya - yb)/d

    J = np.zeros((15, 10))
    J[0, 0], J[0, 1] = dist_grad(x2, y2, x1, y1)
    J[1, 2], J[1, 3] = dist_grad(x3, y3, x1, y1)
    J[2, 4], J[2, 5] = dist_grad(x4, y4, x1, y1)
    J[3, 6], J[3, 7] = dist_grad(x5, y5, x1, y1)
    J[4, 8], J[4, 9] = dist_grad(x6, y6, x1, y1)

    grads = [
        (x2, y2, x3, y3, 0, 2), (x2, y2, x4, y4, 0, 4), (x2, y2, x5, y5, 0, 6),
        (x2, y2, x6, y6, 0, 8), (x3, y3, x4, y4, 2, 4), (x3, y3, x5, y5, 2, 6),
        (x3, y3, x6, y6, 2, 8), (x4, y4, x5, y5, 4, 6), (x4, y4, x6, y6, 4, 8),
        (x5, y5, x6, y6, 6, 8)
    ]
    for k, (xa, ya, xb, yb, ia, ib) in enumerate(grads, start=5):
        d = np.sqrt((xa - xb)**2 + (ya - yb)**2)
        J[k, ia] = (xa - xb)/d
        J[k, ia + 1] = (ya - yb)/d
        J[k, ib] = (xb - xa)/d
        J[k, ib + 1] = (yb - ya)/d

    return J

def newton_gauss(x0, tol=1e-9, max_iter=500):
    x = x0.copy()
    for _ in range(max_iter):
        F = f(x)
        J = jacobian(x)
        lambda_reg = 1e-4
        JTJ = J.T @ J + lambda_reg * np.eye(J.shape[1])
        JTF = J.T @ (-F)
        delta = cho_solve(cho_factor(JTJ), JTF)
        x += delta
        if np.linalg.norm(delta) < tol:
            break
    return x

x0 = np.array([200, 65, 200, 200, 100, 200, 300, 150, 50, 150], dtype=float)

num_execucoes = 100
coordenadas_resultados = []
tempos_execucao = []
usos_memoria = []

for _ in range(num_execucoes):
    tempo_inicio = time.time()
    x_sol = newton_gauss(x0)
    tempo_fim = time.time()
    tempos_execucao.append(tempo_fim - tempo_inicio)
    coordenadas_resultados.append(x_sol)
    if psutil_disponivel:
        mem_uso = processo.memory_info().rss / 1024 / 1024
        usos_memoria.append(mem_uso - memoria_inicial)
    else:
        usos_memoria.append(np.nan)

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

# Tabela
df_resultados = pd.DataFrame({
    'Execução': np.arange(1, num_execucoes + 1),
    'Tempo de Execução (s)': tempos_array,
    'Uso de Memória (MB)': memoria_array
})

print("\n📊 Tabela das 100 execuções:")
print(df_resultados.to_string(index=False))

# Estatísticas finais
nomes_parametros = ['x2', 'y2', 'x3', 'y3', 'x4', 'y4', 'x5', 'y5', 'x6', 'y6']
print("\n📍 Média e desvio padrão das coordenadas:")
for nome, media, std in zip(nomes_parametros, media_coord, std_coord):
    print(f"{nome}: média = {media:.4f}, desvio padrão = {std:.4f}")

print(f"\n⏱ Tempo de execução médio: {media_tempo:.4f} s (± {std_tempo:.4f})")
print(f"🧠 Uso de memória médio: {media_mem:.4f} MB (± {std_mem:.4f})")

# 📝 Exporta para CSV com escolha do local
print("\n💾 Selecione o local e nome do arquivo CSV para salvar os resultados...")
caminho_csv = filedialog.asksaveasfilename(
    defaultextension=".csv",
    filetypes=[("CSV files", "*.csv")],
    title="Salvar resultados como CSV"
)

if caminho_csv:
    df_resultados.to_csv(caminho_csv, index=False)
    print(f"\n✅ Resultados salvos com sucesso em: {caminho_csv}")
else:
    print("\n❌ Salvamento cancelado.")
