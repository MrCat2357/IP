import numpy as np
import pandas as pd
import tkinter as tk
from tkinter import filedialog

# Oculta a janela principal do tkinter
root = tk.Tk()
root.withdraw()

# Coordenadas do método Padrão
padrao = np.array([
    201.321, 74.788, 195.147, 204.644, 95.257,
    199.889, 297.402, 159.465, 47.695, 147.579
])

# Outros métodos
inversa = np.array([
    199.8376, 69.4362, 200.5760, 199.4375, 100.5742,
    199.9997, 300.2849, 148.8852, 50.2986, 150.2929
])

cholesky = np.array([
    199.8376, 69.4362, 200.5760, 199.4375, 100.5742,
    199.9997, 300.2849, 148.8852, 50.2986, 150.2929])

trf = np.array([
    201.4816, 75.4403, 194.4714, 205.2545, 94.6138,
    199.8562, 297.0156, 160.7342, 47.3898, 147.2416
])

ip = np.array([
    199.8382, 69.4381, 200.5741, 199.4394, 100.5723,
    199.9997, 300.2839, 148.8891, 50.2977, 150.2920
])

sqp = np.array([
    199.8384, 69.4389, 200.5734, 199.4401, 100.5716,
    199.9997, 300.2836, 148.8905, 50.2973, 150.2916
])

# Norma do vetor padrão
norma_padrao = np.linalg.norm(padrao)

# Cálculo dos desvios absolutos e percentuais
def calc_desvio(vetor):
    desvio_abs = np.linalg.norm(vetor - padrao)
    desvio_pct = (desvio_abs / norma_padrao) * 100
    return desvio_abs, desvio_pct

# Tabela de dados
dados = [
    ['Padrão'] + padrao.tolist() + [0.0, 0.0],
    ['NG-Inversa'] + inversa.tolist() + list(calc_desvio(inversa)),
    ['NG-Cholesky'] + cholesky.tolist() + list(calc_desvio(cholesky)),
    ['trf'] + trf.tolist() + list(calc_desvio(trf)),
    ['ip'] + ip.tolist() + list(calc_desvio(ip)),
    ['sqp'] + sqp.tolist() + list(calc_desvio(sqp)),
]

# Cabeçalho da tabela
colunas = [
    'método', 'x2', 'y2', 'x3', 'y3', 'x4', 'y4',
    'x5', 'y5', 'x6', 'y6', 'desvio', 'desvio_percentual (%)'
]

# Criar DataFrame
tabela = pd.DataFrame(dados, columns=colunas)

# Exibe no terminal
pd.set_option('display.float_format', '{:.4f}'.format)
print(tabela.to_string(index=False))

# Salvar CSV
print("\n💾 Selecione o local para salvar a tabela como CSV...")
caminho_csv = filedialog.asksaveasfilename(
    defaultextension=".csv",
    filetypes=[("CSV files", "*.csv")],
    title="Salvar tabela como CSV"
)

if caminho_csv:
    tabela.to_csv(caminho_csv, index=False)
    print(f"✅ Tabela salva com sucesso em: {caminho_csv}")
else:
    print("❌ Salvamento cancelado.")
