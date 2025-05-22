import numpy as np
import pandas as pd

# Coordenadas do método Padrão
padrao = np.array([
    201.321, 74.788, 195.147, 204.644, 95.257,
    199.889, 297.402, 159.465, 47.695, 147.579
])

# Outros métodos
newton_gauss = np.array([
    201.4256, 75.2101, 194.7100, 205.0399, 94.8404,
    199.8682, 297.1529, 160.2870, 47.4971, 147.3609
])

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

# Cálculo dos desvios absolutos
desvio_ng = np.linalg.norm(newton_gauss - padrao)
desvio_trf = np.linalg.norm(trf - padrao)
desvio_ip = np.linalg.norm(ip - padrao)
desvio_sqp = np.linalg.norm(sqp - padrao)

# Cálculo dos desvios percentuais
desvio_ng_pct = (desvio_ng / norma_padrao) * 100
desvio_trf_pct = (desvio_trf / norma_padrao) * 100
desvio_ip_pct = (desvio_ip / norma_padrao) * 100
desvio_sqp_pct = (desvio_sqp / norma_padrao) * 100

# Nomes das colunas
colunas = [
    'método', 'x2', 'y2', 'x3', 'y3', 'x4', 'y4',
    'x5', 'y5', 'x6', 'y6', 'desvio', 'desvio_percentual (%)'
]

# Criação da tabela
dados = [
    ['Padrão'] + padrao.tolist() + [0.0, 0.0],
    ['Newton-Gauss'] + newton_gauss.tolist() + [desvio_ng, desvio_ng_pct],
    ['trf'] + trf.tolist() + [desvio_trf, desvio_trf_pct],
    ['ip'] + ip.tolist() + [desvio_ip, desvio_ip_pct],
    ['sqp'] + sqp.tolist() + [desvio_sqp, desvio_sqp_pct]
]

# Criar DataFrame
tabela = pd.DataFrame(dados, columns=colunas)

# Exibe a tabela formatada
pd.set_option('display.float_format', '{:.4f}'.format)
print(tabela.to_string(index=False))
