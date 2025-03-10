import matplotlib.pyplot as plt

# Métodos utilizados
metodos = ['Inversa', 'Cholesky', 'Interior-Point', 'SQP', 'TRF']

# Consumo de memória (em KB) correspondente a cada método
consumo_memoria = [2804, 2648, 796, 32, 608]

# Tempo de execução (em segundos) correspondente a cada método
tempo_execucao = [0.006965, 0.010217, 2.24594, 0.156619, 0.079912]

# Criando os gráficos lado a lado
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Gráfico 1: Consumo de memória
ax1.bar(metodos, consumo_memoria, color='skyblue')
ax1.set_title('Consumo de Memória por Método de Otimização', fontsize=14)
ax1.set_xlabel('Método Utilizado', fontsize=12)
ax1.set_ylabel('Consumo de Memória (KB)', fontsize=12)
ax1.set_xticklabels(metodos, rotation=45)

# Gráfico 2: Tempo de execução
ax2.bar(metodos, tempo_execucao, color='lightgreen')
ax2.set_title('Tempo de Execução por Método de Otimização', fontsize=14)
ax2.set_xlabel('Método Utilizado', fontsize=12)
ax2.set_ylabel('Tempo de Execução (segundos)', fontsize=12)
ax2.set_xticklabels(metodos, rotation=45)

# Ajustando o layout
plt.tight_layout()

# Exibindo os gráficos
plt.show()
