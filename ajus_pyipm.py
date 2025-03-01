import numpy as np
import pyipm


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

# Matriz A (coeficientes das diferenças de altura)
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

# Função objetivo para o MMQ: minimizar || A * x - b ||^2
def objective(x):
    return np.sum((np.dot(A, x) - b) ** 2)

# Definindo o problema de otimização com pyipm
# A função objective deve ser minimizada
def ipm_ajuste_rede():
    # Valor inicial para as altitudes dos pontos (palpite inicial)
    x0 = np.array([0, 0, 0, 0, 0, 0])

    # Criando o objeto 'solver' do pyipm
    solver = pyipm.ipm()
    
    # Definindo o problema para o solver:
    #   Minimizar a função objetivo subject to as condições do problema
    solver.add_objective(objective)  # Função objetivo
    solver.add_constraint(A)         # Matriz A das restrições lineares
    solver.add_rhs(b)                # Vetor b das observações

    # Solucionando o problema com o método interior-point
    solver.solve()

    # Retornando as altitudes ajustadas
    return solver.get_solution()

# Realizando o ajuste da rede de nivelamento
altitudes_ajustadas = ipm_ajuste_rede()

# Exibindo as altitudes ajustadas
print("Altitudes ajustadas: P1 = {:.4f}, P2 = {:.4f}, P3 = {:.4f}, P4 = {:.4f}, P5 = {:.4f}, P6 = {:.4f}".format(altitudes_ajustadas[0], altitudes_ajustadas[1], altitudes_ajustadas[2], altitudes_ajustadas[3], altitudes_ajustadas[4], altitudes_ajustadas[5]))