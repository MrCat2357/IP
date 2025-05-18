import numpy as np
from scipy.optimize import minimize
from itertools import combinations

# 1. Coordenadas reais (somente conhecemos P0)
P0_real = np.array([0.0, 0.0])
P1_real = np.array([3.0, 1.0])
P2_real = np.array([1.0, 4.0])
P3_real = np.array([4.0, 4.0])

# 2. Distâncias reais entre todos os pares
points_real = [P0_real, P1_real, P2_real, P3_real]
distances = {}

for (i, pi), (j, pj) in combinations(enumerate(points_real), 2):
    distances[(i, j)] = np.linalg.norm(pi - pj)

# 3. Estimativas iniciais para P1, P2, P3
P1_init = np.array([2.0, 0.0])
P2_init = np.array([0.0, 3.0])
P3_init = np.array([3.0, 3.0])
initial_guess = np.hstack([P1_init, P2_init, P3_init])  # flat array

# 4. Função de erro (diferença entre distâncias estimadas e reais)
def error_function(x):
    # x = [x1, y1, x2, y2, x3, y3]
    P1 = x[0:2]
    P2 = x[2:4]
    P3 = x[4:6]
    points = [P0_real, P1, P2, P3]
    
    error = 0.0
    for (i, j), d_real in distances.items():
        d_est = np.linalg.norm(points[i] - points[j])
        error += (d_est - d_real) ** 2
    return error

# 5. Otimização
result = minimize(error_function, initial_guess, method='Newton-CG')

# 6. Resultado
if result.success:
    optimized_coords = result.x
    P1_opt = optimized_coords[0:2]
    P2_opt = optimized_coords[2:4]
    P3_opt = optimized_coords[4:6]

    print("Coordenadas estimadas:")
    print("P1:", P1_opt)
    print("P2:", P2_opt)
    print("P3:", P3_opt)

    print("\nCoordenadas reais:")
    print("P1:", P1_real)
    print("P2:", P2_real)
    print("P3:", P3_real)
else:
    print("Otimização falhou:", result.message)
