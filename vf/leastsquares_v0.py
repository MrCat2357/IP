import numpy as np
from scipy.optimize import least_squares

# Coordenadas conhecidas de P1
x1, y1 = 0.0, 0.0

# Estimativas iniciais para P2, P3, P4
# (x2, y2, x3, y3, x4, y4)
initial_guess = np.array([1.0, 2.0, 4.0, 1.0, 3.0, 3.0])

# Distâncias observadas entre os pontos (medidas)
distances = {
    (1, 2): 2.236,  # ~ sqrt(1^2 + 2^2)
    (1, 3): 4.123,  # ~ sqrt(4^2 + 1^2)
    (1, 4): 4.243,  # ~ sqrt(3^2 + 3^2)
    (2, 3): 3.162,  # ~ sqrt(3^2 + 1^2)
    (2, 4): 2.828,  # ~ sqrt(2^2 + 2^2)
    (3, 4): 2.236   # ~ sqrt(1^2 + 2^2)
}

# Função que calcula as diferenças entre distâncias observadas e calculadas
def residuals(params):
    x2, y2, x3, y3, x4, y4 = params
    
    coords = {
        1: (x1, y1),
        2: (x2, y2),
        3: (x3, y3),
        4: (x4, y4)
    }
    
    res = []
    for (i, j), dij_obs in distances.items():
        xi, yi = coords[i]
        xj, yj = coords[j]
        dij_calc = np.sqrt((xi - xj)**2 + (yi - yj)**2)
        res.append(dij_calc - dij_obs)
    
    return res

# Ajustamento via mínimos quadrados não linear
result = least_squares(residuals, initial_guess)

# Resultados
x2, y2, x3, y3, x4, y4 = result.x
print("Coordenadas ajustadas:")
print(f"P2 = ({x2:.4f}, {y2:.4f})")
print(f"P3 = ({x3:.4f}, {y3:.4f})")
print(f"P4 = ({x4:.4f}, {y4:.4f})")
