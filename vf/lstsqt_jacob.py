import numpy as np
from scipy.optimize import least_squares

# Distâncias medidas
distancias = {
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

# Ponto fixo
x1, y1 = 100, 100

# Chute inicial
x_inicial = np.array([200, 65, 200, 200, 100, 200, 300, 150, 50, 150])

def distancia_euclidiana(x1, y1, x2, y2):
    return np.sqrt((x2 - x1)**2 + (y2 - y1)**2)

def residuos(params):
    coords = {
        1: (x1, y1),
        2: (params[0], params[1]),
        3: (params[2], params[3]),
        4: (params[4], params[5]),
        5: (params[6], params[7]),
        6: (params[8], params[9]),
    }
    res = []
    for (i, j), d_obs in distancias.items():
        xi, yi = coords[i]
        xj, yj = coords[j]
        d_calc = distancia_euclidiana(xi, yi, xj, yj)
        res.append(d_calc - d_obs)
    return np.array(res)

# Jacobiana analítica
def jacobiana_analitica(params):
    coords = {
        1: (x1, y1),
        2: (params[0], params[1]),
        3: (params[2], params[3]),
        4: (params[4], params[5]),
        5: (params[6], params[7]),
        6: (params[8], params[9]),
    }
    J = np.zeros((15, 10))
    idx_map = {2: 0, 3: 2, 4: 4, 5: 6, 6: 8}

    def deriv(px, py, qx, qy):
        d = np.sqrt((px - qx)**2 + (py - qy)**2)
        if d == 0:
            return 0, 0
        return (px - qx) / d, (py - qy) / d

    for row_idx, ((i, j), _) in enumerate(distancias.items()):
        xi, yi = coords[i]
        xj, yj = coords[j]
        if i != 1:
            di, dj = deriv(xi, yi, xj, yj)
            J[row_idx, idx_map[i]] = di
            J[row_idx, idx_map[i]+1] = dj
        if j != 1:
            djx, djy = deriv(xj, yj, xi, yi)
            J[row_idx, idx_map[j]] = djx
            J[row_idx, idx_map[j]+1] = djy
    return J

resultado = least_squares(residuos, x_inicial, jac=jacobiana_analitica, method='lm', verbose=1)

# Extrair parâmetros ajustados
params_ajustados = resultado.x
x2, y2, x3, y3, x4, y4, x5, y5, x6, y6 = params_ajustados

# Organizar coordenadas ajustadas
coordenadas_ajustadas = {
    'P1': (x1, y1),
    'P2': (x2, y2),
    'P3': (x3, y3),
    'P4': (x4, y4),
    'P5': (x5, y5),
    'P6': (x6, y6)
}

# Covariância dos parâmetros
jacobiana = resultado.jac
sigma0_quadrado = 0.01**2
cov_params = sigma0_quadrado * np.linalg.inv(jacobiana.T @ jacobiana)

print("\nCoordenadas ajustadas:")
for ponto, coords in coordenadas_ajustadas.items():
    print(f"{ponto}: ({coords[0]:.4f}, {coords[1]:.4f})")

desvios_padrao = np.sqrt(np.diag(cov_params))
print("\nDesvios padrão:")
for i, p in enumerate(['x2','y2','x3','y3','x4','y4','x5','y5','x6','y6']):
    print(f"{p}: {desvios_padrao[i]:.4f} m")
