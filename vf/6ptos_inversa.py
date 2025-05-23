import numpy as np

# Coordenada fixa
x1, y1 = 100, 100

# Função do sistema
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
    x2, y2, x3, y3, x4, y4, x5, y5, x6, y6 = x  # Esperando vetor de 10 elementos

    return np.array([
        # d(x1,x2)
        [(x2 - x1)/np.sqrt((x1 - x2)**2 + (y1 - y2)**2), (y2 - y1)/np.sqrt((x1 - x2)**2 + (y1 - y2)**2), 0, 0, 0, 0, 0, 0, 0, 0],
        # d(x1,x3)
        [0, 0, (x3 - x1)/np.sqrt((x1 - x3)**2 + (y1 - y3)**2), (y3 - y1)/np.sqrt((x1 - x3)**2 + (y1 - y3)**2), 0, 0, 0, 0, 0, 0],
        # d(x1,x4)
        [0, 0, 0, 0, (x4 - x1)/np.sqrt((x1 - x4)**2 + (y1 - y4)**2), (y4 - y1)/np.sqrt((x1 - x4)**2 + (y1 - y4)**2), 0, 0, 0, 0],
        # d(x1,x5)
        [0, 0, 0, 0, 0, 0, (x5 - x1)/np.sqrt((x1 - x5)**2 + (y1 - y5)**2), (y5 - y1)/np.sqrt((x1 - x5)**2 + (y1 - y5)**2), 0, 0],
        # d(x1,x6)
        [0, 0, 0, 0, 0, 0, 0, 0, (x6 - x1)/np.sqrt((x1 - x6)**2 + (y1 - y6)**2), (y6 - y1)/np.sqrt((x1 - x6)**2 + (y1 - y6)**2)],
        # d(x2,x3)
        [(x2 - x3)/np.sqrt((x2 - x3)**2 + (y2 - y3)**2), (y2 - y3)/np.sqrt((x2 - x3)**2 + (y2 - y3)**2),
         (x3 - x2)/np.sqrt((x2 - x3)**2 + (y2 - y3)**2), (y3 - y2)/np.sqrt((x2 - x3)**2 + (y2 - y3)**2), 0, 0, 0, 0, 0, 0],
        # d(x2,x4)
        [(x2 - x4)/np.sqrt((x2 - x4)**2 + (y2 - y4)**2), (y2 - y4)/np.sqrt((x2 - x4)**2 + (y2 - y4)**2),
         0, 0, (x4 - x2)/np.sqrt((x2 - x4)**2 + (y2 - y4)**2), (y4 - y2)/np.sqrt((x2 - x4)**2 + (y2 - y4)**2), 0, 0, 0, 0],
        # d(x2,x5)
        [(x2 - x5)/np.sqrt((x2 - x5)**2 + (y2 - y5)**2), (y2 - y5)/np.sqrt((x2 - x5)**2 + (y2 - y5)**2),
         0, 0, 0, 0, (x5 - x2)/np.sqrt((x2 - x5)**2 + (y2 - y5)**2), (y5 - y2)/np.sqrt((x2 - x5)**2 + (y2 - y5)**2), 0, 0],
        # d(x2,x6)
        [(x2 - x6)/np.sqrt((x2 - x6)**2 + (y2 - y6)**2), (y2 - y6)/np.sqrt((x2 - x6)**2 + (y2 - y6)**2),
         0, 0, 0, 0, 0, 0, (x6 - x2)/np.sqrt((x2 - x6)**2 + (y2 - y6)**2), (y6 - y2)/np.sqrt((x2 - x6)**2 + (y2 - y6)**2)],
        # d(x3,x4)
        [0, 0, (x3 - x4)/np.sqrt((x3 - x4)**2 + (y3 - y4)**2), (y3 - y4)/np.sqrt((x3 - x4)**2 + (y3 - y4)**2),
         (x4 - x3)/np.sqrt((x3 - x4)**2 + (y3 - y4)**2), (y4 - y3)/np.sqrt((x3 - x4)**2 + (y3 - y4)**2), 0, 0, 0, 0],
        # d(x3,x5)
        [0, 0, (x3 - x5)/np.sqrt((x3 - x5)**2 + (y3 - y5)**2), (y3 - y5)/np.sqrt((x3 - x5)**2 + (y3 - y5)**2),
         0, 0, (x5 - x3)/np.sqrt((x3 - x5)**2 + (y3 - y5)**2), (y5 - y3)/np.sqrt((x3 - x5)**2 + (y3 - y5)**2), 0, 0],
        # d(x3,x6)
        [0, 0, (x3 - x6)/np.sqrt((x3 - x6)**2 + (y3 - y6)**2), (y3 - y6)/np.sqrt((x3 - x6)**2 + (y3 - y6)**2),
         0, 0, 0, 0, (x6 - x3)/np.sqrt((x3 - x6)**2 + (y3 - y6)**2), (y6 - y3)/np.sqrt((x3 - x6)**2 + (y3 - y6)**2)],
        # d(x4,x5)
        [0, 0, 0, 0, (x4 - x5)/np.sqrt((x4 - x5)**2 + (y4 - y5)**2), (y4 - y5)/np.sqrt((x4 - x5)**2 + (y4 - y5)**2),
         (x5 - x4)/np.sqrt((x4 - x5)**2 + (y4 - y5)**2), (y5 - y4)/np.sqrt((x4 - x5)**2 + (y4 - y5)**2), 0, 0],
        # d(x4,x6)
        [0, 0, 0, 0, (x4 - x6)/np.sqrt((x4 - x6)**2 + (y4 - y6)**2), (y4 - y6)/np.sqrt((x4 - x6)**2 + (y4 - y6)**2),
         0, 0, (x6 - x4)/np.sqrt((x4 - x6)**2 + (y4 - y6)**2), (y6 - y4)/np.sqrt((x4 - x6)**2 + (y4 - y6)**2)],
        # d(x5,x6)
        [0, 0, 0, 0, 0, 0, (x5 - x6)/np.sqrt((x5 - x6)**2 + (y5 - y6)**2), (y5 - y6)/np.sqrt((x5 - x6)**2 + (y5 - y6)**2),
         (x6 - x5)/np.sqrt((x5 - x6)**2 + (y5 - y6)**2), (y6 - y5)/np.sqrt((x5 - x6)**2 + (y5 - y6)**2)]
    ])

# Método de Newton-Gauss
def newton_gauss(x0, tol=1e-9, max_iter=500):
    x = x0
    for i in range(max_iter):
        F = f(x)
        J = jacobian(x)
        # Regularização tipo Levenberg-Marquardt
        lambda_reg = 1e-4  # Pode ajustar esse valor (quanto maior, mais conservador)
        JTJ = J.T @ J + lambda_reg * np.eye(J.shape[1])  # Adiciona lambda à diagonal
        JTF = J.T @ (-F)
        
        # Resolvendo com inversa
        JTJ_inv = np.linalg.inv(J.T @ J + lambda_reg * np.eye(J.shape[1]))
        delta = JTJ_inv @ J.T @ (-F)

        x = x + delta
        if np.linalg.norm(delta) < tol:
            break
    return x

# Chute inicial
x0 = np.array([200, 65, 200, 200, 100, 200, 300, 150, 50, 150])
x_sol = newton_gauss(x0)
print("Solução estimada:")
print(x_sol)
