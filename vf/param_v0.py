import numpy as np

# Chute inicial
x1 = 100
y1 = 100 

x = np.array([200, 65, 200, 200, 100, 200, 300, 150, 50, 150])

#y = np.array([104.4226, 141.4264, 100.0186, 206.1519, 706.993, 130.0119, 164.0010, 128.0688, 169.9940, 100.0009, 111.7834, 158.1090, 206.1490, 70.6874, 250.0094]) 

# Função do sistema
def f(x):
    x2, y2, x3, y3, x4, y4, x5, y5, x6, y6 = x  

    return np.array([
        np.sqrt((x1 - x2)**2 + (y1 - y2)**2) -104.4226,
        np.sqrt((x1 - x3)**2 + (y1 - y3)**2) -141.4264,
        np.sqrt((x1 - x4)**2 + (y1 - y4)**2) -100.0186,
        np.sqrt((x1 - x5)**2 + (y1 - y5)**2) -206.1519,
        np.sqrt((x1 - x6)**2 + (y1 - y6)**2) -706.993,
        np.sqrt((x2 - x3)**2 + (y2 - y3)**2) -130.0119,
        np.sqrt((x2 - x4)**2 + (y2 - y4)**2) -164.0010,
        np.sqrt((x2 - x5)**2 + (y2 - y5)**2) -128.0688,
        np.sqrt((x2 - x6)**2 + (y2 - y6)**2) -169.9940,
        np.sqrt((x3 - x4)**2 + (y3 - y4)**2) -100.0009,
        np.sqrt((x3 - x5)**2 + (y3 - y5)**2) -111.7834,
        np.sqrt((x3 - x6)**2 + (y3 - y6)**2) -158.1090,
        np.sqrt((x4 - x5)**2 + (y4 - y5)**2) -206.1490,
        np.sqrt((x4 - x6)**2 + (y4 - y6)**2) -70.6874,
        np.sqrt((x5 - x6)**2 + (y5 - y6)**2) -250.0094
    ])

# Jacobiana do sistema
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

# Parâmetros
max_iter = 2000
tolerance = 1e-6

# Iterações de Newton (versão de mínimos quadrados)
for i in range(max_iter):
    fx = f(x)
    J = jacobian(x)
    
    # Resolve (J^T J) * dx = J^T * (-f)
    try:
        JT = J.T
        A = JT @ J              # J^T * J
        b = JT @ (-fx)          # J^T * (-fx)
        dx = np.linalg.solve(A, b)

    except np.linalg.LinAlgError:
        print("Jacobiana transposta * Jacobiana é singular na iteração", i)
        break

    x = x + dx
    print(f"Iteração {i+1}, x = {x}, ||dx|| = {np.linalg.norm(dx):.12f}")

    if np.linalg.norm(dx) < tolerance:
        break

# Resultado final
print("\nSolução aproximada encontrada:")
print(f"x2 = {x[0]:.6f}")
print(f"y2 = {x[1]:.6f}")
print(f"x3 = {x[2]:.6f}")
print(f"y3 = {x[3]:.6f}")
print(f"x4 = {x[4]:.6f}")
print(f"y4 = {x[5]:.6f}")
print(f"x5 = {x[6]:.6f}")
print(f"y5 = {x[7]:.6f}")
print(f"x6 = {x[8]:.6f}")
print(f"y6 = {x[9]:.6f}")
print(f"y4 = {x[5]:.6f}")
