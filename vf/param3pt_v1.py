import numpy as np

x1 = 2
y1 = 1 

# Chute inicial

x = np.array([5.3, 0.7, 2.6, 4.8])

#y = np.array([104.4226, 141.4264, 100.0186, 206.1519, 706.993, 130.0119, 164.0010, 128.0688, 169.9940, 100.0009, 111.7834, 158.1090, 206.1490, 70.6874, 250.0094]) 

# Função do sistema
def f(x):
    x2, y2, x3, y3 = x  

    return np.array([
        np.sqrt((x1 - x2)**2 + (y1 - y2)**2) - 3,
        np.sqrt((x1 - x3)**2 + (y1 - y3)**2) - 4,
        np.sqrt((x2 - x3)**2 + (y2 - y3)**2) - 5,
    ])

# Jacobiana do sistema
def jacobian(x):
    x2, y2, x3, y3 = x  # Esperando vetor de 10 elementos

    return np.array([
        # d(x1,x2)
        [(x2 - x1)/np.sqrt((x1 - x2)**2 + (y1 - y2)**2), (y2 - y1)/np.sqrt((x1 - x2)**2 + (y1 - y2)**2), 0, 0],
        # d(x1,x3)
        [0, 0, (x3 - x1)/np.sqrt((x1 - x3)**2 + (y1 - y3)**2), (y3 - y1)/np.sqrt((x1 - x3)**2 + (y1 - y3)**2)],
        [(x2 - x3)/np.sqrt((x2 - x3)**2 + (y2 - y3)**2), (y2 - y3)/np.sqrt((x2 - x3)**2 + (y2 - y3)**2),
         (x3 - x2)/np.sqrt((x2 - x3)**2 + (y2 - y3)**2), (y3 - y2)/np.sqrt((x2 - x3)**2 + (y2 - y3)**2)]
    ])

# Parâmetros
max_iter = 100
tolerance = 1e-10

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
    print(f"Iteração {i+1}, x = {x}, ||dx|| = {np.linalg.norm(dx):.6f}")

    if np.linalg.norm(dx) < tolerance:
        break

# Resultado final
print("\nSolução aproximada encontrada:")
print(f"x2 = {x[0]:.6f}")
print(f"y2 = {x[1]:.6f}")
print(f"x3 = {x[2]:.6f}")
print(f"y3 = {x[3]:.6f}")


