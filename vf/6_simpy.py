import sympy as sp
import math

# 1. Definição dos símbolos
x_syms = sp.symbols('x2 y2 x3 y3 x4 y4 x5 y5 x6 y6')
x1, y1 = 100, 100  # ponto fixo

# 2. Coordenadas dos pontos estimados
pts = [(x_syms[0], x_syms[1]), (x_syms[2], x_syms[3]), (x_syms[4], x_syms[5]),
       (x_syms[6], x_syms[7]), (x_syms[8], x_syms[9])]

# 3. Distâncias observadas
L_obs = [
    104.4226, 141.4264, 100.0186, 200.1519, 70.6993,
    130.0119, 164.0010, 128.0688, 169.9940,
    100.0009, 111.7834, 158.1090,
    206.1490, 70.6874,
    250.0094
]

# 4. Vetor de funções f(x)
F = []

# f1–f5: distância do ponto fixo aos pontos 2 a 6
for xi, yi in pts:
    dist = sp.sqrt((x1 - xi)**2 + (y1 - yi)**2)
    F.append(dist)

# f6–f15: distâncias entre os pares (i, j)
for i in range(5):
    for j in range(i+1, 5):
        xi, yi = pts[i]
        xj, yj = pts[j]
        dist = sp.sqrt((xi - xj)**2 + (yi - yj)**2)
        F.append(dist)

# 5. Jacobiana simbólica
J = sp.Matrix(F).jacobian(x_syms)

# 6. Funções lambdificadas (para avaliação numérica rápida)
f_func = sp.lambdify(x_syms, F, modules="math")
J_func = sp.lambdify(x_syms, J, modules="math")

# 7. Inicialização
x = [200, 75, 195, 205, 95, 200, 300, 160, 50, 150]
max_iter = 100
tolerance = 1e-10

# 8. Funções auxiliares
def dot(u, v):
    return sum(ui * vi for ui, vi in zip(u, v))

def mat_mult(A, B):
    return [[sum(a*b for a, b in zip(A_row, B_col)) for B_col in zip(*B)] for A_row in A]

def mat_transpose(A):
    return [list(row) for row in zip(*A)]

def solve_gauss(A, b):
    n = len(A)
    Ab = [A[i] + [b[i]] for i in range(n)]

    for i in range(n):
        pivot = Ab[i][i]
        if abs(pivot) < 1e-12:
            raise ValueError("Singular matrix")
        for j in range(i+1, n):
            ratio = Ab[j][i] / pivot
            for k in range(i, n+1):
                Ab[j][k] -= ratio * Ab[i][k]

    x = [0] * n
    for i in reversed(range(n)):
        x[i] = Ab[i][n] / Ab[i][i]
        for j in range(i):
            Ab[j][n] -= Ab[j][i] * x[i]
    return x

def norm(v):
    return math.sqrt(sum(vi**2 for vi in v))

# 9. Gauss-Newton
for i in range(max_iter):
    fx = f_func(*x)
    Jx = J_func(*x)

    # Convert fx to residual vector: fx - L_obs
    residuals = [fi - Li for fi, Li in zip(fx, L_obs)]

    JT = mat_transpose(Jx)
    A = mat_mult(JT, Jx)
    b = mat_mult(JT, [[-r] for r in residuals])
    b = [row[0] for row in b]

    try:
        dx = solve_gauss([row[:] for row in A], b)
    except ValueError:
        print("Sistema singular na iteração", i)
        break

    x = [xi + dxi for xi, dxi in zip(x, dx)]

    print(f"Iteração {i+1}, ||dx|| = {norm(dx):.12f}")
    if norm(dx) < tolerance:
        break

# 10. Resultado
print("\nSolução final:")
for i in range(5):
    print(f"x{i+2} = {x[2*i]:.6f}, y{i+2} = {x[2*i+1]:.6f}")
