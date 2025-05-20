import sympy as sp
import math

# 1. Coordenadas conhecidas
x1, y1 = 2, 1

# 2. Variáveis simbólicas
x2, y2, x3, y3, x4, y4 = sp.symbols('x2 y2 x3 y3 x4 y4')
x_syms = [x2, y2, x3, y3, x4, y4]

# 3. Distâncias conhecidas
d12, d13, d14 = 3, 4, 5
d23, d24, d34 = 5, 4, 3

# 4. Funções do sistema
F = [
    sp.sqrt((x2 - x1)**2 + (y2 - y1)**2) - d12,
    sp.sqrt((x3 - x1)**2 + (y3 - y1)**2) - d13,
    sp.sqrt((x4 - x1)**2 + (y4 - y1)**2) - d14,
    sp.sqrt((x2 - x3)**2 + (y2 - y3)**2) - d23,
    sp.sqrt((x2 - x4)**2 + (y2 - y4)**2) - d24,
    sp.sqrt((x3 - x4)**2 + (y3 - y4)**2) - d34,
]

# 5. Jacobiana simbólica
J = sp.Matrix(F).jacobian(x_syms)

# 6. Lambdify funções e jacobiana
f_func = sp.lambdify(x_syms, F, modules='math')
J_func = sp.lambdify(x_syms, J, modules='math')

# 7. Chute inicial
x = [5.3, 0.7, 2.6, 4.8, 5.7, 5.3]

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

# 9. Iterações de Gauss-Newton
max_iter = 100
tolerance = 1e-10

for i in range(max_iter):
    fx = f_func(*x)
    Jx = J_func(*x)

    JT = mat_transpose(Jx)
    A = mat_mult(JT, Jx)
    b = mat_mult(JT, [[-r] for r in fx])
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

# 10. Resultado final
print("\nSolução aproximada encontrada:")
print(f"x2 = {x[0]:.6f}, y2 = {x[1]:.6f}")
print(f"x3 = {x[2]:.6f}, y3 = {x[3]:.6f}")
print(f"x4 = {x[4]:.6f}, y4 = {x[5]:.6f}")
