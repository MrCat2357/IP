import math

# Funções auxiliares de álgebra linear
def dot_product(a, b):
    return sum(x * y for x, y in zip(a, b))

def mat_mult(A, B):
    """Multiplica duas matrizes."""
    result = [[0.0] * len(B[0]) for _ in range(len(A))]
    for i in range(len(A)):
        for j in range(len(B[0])):
            for k in range(len(B)):
                result[i][j] += A[i][k] * B[k][j]
    return result

def mat_transpose(A):
    return list(map(list, zip(*A)))

def vec_sub(a, b):
    return [x - y for x, y in zip(a, b)]

def vec_add(a, b):
    return [x + y for x, y in zip(a, b)]

def scalar_mult(v, s):
    return [x * s for x in v]

def norm(v):
    return math.sqrt(sum(x**2 for x in v))

def solve_linear_system(A, b):
    """Resolve Ax = b usando eliminação de Gauss."""
    n = len(A)
    # Criar matriz aumentada
    for i in range(n):
        A[i].append(b[i])
    # Triangularização
    for i in range(n):
        pivot = A[i][i]
        if pivot == 0:
            raise ValueError("Matriz singular")
        for j in range(i+1, n):
            ratio = A[j][i] / pivot
            for k in range(i, n+1):
                A[j][k] -= ratio * A[i][k]
    # Substituição reversa
    x = [0] * n
    for i in reversed(range(n)):
        x[i] = A[i][n] / A[i][i]
        for j in range(i):
            A[j][n] -= A[j][i] * x[i]
    return x


# Ponto fixo
x1, y1 = 100, 100

# Chute inicial
x = [200, 65, 195, 205, 95, 200, 300, 160, 50, 150]

# Distâncias
d = [
    104.4226, 141.4264, 100.0186, 200.1519, 70.6993,
    130.0119, 164.0010, 128.0688, 169.9940,
    100.0009, 111.7834, 158.1090,
    206.1490, 70.6874,
    250.0094
]

def f(x):
    x2, y2, x3, y3, x4, y4, x5, y5, x6, y6 = x
    pts = [(x2, y2), (x3, y3), (x4, y4), (x5, y5), (x6, y6)]
    out = []
    # f1–f5: ponto fixo
    for px, py in pts:
        out.append(math.hypot(x1 - px, y1 - py))
    # f6–f15: entre pares
    for i in range(5):
        for j in range(i+1, 5):
            dx = pts[i][0] - pts[j][0]
            dy = pts[i][1] - pts[j][1]
            out.append(math.hypot(dx, dy))
    return [fi - di for fi, di in zip(out, d)]

def deriv(px, py, qx, qy):
    dx, dy = px - qx, py - qy
    dist = math.hypot(dx, dy)
    if dist == 0:
        return 0.0, 0.0
    return dx / dist, dy / dist

def jacobian(x):
    x2, y2, x3, y3, x4, y4, x5, y5, x6, y6 = x
    J = [[0.0]*10 for _ in range(15)]
    coords = [(x2, y2), (x3, y3), (x4, y4), (x5, y5), (x6, y6)]

    # f1–f5: ponto fixo
    for i, (xi, yi) in enumerate(coords):
        dx, dy = deriv(xi, yi, x1, y1)
        J[i][2*i] = dx
        J[i][2*i + 1] = dy

    # f6–f15
    row = 5
    for i in range(5):
        for j in range(i+1, 5):
            dx1, dy1 = deriv(coords[i][0], coords[i][1], coords[j][0], coords[j][1])
            dx2, dy2 = deriv(coords[j][0], coords[j][1], coords[i][0], coords[i][1])
            J[row][2*i] = dx1
            J[row][2*i + 1] = dy1
            J[row][2*j] = dx2
            J[row][2*j + 1] = dy2
            row += 1
    return J


max_iter = 100
tolerance = 1e-10

for it in range(max_iter):
    fx = f(x)
    J = jacobian(x)
    JT = mat_transpose(J)
    A = mat_mult(JT, J)
    b = mat_mult(JT, [[-v] for v in fx])
    b = [v[0] for v in b]  # vetor coluna para lista

    try:
        dx = solve_linear_system([row[:] for row in A], b)
    except ValueError:
        print("Sistema singular na iteração", it)
        break

    x = vec_add(x, dx)
    print(f"Iteração {it+1}, ||dx|| = {norm(dx):.12f}")
    if norm(dx) < tolerance:
        break

# Resultado final
print("\nSolução final:")
for i in range(5):
    print(f"x{i+2} = {x[2*i]:.6f}, y{i+2} = {x[2*i+1]:.6f}")
