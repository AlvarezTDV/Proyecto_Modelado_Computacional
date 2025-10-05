import streamlit
import numpy as np
import matplotlib as plt
import pandas as pd
import sympy as sp
from sympy import symbols, lambdify, sympify
from sympy.parsing.sympy_parser import parse_expr

# ===================================
# FUNCIONES PARA SISTEMAS NO LINEALES
# ===================================

# BISECCIÓN
def metodo_biseccion():
    print("\n--- MÉTODO DE BISECCIÓN ---")
    x = symbols('x')
    ecuacion = input("Ingrese la función en x (ejemplo: x**3 - x - 2): ")
    f = sympify(ecuacion)
    f_lambda = lambdify(x, f, 'numpy')

    a = float(input("Ingrese el límite inferior (a): "))
    b = float(input("Ingrese el límite superior (b): "))
    tol = float(input("Ingrese la tolerancia (por ejemplo 0.0001): "))

    if f_lambda(a) * f_lambda(b) > 0:
        print("⚠ No hay cambio de signo en el intervalo.")
        return

    iteracion = 0
    c = None
    while abs(b - a) > tol:
        iteracion += 1
        c = (a + b) / 2
        if f_lambda(a) * f_lambda(c) < 0:
            b = c
        else:
            a = c
        if abs(f_lambda(c)) < tol:
            break

    print(f"\nRaíz aproximada: {c}")
    print(f"Número de iteraciones: {iteracion}")
    print(f"Tolerancia usada: {tol}")

    X = np.linspace(a - 1, b + 1, 400)
    Y = f_lambda(X)
    plt.axhline(0, color='black', lw=0.8)
    plt.plot(X, Y, label=f"f(x) = {ecuacion}")
    plt.scatter(c, f_lambda(c), color='red', label=f"Raíz ≈ {round(c,4)}")
    plt.legend()
    plt.show()

# MÉTODO SECANTE
def metodo_secante():
    print("\n--- MÉTODO DE LA SECANTE ---")
    x = symbols('x')
    ecuacion = input("Ingrese la función en x (ejemplo: x**3 - x - 2): ")
    f = sympify(ecuacion)
    f_lambda = lambdify(x, f, 'numpy')

    x0 = float(input("Ingrese el primer valor inicial (x0): "))
    x1 = float(input("Ingrese el segundo valor inicial (x1): "))
    tol = float(input("Ingrese la tolerancia (por ejemplo 0.0001): "))
    max_iter = int(input("Ingrese el número máximo de iteraciones: "))

    iteracion = 0
    error = abs(x1 - x0)
    x2 = x1
    while error > tol and iteracion < max_iter:
        iteracion += 1
        f0 = f_lambda(x0)
        f1v = f_lambda(x1)

        if (f1v - f0) == 0:
            print("⚠ División por cero, el método falla.")
            return

        x2 = x1 - f1v * (x1 - x0) / (f1v - f0)
        error = abs(x2 - x1)
        x0, x1 = x1, x2

    print("\n>> Raíz aproximada:", x2)
    print(f">> Iteraciones realizadas: {iteracion}")
    print(f">> Tolerancia usada: {tol}")

    X = np.linspace(x2 - 3, x2 + 3, 400)
    Y = f_lambda(X)
    plt.axhline(0, color='black', lw=0.8)
    plt.plot(X, Y, label=f"f(x) = {ecuacion}")
    plt.scatter(x2, f_lambda(x2), color='red', s=50, label=f"Raíz ≈ {round(x2,4)}")
    plt.legend()
    plt.title("Método de la Secante")
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.grid(True)
    plt.show()

# NEWTON-RAPHSON 2 VARIABLES
def newton_raphson_2v():
    print("\n--- NEWTON-RAPHSON (2 variables) ---")
    x, y = sp.symbols("x y")
    f1_str = input("f1(x,y) = ")
    f2_str = input("f2(x,y) = ")

    f1_expr = parse_expr(f1_str, {"x": x, "y": y, **user_funcs})
    f2_expr = parse_expr(f2_str, {"x": x, "y": y, **user_funcs})

    f1 = lambdify((x, y), f1_expr, "numpy")
    f2 = lambdify((x, y), f2_expr, "numpy")

    guess = list(map(float, input("Ingrese punto inicial (ej: 1 1): ").split()))

    def sistema(vars):
        return [f1(vars[0], vars[1]), f2(vars[0], vars[1])]

    from scipy.optimize import fsolve
    sol = fsolve(sistema, guess)
    print(">> Solución aproximada:", sol)

    try:
        x_vals = np.linspace(sol[0] - 3, sol[0] + 3, 200)
        y_vals = np.linspace(sol[1] - 3, sol[1] + 3, 200)
        X, Y = np.meshgrid(x_vals, y_vals)
        Z1, Z2 = f1(X,Y), f2(X,Y)
        plt.figure(figsize=(7, 6))
        plt.contour(X, Y, Z1, levels=[0], colors='blue', linewidths=2)
        plt.contour(X, Y, Z2, levels=[0], colors='red', linewidths=2)
        plt.scatter(sol[0], sol[1], color='black', s=80)
        plt.title("Newton-Raphson (2 variables)")
        plt.xlabel("x"); plt.ylabel("y")
        plt.legend(["f1=0","f2=0","Solución"])
        plt.grid(True); plt.show()
    except Exception as e:
        print("⚠ No se pudo graficar:", e)

# NEWTON-RAPHSON 3 VARIABLES
def newton_raphson_3v():
    print("\n--- NEWTON-RAPHSON (3 variables) ---")
    x, y, z = sp.symbols("x y z")
    f1_str = input("f1(x,y,z) = ")
    f2_str = input("f2(x,y,z) = ")
    f3_str = input("f3(x,y,z) = ")

    f1_expr = parse_expr(f1_str, {"x":x,"y":y,"z":z,**user_funcs})
    f2_expr = parse_expr(f2_str, {"x":x,"y":y,"z":z,**user_funcs})
    f3_expr = parse_expr(f3_str, {"x":x,"y":y,"z":z,**user_funcs})

    f1 = lambdify((x,y,z),f1_expr,"numpy")
    f2 = lambdify((x,y,z),f2_expr,"numpy")
    f3 = lambdify((x,y,z),f3_expr,"numpy")

    guess = list(map(float, input("Ingrese punto inicial (ej: 1 1 1): ").split()))

    def sistema(vars):
        return [f1(vars[0],vars[1],vars[2]),f2(vars[0],vars[1],vars[2]),f3(vars[0],vars[1],vars[2])]

    from scipy.optimize import fsolve
    sol = fsolve(sistema, guess)
    print(">> Solución aproximada:", sol)

    try:
        from mpl_toolkits.mplot3d import Axes3D
        pts = [guess]; v=np.array(guess)
        for _ in range(8):
            v = v + 0.5*(np.array(sol)-v); pts.append(v.copy())
        pts = np.array(pts)
        fig=plt.figure(figsize=(7,6))
        ax=fig.add_subplot(111,projection='3d')
        ax.plot(pts[:,0],pts[:,1],pts[:,2],marker='o')
        ax.scatter(sol[0],sol[1],sol[2],color='red',label='Solución')
        ax.set_xlabel('x'); ax.set_ylabel('y'); ax.set_zlabel('z')
        ax.set_title('Newton-Raphson (3 variables)'); ax.legend()
        plt.show()
    except Exception as e:
        print("⚠ No se pudo graficar:", e)

# PUNTO FIJO 2 VARIABLES
def punto_fijo_2():
    print("\n--- Punto Fijo (2 variables) ---")
    fx = limpiar_input(input("x = "))
    fy = limpiar_input(input("y = "))

    x0,y0 = map(float,input("Punto inicial (x0 y0): ").split())
    tol = float(input("Tolerancia: "))
    max_iter = int(input("Máximo iteraciones: "))

    def g(v):
        x,y=v
        locals_dict={"x":x,"y":y,**safe_funcs}
        return np.array([eval(fx,{"__builtins__":None},locals_dict),
                         eval(fy,{"__builtins__":None},locals_dict)])

    datos=[];v=np.array([x0,y0])
    for i in range(max_iter):
        nuevo=g(v);error=np.linalg.norm(nuevo-v)
        datos.append([i+1,v[0],v[1],nuevo[0],nuevo[1],error])
        if error<tol:v=nuevo;break
        v=nuevo
    else:print("⚠ No se alcanzó la tolerancia")

    tabla=pd.DataFrame(datos,columns=["Iteración","x0","y0","x1","y1","Error"])
    print(tabla.to_string(index=False))
    print(f"\n>> Solución aproximada: x={v[0]}, y={v[1]}")

    try:
        x_vals=np.linspace(v[0]-3,v[0]+3,200)
        y_vals=np.linspace(v[1]-3,v[1]+3,200)
        X,Y=np.meshgrid(x_vals,y_vals)
        Z1=np.vectorize(lambda x,y:eval(fx,{"__builtins__":None,"x":x,"y":y,**safe_funcs}))(X,Y)-X
        Z2=np.vectorize(lambda x,y:eval(fy,{"__builtins__":None,"x":x,"y":y,**safe_funcs}))(X,Y)-Y
        plt.figure(figsize=(7,6))
        plt.contour(X,Y,Z1,levels=[0],colors='blue',linewidths=2)
        plt.contour(X,Y,Z2,levels=[0],colors='red',linewidths=2)
        plt.scatter(v[0],v[1],color='black',s=70)
        plt.title("Punto Fijo (2 variables)");plt.xlabel("x");plt.ylabel("y");plt.grid(True)
        plt.show()
    except Exception as e:
        print("⚠ No se pudo graficar:", e)

# PUNTO FIJO 3 VARIABLES
def punto_fijo_3():

    print("\n--- Punto Fijo (3 variables) ---")
    fx = limpiar_input(input("x = "))
    fy = limpiar_input(input("y = "))
    fz = limpiar_input(input("z = "))

    x0, y0, z0 = map(float, input("Punto inicial (x0 y0 z0): ").split())
    tol = float(input("Tolerancia: "))
    max_iter = int(input("Máximo iteraciones: "))

    def g(v):
        x, y, z = v
        locals_dict = {"x": x, "y": y, "z": z, **safe_funcs}
        return np.array([
            eval(fx, {"__builtins__": None}, locals_dict),
            eval(fy, {"__builtins__": None}, locals_dict),
            eval(fz, {"__builtins__": None}, locals_dict)
        ])

    datos = []
    v = np.array([x0, y0, z0])
    pts = [v.copy()]  # Guardar cada iteración
    for i in range(max_iter):
        nuevo = g(v)
        error = np.linalg.norm(nuevo - v)
        datos.append([i+1, v[0], v[1], v[2], nuevo[0], nuevo[1], nuevo[2], error])
        v = nuevo
        pts.append(v.copy())
        if error < tol:
            break
    else:
        print("⚠ No se alcanzó la tolerancia")

    tabla = pd.DataFrame(datos, columns=["Iteración", "x0", "y0", "z0", "x1", "y1", "z1", "Error"])
    print(tabla.to_string(index=False))
    print(f"\n>> Solución aproximada: x={v[0]}, y={v[1]}, z={v[2]}")

    try:
        from mpl_toolkits.mplot3d import Axes3D
        pts = np.array(pts)
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d')
        ax.plot(pts[:, 0], pts[:, 1], pts[:, 2], marker='o', color='blue', label="Trayectoria")
        ax.scatter(pts[-1, 0], pts[-1, 1], pts[-1, 2], color='red', s=80, label="Solución")
        ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
        ax.set_title("Punto Fijo (3 variables) - Trayectoria")
        ax.legend()
        plt.show()
    except Exception as e:
        print("⚠ No se pudo graficar:", e)

# ======================
# SISTEMAS DE ECUACIONES
# ======================

# DIRECTOS

# MÉTODO INVERSA
def metodo_inversa(A, b):
    print("\n--- Método de la Inversa ---")
    det = np.linalg.det(A)
    if det == 0:
        print(" El sistema no tiene solución única (det(A) = 0).")
        return
    A_inv = np.linalg.inv(A)
    x = np.dot(A_inv, b)
    print("\nMatriz A:\n", A)
    print("\nVector b:\n", b)
    print("\nInversa de A:\n", A_inv)
    print("\nSolución del sistema (x = A⁻¹·b):\n", x)
    return x

# ELIMINACIÓN DE GAUSS
def gauss_elimination(A, b):
    print("\n--- Método de Eliminación de Gauss ---")
    n = len(b)
    M = np.hstack([A.astype(float), b.reshape(-1,1)])

    for k in range(n):
        max_row = np.argmax(abs(M[k:,k])) + k
        M[[k, max_row]] = M[[max_row, k]]
        for i in range(k+1, n):
            factor = M[i][k] / M[k][k]
            M[i] = M[i] - factor * M[k]

    x = np.zeros(n)
    for i in range(n-1, -1, -1):
        x[i] = (M[i, -1] - np.dot(M[i,i+1:n], x[i+1:n])) / M[i,i]

    return x

# GAUSS-JORDAN
def gauss_jordan(A, b):
    print("\n--- Método de Gauss-Jordan ---")
    n = len(b)
    M = np.hstack([A.astype(float), b.reshape(-1,1)])

    for k in range(n):
        M[k] = M[k] / M[k][k]
        for i in range(n):
            if i != k:
                M[i] = M[i] - M[i][k] * M[k]

    x = M[:, -1]
    print("\nMatriz reducida a forma identidad:\n", M)
    print("\nSolución del sistema:\n", x)
    return x

# REPETITIVOS

# JACOBI
def jacobi(A, b, tol=1e-6, max_iter=100):
    print("\n--- Método Iterativo de Jacobi ---")
    n = len(b)
    x = np.zeros(n)
    historial = []

    for it in range(max_iter):
        x_new = np.zeros(n)
        for i in range(n):
            s = sum(A[i][j] * x[j] for j in range(n) if j != i)
            x_new[i] = (b[i] - s) / A[i][i]
        historial.append([it+1] + list(x_new))

        if np.linalg.norm(x_new - x, ord=np.inf) < tol:
            print(f"\n Convergió en {it+1} iteraciones")
            df = pd.DataFrame(historial, columns=["Iteración"] + [f"x{i+1}" for i in range(n)])
            print(df)
            return x_new
        x = x_new

    print("\n No convergió en el número máximo de iteraciones.")
    df = pd.DataFrame(historial, columns=["Iteración"] + [f"x{i+1}" for i in range(n)])
    print(df)
    return x

# GAUSS-SEIDEL
def gauss_seidel(A, b, tol=1e-6, max_iter=100):
    print("\n--- Método Iterativo de Gauss-Seidel ---")
    n = len(b)
    x = np.zeros(n)
    historial = []

    for it in range(max_iter):
        x_new = np.copy(x)
        for i in range(n):
            s = sum(A[i][j] * x_new[j] for j in range(n) if j != i)
            x_new[i] = (b[i] - s) / A[i][i]
        historial.append([it+1] + list(x_new))

        if np.linalg.norm(x_new - x, ord=np.inf) < tol:
            print(f"\n Convergió en {it+1} iteraciones")
            df = pd.DataFrame(historial, columns=["Iteración"] + [f"x{i+1}" for i in range(n)])
            print(df)
            return x_new
        x = x_new

    print("\n❌ No convergió en el número máximo de iteraciones.")
    df = pd.DataFrame(historial, columns=["Iteración"] + [f"x{i+1}" for i in range(n)])
    print(df)
    return x

# =================
# ALGEBRA MATRICIAL
# =================

# FUNCIONES AUXILIARES

def leer_matriz():
    while True:
        try:
            filas = int(input("Número de filas: "))
            columnas = int(input("Número de columnas: "))
            break
        except ValueError:
            print("Ingrese un número entero válido.")
    
    matriz = []
    print("Ingrese los elementos de la matriz:")
    for i in range(filas):
        fila = []
        for j in range(columnas):
            while True:
                try:
                    valor = float(input(f"Elemento [{i+1}][{j+1}]: "))
                    break
                except ValueError:
                    print("Ingrese un número válido.")
            fila.append(valor)
        matriz.append(fila)
    return np.array(matriz)

def imprimir_matriz(matriz, titulo="Resultado"):
    print(f"\n--- {titulo} ---")
    print(matriz)

# SUMA DE MATRICES
def suma():
    #limpiar_pantalla()
    print("\n--- SUMA DE MATRICES ---")
    print("Primera matriz:")
    A = leer_matriz()
    print("\nSegunda matriz:")
    B = leer_matriz()
    if A.shape != B.shape:
        print(" Error: las matrices deben tener el mismo tamaño.")
    else:
        imprimir_matriz(A + B, "Suma de matrices")

# MULTIPLICACIÓN DE MATRICES
def multiplicacion():
    #limpiar_pantalla()
    print("\n--- MULTIPLICACIÓN DE MATRICES ---")
    print("Primera matriz:")
    A = leer_matriz()
    print("\nSegunda matriz:")
    B = leer_matriz()
    if A.shape[1] != B.shape[0]:
        print("Error: columnas de A deben coincidir con filas de B.")
    else:
        imprimir_matriz(np.dot(A, B), "Multiplicación de matrices")

# DETERMINANTE DE MATRIZ
def determinante():
    #limpiar_pantalla()
    print("\n--- DETERMINANTE DE MATRIZ ---")
    A = leer_matriz()
    if A.shape[0] != A.shape[1]:
        print("Error: la matriz debe ser cuadrada.")
    else:
        det = np.linalg.det(A)
        print(f"\nDeterminante: {det:.6f}")

# INVERSA DE MATRIZ
def inversa():
    #limpiar_pantalla()
    print("\n--- INVERSA DE MATRIZ ---")
    A = leer_matriz()
    if A.shape[0] != A.shape[1]:
        print("Error: la matriz debe ser cuadrada.")
    else:
        try:
            inv = np.linalg.inv(A)
            imprimir_matriz(inv, "Matriz inversa")
        except np.linalg.LinAlgError:
            print("Error: la matriz no es invertible.")