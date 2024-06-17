def neural_network_method(X, W, Y, epsilon):
    n = len(X)
    m = len(X[0])

    iteration = 0
    errors = []

    def relative_error(x_new, x):
        return [abs((x_new[i] - x[i]) / x_new[i]) * 100 if x_new[i] != 0 else float('inf') for i in range(n)]

    # Mostrar el procedimiento XW=Y
    print("\nProcedimiento XW = Y:")
    for i in range(n):
        print(f"Y{i+1} = {Y[i]}")
        print(f"X{i+1} * W = {sum(X[i][j] * W[j] for j in range(m))}")

    # Utilizar los pesos W directamente en la primera iteración
    W_new = W.copy()

    while True:
        if iteration == 0:
            W_new = W.copy()
        else:
            for i in range(m):
                for k in range(n):
                    W_new[i] += 0.01 * X[k][i] * (Y[k] - sum(X[k][j] * W[j] for j in range(m)))

        error = relative_error(W_new, W)
        iteration += 1
        print(f"Iteración {iteration}")
        print(f"Error en la iteración {iteration}: {[round(err, 5) for err in error]}")
        if all(err < epsilon for err in error):
            break
        W = W_new.copy()
        if iteration >= 1000:
            print("\nSe ha alcanzado el límite de iteraciones.")
            break

        return [round(val, 5) for val in W]

def main():
    print("Seleccione el método:")
    print("1. Método de Jacobi")
    print("2. Método de Gauss-Seidel")
    print("3. Método de Doolittle")
    print("4. Red Neuronal")
    
    method = int(input())
    
    if method in [1, 2]:
        n = int(input("\nIngrese el tamaño de la matriz (n x n): "))
        A = []
        print("\nIngrese los elementos de la matriz A:")
        for i in range(n):
            row = list(map(float, input().split()))
            A.append(row)
        b = list(map(float, input("\nIngrese los elementos del vector b: ").split()))
        
        is_dominant = True
        for i in range(n):
            sum_row = sum(abs(A[i][j]) for j in range(n) if i != j)
            if abs(A[i][i]) <= sum_row:
                is_dominant = False
            print(f"Fila {i+1}: {abs(A[i][i])} {'>=' if abs(A[i][i]) >= sum_row else '<'} {sum_row}")

        if is_dominant:
            print("\nLa matriz es diagonalmente dominante.")
        else:
            print("\nLa matriz NO es diagonalmente dominante.")

        epsilon = 0.001
        if method == 1:
            solution = jacobi_method(A, b, epsilon)
        else:
            solution = gauss_seidel_method(A, b, epsilon)
    
    elif method == 3:
        n = int(input("\nIngrese el tamaño de la matriz (n x n): "))
        A = []
        print("\nIngrese los elementos de la matriz A:")
        for i in range(n):
            row = list(map(float, input().split()))
            A.append(row)
        b = list(map(float, input("\nIngrese los elementos del vector b: ").split()))
        solution = doolittle_method(A, b)
    
    elif method == 4:
        n = int(input("\nIngrese el tamaño de la matriz X (n x m): "))
        print("\nIngrese los datos de entrada (X) en forma de matriz:")
        X = []
        for i in range(n):
            row = list(map(float, input().split()))
            X.append(row)

        print("\nIngrese los pesos de la red neuronal (W) en forma de vector:")
        W = list(map(float, input().split()))

        print("\nIngrese los datos de salida deseada (Y) en forma de vector:")
        Y = list(map(float, input().split()))

        epsilon = 0.001
        W = neural_network_method(X, W, Y, epsilon)
        
        A = [[X[i][j] for j in range(len(W))] for i in range(len(X))]
        b = Y
        solution = jacobi_method(A, b, epsilon)
    
    else:
        print("\nMétodo no válido")
        return
    
    print(f"\nSolución final: {solution}")

if __name__ == "__main__":
    main()

