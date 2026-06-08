import numpy as np
import matplotlib.pyplot as plt

# ==========================================================
# 1. Definir la cadena de Markov
# ==========================================================
# Estados: 0 y 1
P = np.array([
    [0.8, 0.2],
    [0.3, 0.7]
])

# Número de pasos de la simulación
n = 10000

# Arreglo para guardar la cadena simulada
X = np.zeros(n, dtype=int)

# Estado inicial
X[0] = 0

# ==========================================================
# 2. Simular la cadena de Markov
# ==========================================================
for k in range(1, n):
    estado_actual = X[k - 1]
    X[k] = np.random.choice([0, 1], p=P[estado_actual])

# ==========================================================
# 3. Definir la función f(X_k)
# ==========================================================
# Aquí tomamos f(x) = x
# Entonces:
# f(0) = 0
# f(1) = 1
fX = X.astype(float)

# ==========================================================
# 4. Promedio temporal
# ==========================================================
promedio_temporal = np.cumsum(fX) / np.arange(1, n + 1)

# ==========================================================
# 5. Distribución estacionaria teórica
# ==========================================================
# Resolver pi P = pi, con pi_0 + pi_1 = 1
# Para esta matriz, la solución teórica es:
# pi = (0.6, 0.4)
pi = np.array([0.6, 0.4])

# Media teórica de f(X) bajo la distribución estacionaria
# Como f(0)=0 y f(1)=1, entonces mu = 0*0.6 + 1*0.4 = 0.4
mu = np.dot(pi, [0, 1])

# ==========================================================
# 6. Función para estimar autocovarianzas
# ==========================================================
def autocovarianza(y, lag):
    """
    Calcula la autocovarianza en un retardo 'lag'.
    """
    m = np.mean(y)
    N = len(y)
    return np.sum((y[:N-lag] - m) * (y[lag:] - m)) / N

# ==========================================================
# 7. Estimación de varianza asintótica
# ==========================================================
# sigma^2 ≈ gamma_0 + 2 * sum_{k=1}^m gamma_k
max_lag = 50
gamma = np.array([autocovarianza(fX, k) for k in range(max_lag + 1)])
sigma2_hat = gamma[0] + 2 * np.sum(gamma[1:])

# ==========================================================
# 8. Verificar la idea del TLC
# ==========================================================
# Tomamos varios bloques de tamaño m_block y calculamos:
# sqrt(m_block) * (promedio_bloque - mu)
# Si el TLC aplica, estos valores deberían parecer normales.
m_block = 200
num_blocks = n // m_block

bloques = fX[:num_blocks * m_block].reshape(num_blocks, m_block)
promedios_bloques = np.mean(bloques, axis=1)
Z = np.sqrt(m_block) * (promedios_bloques - mu)

# ==========================================================
# 9. Mostrar resultados numéricos
# ==========================================================
print("========== RESULTADOS ==========")
print("Distribución estacionaria teórica π =", pi)
print("Media teórica mu =", mu)
print("Último promedio temporal =", promedio_temporal[-1])
print("Varianza muestral gamma_0 =", gamma[0])
print("Varianza asintótica estimada sigma^2 =", sigma2_hat)

# ==========================================================
# 10. Gráficas
# ==========================================================
plt.figure(figsize=(10, 5))
plt.plot(promedio_temporal, label="Promedio temporal")
plt.axhline(mu, color="red", linestyle="--", label=f"Media teórica μ = {mu}")
plt.title("Convergencia del promedio temporal")
plt.xlabel("Número de pasos")
plt.ylabel("Promedio")
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(10, 5))
plt.stem(range(max_lag + 1), gamma)
plt.title("Autocovarianzas estimadas")
plt.xlabel("Retardo k")
plt.ylabel("Covarianza")
plt.grid(True)
plt.show()

plt.figure(figsize=(10, 5))
plt.hist(Z, bins=20, edgecolor="black", density=True)
plt.title("Histograma de √m (promedio de bloque - μ)")
plt.xlabel("Valor")
plt.ylabel("Densidad")
plt.grid(True)
plt.show()