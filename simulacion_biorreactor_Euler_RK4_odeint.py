import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint  # Para resolver EDOs de forma automática

# ===============================================================
#   PARÁMETROS DEL MODELO
# ===============================================================

V = 50                      # Volumen del biorreactor [m^3]
Y = 0.6                     # Coeficiente de rendimiento Biomasa-Sustrato [-]
mumax = 0.5                 # Velocidad máxima de crecimiento específico de la biomasa [1/h]
K1 = 0.35                   # Constante de saturación del sustrato [kg/m^3]
K2 = 5                       # Constante de inhibición del sustrato [m^3/kg]
F = 3                       # Caudal de alimentación al biorreactor [m^3/h]

# ===============================================================
#   CONDICIONES INICIALES
# ===============================================================
S0 = 10                     # Sustrato inicial [kg/m^3]
X0 = 0.1                     # Biomasa inicial [kg/m^3]
y0 = [S0, X0]                # Vector de condiciones iniciales para odeint

# Concentraciones de entrada del sustrato (flujo de alimentación)
Sin_normal = 50             # Sustrato de entrada normal [kg/m^3]
Sin_reducido = 15           # Sustrato reducido temporalmente [kg/m^3]

# ===============================================================
#   TIEMPO DE SIMULACIÓN
# ===============================================================
t0 = 0
tf = 500
dt = 0.09
t = np.arange(t0, tf + dt, dt)  # Vector de tiempo para simulación

# ===============================================================
#   FUNCIÓN PARA DEFINIR Sin EN FUNCION DEL TIEMPO
# ===============================================================
def Sin_func(tiempo):
    # Durante 150 <= t <= 174 h, el sustrato de entrada se reduce a 15 kg/m³
    return Sin_reducido if 150 <= tiempo <= 174 else Sin_normal

# ===============================================================
#   FUNCIÓN PARA CALCULAR mu (CRECIMIENTO ESPECÍFICO)
# ===============================================================
def mu(S):
    denom = (K2 * S + K1)   # Denominador de la función Monod con inhibición por sustrato
    return 0 if denom == 0 else mumax * S / denom  # Evita división por cero

# ===============================================================
#   MÉTODO DE EULER
# ===============================================================

S_euler = np.zeros_like(t)  # Array para almacenar S durante la simulación
X_euler = np.zeros_like(t)  # Array para almacenar X durante la simulación
S = S0
X = X0

for i in range(len(t)):
    S_euler[i] = S  # Guardar valor actual de S
    X_euler[i] = X  # Guardar valor actual de X

    tiempo = t[i]
    Sin_actual = Sin_func(tiempo)  # Determinar Sin en el tiempo actual
    mu_val = mu(S)                 # Velocidad específica en el tiempo actual

    # EDOs discretizadas con Euler
    dXdt = mu_val * X - (F / V) * X                  # Cambio de biomasa
    dSdt = (F / V) * (Sin_actual - S) - (mu_val * X) / Y  # Cambio de sustrato

    # Actualización de valores usando Euler
    S += dSdt * dt
    X += dXdt * dt

    # Evitar valores negativos
    if S < 0: S = 0
    if X < 0: X = 0

# ===============================================================
#   MÉTODO RUNGE-KUTTA 4 (RK4)
# ===============================================================

S_rk4 = np.zeros_like(t)
X_rk4 = np.zeros_like(t)
S = S0
X = X0

# Función para devolver derivadas de S y X
def derivadas(tiempo, S, X, Sin):
    mu_val = mu(S)
    dXdt = mu_val * X - (F / V) * X
    dSdt = (F / V) * (Sin - S) - (mu_val * X) / Y
    return dSdt, dXdt

for i in range(len(t)):
    S_rk4[i] = S
    X_rk4[i] = X
    tiempo = t[i]
    Sin_actual = Sin_func(tiempo)

    # Cálculo de k1, k2, k3, k4 para S y X
    k1_S, k1_X = derivadas(tiempo, S, X, Sin_actual)
    k2_S, k2_X = derivadas(tiempo + dt/2, S + dt*k1_S/2, X + dt*k1_X/2, Sin_actual)
    k3_S, k3_X = derivadas(tiempo + dt/2, S + dt*k2_S/2, X + dt*k2_X/2, Sin_actual)
    k4_S, k4_X = derivadas(tiempo + dt, S + dt*k3_S, X + dt*k3_X, Sin_actual)

    # Actualización de S y X usando RK4
    S += (dt/6)*(k1_S + 2*k2_S + 2*k3_S + k4_S)
    X += (dt/6)*(k1_X + 2*k2_X + 2*k3_X + k4_X)

    if S < 0: S = 0
    if X < 0: X = 0

# ===============================================================
#   RESOLUCIÓN CON ODEINT
# ===============================================================

def modelo(y, tiempo):
    S, X = y
    Sin_actual = Sin_func(tiempo)
    mu_val = mu(S)
    dXdt = mu_val * X - (F / V) * X
    dSdt = (F / V) * (Sin_actual - S) - (mu_val * X) / Y
    return [dSdt, dXdt]

sol = odeint(modelo, y0, t)
S_ode = sol[:, 0]
X_ode = sol[:, 1]

# ===============================================================
#   PERFIL DE ENTRADA DE S y F
# ===============================================================
Sin_plot = np.array([Sin_func(ti) for ti in t])
F_plot = np.full_like(t, F)

# ===============================================================
#   IMPRIMIR PARÁMETROS DEL MODELO
# ===============================================================
print("Volumen del biorreactor (V) =", V, "[m^3]")
print("Coeficiente de rendimiento (Y) =", Y)
print("Velocidad máxima (mumax) =", mumax, "[1/h]")
print("K1 =", K1, "[kg/m^3]")
print("K2 =", K2, "[m^3/kg]")
print("Caudal (F) =", F, "[m^3/h]")
print("Tiempo final =", tf, "[h]")
print("Paso de tiempo =", dt, "[h]")
print("Número de puntos de simulación =", len(t))

# ===============================================================
#   GRAFICAS
# ===============================================================

# 1. Biomasa
plt.figure(figsize=(10, 5))
plt.plot(t, X_euler, 'g-', linewidth=0.8, label='Euler')
plt.plot(t, X_rk4, 'b--', linewidth=0.8, label='RK4')
plt.plot(t, X_ode, 'r-.', linewidth=0.8, label='odeint')
plt.axvspan(150, 174, color='orange', alpha=0.2, label='Reducción Sin a 15 kg/m³')
plt.title("Comparación de Biomasa [X] por método numérico")
plt.xlabel("Tiempo [h]")
plt.ylabel("X [kg/m³]")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# 2. Sustrato
plt.figure(figsize=(10, 5))
plt.plot(t, S_euler, 'g-', linewidth=0.8, label='Euler')
plt.plot(t, S_rk4, 'b--', linewidth=0.8, label='RK4')
plt.plot(t, S_ode, 'r-.', linewidth=0.8, label='odeint')
plt.axvspan(150, 174, color='orange', alpha=0.2, label='Reducción Sin a 15 kg/m³')
plt.title("Comparación de Sustrato [S] por método numérico")
plt.xlabel("Tiempo [h]")
plt.ylabel("S [kg/m³]")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# 3. Perfil de Sin
plt.figure(figsize=(10, 3))
plt.plot(t, Sin_plot, 'm-', linewidth=0.8)
plt.title("Perfil del sustrato de entrada (Sin)")
plt.xlabel("Tiempo [h]")
plt.ylabel("Sin [kg/m³]")
plt.axvspan(150, 174, color='orange', alpha=0.2, label='Reducción Sin a 15 kg/m³')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# 4. Perfil del caudal F
plt.figure(figsize=(10, 3))
plt.plot(t, F_plot, 'c-', linewidth=0.8)
plt.title("Perfil del caudal de alimentación (F)")
plt.xlabel("Tiempo [h]")
plt.ylabel("F [m³/h]")
plt.grid(True)
plt.tight_layout()
plt.show()
