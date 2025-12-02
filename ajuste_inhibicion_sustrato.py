import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import pandas as pd

# ===============================================================
#   DATOS EXPERIMENTALES  (modificables si quieres probar Ki)
#   S: concentración de sustrato (g/L)
#   v0: velocidad inicial medida (mg MetOH·min⁻¹·g⁻¹)
#   En esta versión v0 decrece a altas concentraciones → típico de inhibición por sustrato
# ===============================================================
S = np.array([0.0, 0.05, 0.15, 0.25, 0.5, 1.0, 1.5, 2.5, 5.0, 25.0, 37.5, 50.0, 75.0, 100.0])
v0 = np.array([
    0.0, 0.76, 2.21, 3.63, 6.98, 12.68, 16.97, 22.09, 26.41,
    27.10, 24.50, 20.20, 14.70, 10.80   # <-- caída a altas [S] = señal clara de inhibición
])

# ===============================================================
#   MODELO CINÉTICO: MICHAELIS–MENTEN con inhibición por sustrato
#
#   Ecuación:
#         v = (Vmax * S) / (Km + S + S²/Ki)
#
#   Donde:
#       - Vmax : velocidad máxima alcanzable
#       - Km   : constante de Michaelis (afinidad)
#       - Ki   : constante de inhibición por sustrato
#   Este modelo es de 3 parámetros y produce la forma "sube y baja".
# ===============================================================
def substrate_inhibition(S, Vmax, Km, Ki):
    return (Vmax * S) / (Km + S + (S**2 / Ki))

# ===============================================================
#   ESTIMACIONES INICIALES (p0)
#   MUY críticas: sin buenos guesses, el ajuste no converge.
#
#   - Vmax_guess: se usa el máximo observado.
#   - Km_guess: un valor razonable típico.
#   - Ki_guess: valor grande para generar inhibición moderada.
# ===============================================================
Vmax_guess = np.max(v0)
Km_guess = 1.0
Ki_guess = 30.0

# Vector con los 3 parámetros iniciales
p0 = [Vmax_guess, Km_guess, Ki_guess]

# ===============================================================
#   AJUSTE NO LINEAL (Levenberg–Marquardt)
#
#   curve_fit encuentra los valores óptimos de Vmax, Km y Ki
#   minimizando la suma de cuadrados entre v0 observado y v predicho.
#   pcov = matriz de covarianza → permite obtener errores estándar.
# ===============================================================
popt, pcov = curve_fit(
    substrate_inhibition,
    S, v0,
    p0=p0,
    method='lm',       # Levenberg–Marquardt → muy bueno para cinética enzimática
    maxfev=200000      # aumento de iteraciones para estabilidad
)

# Parámetros ajustados
Vmax_fit, Km_fit, Ki_fit = popt

# Error estándar de los parámetros = raíz de la diagonal de la covarianza
perr = np.sqrt(np.diag(pcov))
Vmax_se, Km_se, Ki_se = perr

# ===============================================================
#   CÁLCULO DE R² (coeficiente de determinación)
#   R² mide la calidad del ajuste: 1 = perfecto, 0 = sin capacidad predictiva
# ===============================================================
v_pred = substrate_inhibition(S, *popt)    # valores predichos
SS_res = np.sum((v0 - v_pred)**2)          # suma de cuadrados de residuos
SS_tot = np.sum((v0 - np.mean(v0))**2)     # variabilidad total
R2 = 1 - SS_res/SS_tot                      # R² clásico

# ===============================================================
#   IMPRIMIR RESULTADOS
# ===============================================================
print("\n===== Parámetros cinéticos con inhibición por sustrato =====")
print(f"Vmax = {Vmax_fit:.4f} ± {Vmax_se:.4f} mg MetOH·min⁻¹·g⁻¹")
print(f"Km   = {Km_fit:.4f} ± {Km_se:.4f} g·L⁻¹")
print(f"Ki   = {Ki_fit:.4f} ± {Ki_se:.4f} g·L⁻¹")
print(f"R²   = {R2:.5f}")
print("============================================================\n")

# ===============================================================
#   GRAFICAR EL AJUSTE
#
#   - Puntos negros = datos experimentales
#   - Círculos azules o rojos = región <25 o ≥25 g/L
#   - Curva negra = modelo ajustado
# ===============================================================
S_plot = np.linspace(0, np.max(S)*1.05, 400)
v_plot = substrate_inhibition(S_plot, *popt)

plt.figure(figsize=(8,5))
plt.scatter(S, v0, color="black", label="Datos experimentales", zorder=3)

# Encerrar los puntos según el valor de S
for x, y in zip(S, v0):
    edge = "blue" if x < 25 else "red"
    plt.scatter(x, y, s=200, facecolors='none', edgecolors=edge, linewidth=2)

plt.plot(S_plot, v_plot, color="black", linewidth=2.2,
         label="Ajuste con inhibición por sustrato")

plt.xlabel("Pectina [g L⁻¹]", fontsize=12)
plt.ylabel("v₀ [mg MetOH min⁻¹ g⁻¹]", fontsize=12)
plt.title("Ajuste cinético (modelo con inhibición por sustrato)", fontsize=14)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# ===============================================================
#   RESIDUALES
#   Residuales = observado - predicho
#   Deben distribuirse aleatoriamente (ideal).
# ===============================================================
residuals = v0 - v_pred

plt.figure(figsize=(8,3.5))
plt.axhline(0, linestyle='--', color="black")  # línea de referencia

plt.scatter(S, residuals, color="purple", label="Residuales")

plt.xlabel("Pectina [g L⁻¹]")
plt.ylabel("Residuales")
plt.title("Residuales del ajuste cinético")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
