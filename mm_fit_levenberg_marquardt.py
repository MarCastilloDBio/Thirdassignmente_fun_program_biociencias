import numpy as np 
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import pandas as pd

# --- Experimental data ---
# S : concentración de sustrato (pectina) en g/L
# v0: velocidad inicial medida (mg MetOH min⁻¹ g⁻¹)
S = np.array([0.0, 0.05, 0.15, 0.25, 0.5, 1.0, 1.5, 2.5, 5.0, 25.0, 37.5, 50.0, 75.0, 100.0])
v0 = np.array([0.0, 0.76, 2.21, 3.63, 6.98, 12.68, 16.97, 22.09, 26.41, 29.38, 29.59, 29.69, 29.79, 29.85])

# --- Michaelis–Menten model ---
# Esta función define la ecuación clásica:
# v = (Vmax * S) / (Km + S)
# curve_fit ajustará Vmax y Km para minimizar el error.
def mm_model(S, Vmax, Km):
    return Vmax * S / (Km + S)

# --- Initial guesses (estimaciones iniciales para el ajuste) ---
# curve_fit necesita aproximaciones para Vmax y Km (parámetros a optimizar).

# Se estima Vmax como ~5% mayor que el valor máximo experimental
Vmax_guess = np.max(v0) * 1.05

# Se encuentra el S cuya velocidad está más cerca de la mitad de Vmax
# Esto permite estimar Km aproximadamente (definición de Km).
half = Vmax_guess / 2.0
idx = np.argmin(np.abs(v0 - half))

# Si la concentración en ese índice es válida, se usa como Km_guess
Km_guess = S[idx] if S[idx] > 0 else 1.0

# Vector con estimaciones iniciales
p0 = [Vmax_guess, Km_guess]

# --- Nonlinear fit ---
# curve_fit ajusta el modelo a los datos experimentales usando:
# - método LM (Levenberg–Marquardt)
# - mm_model como función objetivo
# - p0 como valores iniciales
# Devuelve:
#   popt = parámetros óptimos [Vmax, Km]
#   pcov = matriz de covarianza (para calcular error estándar)
popt, pcov = curve_fit(mm_model, S, v0, p0=p0, method='lm', maxfev=100000)
Vmax_fit, Km_fit = popt

# Error estándar de los parámetros = sqrt(diagonal de la matriz de covarianza)
perr = np.sqrt(np.diag(pcov))
Vmax_se, Km_se = perr

# --- Predictions and R² ---
# Se calculan las velocidades predichas con el modelo ajustado
v_pred = mm_model(S, *popt)

# Suma de cuadrados de residuos (error del modelo)
SS_res = np.sum((v0 - v_pred)**2)

# Suma total de cuadrados (variabilidad de los datos)
SS_tot = np.sum((v0 - np.mean(v0))**2)

# Coeficiente de determinación (bondad del ajuste)
R2 = 1 - SS_res/SS_tot

# --- PRINT RESULTS ---
print("===== Parámetros cinéticos (Michaelis–Menten) =====")
print(f"Vmax = {Vmax_fit:.4f} ± {Vmax_se:.4f} mg MetOH min⁻¹ g⁻¹")
print(f"Km   = {Km_fit:.4f} ± {Km_se:.4f} g L⁻¹")
print(f"R²   = {R2:.5f}")
print("====================================================")

# --- Plot data + fit ---
# S_plot: rango continuo de S para dibujar la curva suave
S_plot = np.linspace(0, np.max(S)*1.05, 300)
# v_plot: curva predicha usando los parámetros ajustados
v_plot = mm_model(S_plot, *popt)

plt.figure(figsize=(8,5))

# Puntos experimentales en negro
plt.scatter(S, v0, color="black", label="Datos experimentales", zorder=3)

# Se rodean los puntos:
# - azul si S < 25
# - rojo si S >= 25
# Los puntos siguen siendo negros, solo se añade un borde coloreado
for x, y in zip(S, v0):
    edge = "blue" if x < 25 else "red"
    plt.scatter(x, y, s=200, facecolors='none', edgecolors=edge, linewidth=2)

# Curva ajustada en negro
plt.plot(S_plot, v_plot, color="black", linewidth=2.2, label="Ajuste Michaelis–Menten")

plt.xlabel("Pectina [g L⁻¹]", fontsize=12)
plt.ylabel("v₀ [mg MetOH min⁻¹ g⁻¹]", fontsize=12)
plt.title("Ajuste Michaelis–Menten (Levenberg–Marquardt)", fontsize=14)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# --- Residual plot ---
# Residuales = observado – predicho
residuals = v0 - v_pred

# Se separan las regiones <25 y ≥25 para colorearlas
S_low = S[S < 25]
S_high = S[S >= 25]

plt.figure(figsize=(8,3.5))
plt.axhline(0, linestyle='--', color="black")  # referencia horizontal

# Residuales de cada región
plt.scatter(S_low, residuals[S < 25], color="blue", label="Residuales (<25 g L⁻¹)")
plt.scatter(S_high, residuals[S >= 25], color="orange", label="Residuales (≥25 g L⁻¹)")

plt.xlabel("Pectina [g L⁻¹]", fontsize=12)
plt.ylabel("Residuales (observado - predicho)", fontsize=12)
plt.title("Residuales del ajuste", fontsize=13)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
