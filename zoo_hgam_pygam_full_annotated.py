import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pygam import LinearGAM, s
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_squared_error

sns.set(style="whitegrid")

# ----------------------
# 1) CARGAR EL CSV
# ----------------------
df = pd.read_csv("zooplankton_example.csv")
# Mostrar columnas para verificar
print("Columnas del CSV:", df.columns.values)

# ----------------------
# 2) ASEGURAR TRANSFORMACIONES
# ----------------------
# Si density_adj no existe, crearlo de la forma (density + 1000)/10000
if 'density_adj' not in df.columns:
    df['density'] = df['density'].fillna(0)   # reemplazar NA por 0 (procedimiento del repo original)
    df['density_adj'] = (df['density'] + 1000) / 10000

# Para evitar issues con log(0), usaremos un pequeño epsilon
eps = 1e-8

# ----------------------
# 3) FILTRAR UN LAGO (ejemplo: Mendota)
# ----------------------
df_mend = df[df['lake'] == 'Mendota'].copy()
print("Observaciones Mendota:", df_mend.shape[0])

# ----------------------
# 4) TRAIN / TEST por año: pares -> train, impares -> test (práctica tomada del artículo)
# ----------------------
df_mend['year'] = df_mend['year'].astype(int)
train = df_mend[df_mend['year'] % 2 == 0].copy()
test  = df_mend[df_mend['year'] % 2 == 1].copy()
print("Train:", train.shape, "Test:", test.shape)

# ----------------------
# 5) DISEÑO: crear columnas day (global) + day*I(taxon==g) para cada taxon
# Esto aproxima s(day, by=taxon) / factor-smoother
# ----------------------
taxa = np.sort(train['taxon'].unique())
print("Taxa (orden):", taxa)

# OneHotEncoder para las categorías de taxon (garantiza mismo orden)
enc = OneHotEncoder(sparse_output=False, drop=None) #convierte taxón en matriz
oh_train = enc.fit_transform(train[['taxon']])  # matriz densa (n_train x n_taxa)

# Construir matriz X_train:
# columna 0: day (global)
# columnas 1..G: day * I(taxon==g)
X_parts = [train['day'].values.reshape(-1,1)] # columna global
for j in range(oh_train.shape[1]):
    X_parts.append((train['day'].values * oh_train[:, j]).reshape(-1,1))
X_train = np.hstack(X_parts) #concatena columnas
print("X_train shape:", X_train.shape)

# Respuesta: usamos log(density_adj) -> modelo en escala log (aprox log-link)
y_train = np.log(train['density_adj'].values + eps)

# ----------------------
# 6) CONSTRUIR Y AJUSTAR LinearGAM
# Término s(0) = spline para columna 0 (global), s(1..G) = splines por taxon
# ----------------------
terms = s(0, n_splines=25)  # spline global con flexibilidad moderada
for j in range(1, X_train.shape[1]):
    terms += s(j, n_splines=10)  # splines por taxon (menos nodos por evitar sobreajuste)

gam = LinearGAM(terms)  # crear el objeto
# gridsearch busca la penalización (lambda) óptima
gam.gridsearch(X_train, y_train)

# Resumen del ajuste
print("\nResumen del modelo (pyGAM):")
print(gam.summary())

# ----------------------
# 7) PREDICCIONES y GRÁFICAS por taxon
# ----------------------
x_grid = np.linspace(train['day'].min(), train['day'].max(), 200)
colors = sns.color_palette("tab10", n_colors=len(taxa))
plt.figure(figsize=(11,7))

for k, t in enumerate(taxa):
    # Construir Xnew para este taxón: activamos la columna correspondiente
    parts = [x_grid.reshape(-1,1)]
    for j in range(len(taxa)):
        if j == k:
            parts.append(x_grid.reshape(-1,1))
        else:
            parts.append(np.zeros_like(x_grid).reshape(-1,1))
    Xnew = np.hstack(parts)

    # Predecir en escala link (log)
    yhat_link = gam.predict(Xnew)
    # Intervalo de predicción (link-scale)
    yhat_int = gam.prediction_intervals(Xnew, width=0.95)

    # Transformar a escala original (densidad_adj): inverse of log is exp
    yhat_orig = np.exp(yhat_link) - 0.0  # restar 0 si queremos exactamente density_adj
    yhat_int_orig = np.exp(yhat_int) - 0.0

    # Graficar línea ajustada (en escala original para mejor interpretación visual)
    plt.plot(x_grid, yhat_orig, color=colors[k], label=f"{t} fitted")
    # Banda de confianza
    plt.fill_between(x_grid, yhat_int_orig[:,0], yhat_int_orig[:,1], alpha=0.15, color=colors[k])
    # puntos observados (train) en escala original
    subset = train[train['taxon'] == t]
    plt.scatter(subset['day'], subset['density_adj'], s=8, color=colors[k], alpha=0.25)

plt.xlabel("Day of year")
plt.ylabel("density_adj (escala original)")
plt.title("Curvas estacionales por taxon (ajustadas, pyGAM - aproximación)")
plt.legend(bbox_to_anchor=(1.05,1), loc='upper left')
plt.tight_layout()
plt.show()


plt.savefig("zoo_seasonal_curves_pygam.png", bbox_inches='tight', dpi=200)

# ----------------------
# 8) EVALUACIÓN out-of-sample (TEST)
# Construir X_test con la misma codificación (usar enc.transform)
# ----------------------
# Para test puede haber taxones en test que no estaban en train; manejamos eso:
# Vamos a transformar test[['taxon']] con enc; si aparece categoría desconocida -> error.
# Por simplicidad, filtramos observations con taxon no visto en train.
test_valid = test[test['taxon'].isin(taxa)].copy()
oh_test = enc.transform(test_valid[['taxon']])  # usa mismo orden de columnas que enc
X_test_parts = [test_valid['day'].values.reshape(-1,1)]
for j in range(oh_test.shape[1]):
    X_test_parts.append((test_valid['day'].values * oh_test[:, j]).reshape(-1,1))
X_test = np.hstack(X_test_parts)
y_test_link = np.log(test_valid['density_adj'].values + eps)

# Predecir en link-scale
y_pred_link = gam.predict(X_test)

# MSE en link-scale (lo que obtuviste)
mse_link = mean_squared_error(y_test_link, y_pred_link)
print("MSE (link-scale) out-of-sample:", mse_link)
print("RMSE (link-scale):", np.sqrt(mse_link))

# ----------------------
# 9) Métricas en escala original (retransformar)
# ----------------------
y_pred_orig = np.exp(y_pred_link)  # inverse of log
y_test_orig = test_valid['density_adj'].values

mse_orig = mean_squared_error(y_test_orig, y_pred_orig)
rmse_orig = np.sqrt(mse_orig)
# MAPE (porcentaje medio absoluto)
mape = np.mean(np.abs((y_test_orig - y_pred_orig) / (y_test_orig + eps))) * 100

print("MSE (original scale):", mse_orig)
print("RMSE (original scale):", rmse_orig)
print("MAPE (%):", mape)

# ----------------------
# 10) Métricas por taxon (tabla)
# ----------------------
results = []
for t in taxa:
    tv = test_valid[test_valid['taxon'] == t]
    if tv.shape[0] == 0:
        continue
    oh_tv = enc.transform(tv[['taxon']])
    X_tv_parts = [tv['day'].values.reshape(-1,1)]
    for j in range(oh_tv.shape[1]):
        X_tv_parts.append((tv['day'].values * oh_tv[:, j]).reshape(-1,1))
    X_tv = np.hstack(X_tv_parts)
    y_link_pred = gam.predict(X_tv)
    y_orig_pred = np.exp(y_link_pred)
    y_orig_true = tv['density_adj'].values
    mse_t = mean_squared_error(y_orig_true, y_orig_pred)
    rmse_t = np.sqrt(mse_t)
    results.append((t, tv.shape[0], mse_t, rmse_t))
res_df = pd.DataFrame(results, columns=['taxon','n_obs_test','mse_orig','rmse_orig'])
print(res_df.sort_values('rmse_orig', ascending=False).head(20))

res_df.to_csv("metrics_by_taxon.csv", index=False)
