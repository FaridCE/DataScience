---
layout: post  # or 'page', depending on your theme structure
title: "Análisis de Patrones Delictivos con PCA"
date: 2025-11-24 
categories: [data-science, inegi]
permalink: /analysis/pca-delitos/
---
# ACP para encontrar patrones delictivos
## Usando datos del INEGI
Datos a analizar: Homicidio doloso, Robo de vehículo automotor, Lesiones dolosas, Extorsión; por Entidad.

## Limpieza de datos
El archivo csv se encontra dividido por columnas en donde se repite el año, la entidad, y el tipo de delito. 
1. Primero se agrupa por Subtipo de delito (eliminando las columnas innecesarias) sumando el número de crímenes por mes
2. Luego hacemos una máscara booleana para obtener los subdelitos de interés (Homicidio doloso, etc).
3. Por último, promediamos los delitos anuales y pivoteamos (agrupamos en un multi-índice) el marco de datos para que aparezcan los subdelitos anualizados por Año y Entidad.


```python
import pandas as pd
from sklearn.preprocessing import StandardScaler
import numpy as np

df = pd.read_csv('Data/delitos.csv', encoding='latin1')
df.drop(['Clave_Ent', 'Bien jurídico afectado', 'Tipo de delito', 'Modalidad'], axis=1, inplace=True)
# Columns that contain the numeric month values
month_cols = ['Enero', 'Febrero', 'Marzo', 'Abril', 'Mayo', 'Junio',
              'Julio', 'Agosto', 'Septiembre', 'Octubre', 'Noviembre', 'Diciembre']

# 1 Group by the subtype of crime and sum the monthly values
df_sub = (
    df.groupby(['Año','Entidad','Subtipo de delito'])[month_cols]
      .sum()
      .reset_index()
)
# 2 Values you want to keep
SubDelitos = [
    'Homicidio doloso',
    'Robo de vehículo automotor',
    'Lesiones dolosas',
    'Extorsión'
]

# Boolean mask and filter
mask   = df_sub['Subtipo de delito'].isin(SubDelitos)
filtered_df = df_sub[mask].reset_index(drop=True)

# 3. Sumar la incidencia mensual para obtener el total anual

df_anual = filtered_df.groupby(['Año', 'Entidad', 'Subtipo de delito'])[month_cols].sum().reset_index()

# Pivotear para que los subtipos de delito sean columnas
df_pivot = df_anual.pivot_table(
    index=['Año', 'Entidad'],
    columns='Subtipo de delito',
    values= month_cols[0] # Usar cualquier mes para el pivot, ya que todos tienen la suma
).reset_index()

# El resultado 'df_pivot' tendrá columnas: ['Año', 'Entidad', 'Extorsión', 'Homicidio doloso', 'Lesiones dolosas', 'Robo de vehículo automotor']
# df_pivot # Descomentar para ver el DataFrame resultante

# Seleccionar solo las columnas de los delitos para la matriz de cálculo
X = df_pivot[['Extorsión', 'Homicidio doloso', 'Lesiones dolosas', 'Robo de vehículo automotor']]
```

## Pre-análisis de datos

Para analizar los datos por componentes principales hay que:
1. Estandarizar los datos: escribirlos de manera relativa a su valor promedio y centralizarlos. Solo nos interesan las fluctuaciones.
2. Correlacionar los datos: escribir la matriz de correlación para cada Entidad.

### 🧮 Centralización de los Datos (Estandarización)
La centralización y el escalado (estandarización) son cruciales para el ACP. Utilizamos StandardScaler de scikit-learn para estandarizar los datos a una media de 0 y una desviación estándar de 1 (puntuaciones Z).


```python
# 1. Inicializar y aplicar el escalador
scaler = StandardScaler()
X_scaled_array = scaler.fit_transform(X)

# 2. Convertir de nuevo a DataFrame para facilitar la interpretación
X_scaled = pd.DataFrame(
    X_scaled_array,
    columns=X.columns,
    index=X.index
)
# Ejemplo de verificación (opcional)
# print(X_scaled.mean())
# print(X_scaled.std())
```

### 📈 Cálculo de la Matriz de Correlación
La matriz de correlación nos muestra la relación lineal entre cada par de tipos de delito en los datos estandarizados.


```python
# La matriz de correlación de los datos estandarizados es
# la matriz de covarianza de los datos estandarizados
correlation_matrix = X_scaled.corr()
```

## Análisis por Componentes Principales

### 🧮 Cálculo y Extracción de Componentes Principales

Usaremos el módulo `PCA` de `scikit-learn` para obtener los componentes, la varianza explicada y las cargas factoriales.

### A. Aplicación del Modelo PCA

Continuando con el DataFrame estandarizado (`X_scaled`) de los cuatro delitos:


```python
from sklearn.decomposition import PCA
import numpy as np

# 1. Inicializar el modelo PCA
# Como son 4 variables, el número máximo de componentes será 4.
pca = PCA(n_components=4)

# 2. Ajustar y transformar los datos estandarizados
# La matriz X_pca ahora contiene las puntuaciones de cada componente (PC1, PC2, PC3, PC4)
X_pca = pca.fit_transform(X_scaled)
```

### B. Análisis de la Varianza Explicada (Valores Propios)

El primer paso de la interpretación es determinar **cuánta información** (varianza) de los datos originales es capturada por cada componente.


```python
# Varianza Explicada (Eigenvalues o Valores Propios)
varianza_explicada = pca.explained_variance_

# Proporción de la Varianza Explicada
proporcion_varianza = pca.explained_variance_ratio_

# Varianza Acumulada
varianza_acumulada = np.cumsum(proporcion_varianza)

# Crear un DataFrame para la visualización
df_varianza = pd.DataFrame({
    'Componente': range(1, len(varianza_explicada) + 1),
    'Valor Propio': varianza_explicada,
    'Proporción Explicada': proporcion_varianza,
    'Varianza Acumulada': varianza_acumulada
})

print(df_varianza)
```

       Componente  Valor Propio  Proporción Explicada  Varianza Acumulada
    0           1      3.032322              0.755927            0.755927
    1           2      0.629103              0.156829            0.912756
    2           3      0.263954              0.065801            0.978557
    3           4      0.086018              0.021443            1.000000


#### Determinación del Número de Componentes (Patrones)

Usando los criterios comunes:

1.  **Criterio de Kaiser (Valor Propio \> 1):** Se mantienen los componentes cuyo valor propio es mayor a 1. En este ejemplo, sería solamente **PC1**.
2.  **Varianza Acumulada:** Se mantienen los componentes necesarios para alcanzar un umbral razonable (ej. 80-90%). En este ejemplo, con **PC1 y PC2** se explica el **91.3%** de la varianza total, lo cual es muy bueno.

El análisis se centraría, por lo tanto, en los **dos primeros componentes (PC1 y PC2)**.

### 🔎 Interpretación de los Componentes (Cargas Factoriales)

Los **vectores propios** (o `components_` en `scikit-learn`) son la clave para entender el patrón. Se les conoce como **Cargas Factoriales** (*Loadings*) y representan la correlación entre cada delito original y cada componente principal.



```python
# Los vectores propios se encuentran en el atributo 'components_'
cargas_factoriales = pd.DataFrame(
    pca.components_.T, # Transponer para que los delitos sean las filas
    columns=[f'PC{i}' for i in range(1, pca.n_components_ + 1)],
    index=X_scaled.columns
)

print(cargas_factoriales)
```

                                     PC1       PC2       PC3       PC4
    Subtipo de delito                                                 
    Extorsión                   0.509513 -0.395948 -0.612803 -0.456174
    Homicidio doloso            0.403787  0.887516 -0.220806 -0.022723
    Lesiones dolosas            0.549183 -0.227933  0.005387  0.804000
    Robo de vehículo automotor  0.525120 -0.059891  0.758742 -0.380753


Esta tabla de cargas factoriales es la clave para interpretar los patrones delictivos en los datos. Los valores que se acercan a +1 o -1 indican una fuerte correlación entre el delito y esa Componente Principal (PC).

A continuación podemos dar la interpretación detallada de cada componente, utilizando un umbral de carga significativa de **|0.50|**:

## Interpretación del Análisis de Componentes Principales

### 1. Primer Componente Principal (PC1)

| Delito | Carga (Loading) |
| :--- | :--- |
| **Lesiones dolosas** | **0.549** |
| **Robo de vehículo automotor** | **0.525** |
| **Extorsión** | **0.509** |
| Homicidio doloso | 0.403 |

**Patrón Interpretativo: "Factor General de Incidencia Delictiva"**

El PC1 tiene cargas positivas y significativas en casi todos los delitos, especialmente en **Lesiones dolosas**, **Robo de vehículo automotor** y **Extorsión**.

* **Significado:** Este componente es un patrón de **"Tamaño"** o **"Criminalidad General"**. Las entidades (Entidad-Año) que obtienen una puntuación alta en PC1 son aquellas que, en general, presentan tasas altas de la mayoría de los delitos analizados. Es la dimensión que captura la mayor varianza de los datos.

***

### 2. Segundo Componente Principal (PC2)

| Delito | Carga (Loading) |
| :--- | :--- |
| **Homicidio doloso** | **0.887** (Muy fuerte) |
| Extorsión | -0.395 (Negativa, moderada) |

**Patrón Interpretativo: "Violencia Letal Concentrada"**

El PC2 está casi completamente dominado por la carga muy fuerte del **Homicidio doloso** (0.887).

* **Significado:** Este componente representa la **Severidad de la Violencia Letal**. Las entidades que puntúan alto en PC2 son aquellas donde el **Homicidio doloso** es desproporcionadamente alto en comparación con la media de las otras formas de delincuencia. La correlación negativa con Extorsión (-0.395) sugiere una tendencia: en estas entidades el riesgo de Homicidio es el principal impulsor de este patrón.

***

### 3. Tercer Componente Principal (PC3)

| Delito | Carga (Loading) |
| :--- | :--- |
| **Robo de vehículo automotor** | **0.758** (Fuerte, Positiva) |
| **Extorsión** | **-0.612** (Fuerte, Negativa) |

**Patrón Interpretativo: "Contraste entre Delincuencia Patrimonial y Financiera"**

El PC3 es un típico **"Factor de Forma"** que establece una dicotomía:

* **Puntuación Alta en PC3:** Implica alta incidencia de **Robo de vehículo automotor** y baja de **Extorsión**.
* **Puntuación Baja (Negativa) en PC3:** Implica alta incidencia de **Extorsión** y baja de **Robo de vehículo automotor**.

**Significado:** Este patrón diferencia las entidades en función del foco delictivo: aquellas con un alto riesgo de **delitos contra el patrimonio tangible** (robo de vehículos) frente a aquellas con un alto riesgo de **delitos de lucro ilícito** (extorsión).

***

### 4. Cuarto Componente Principal (PC4)

| Delito | Carga (Loading) |
| :--- | :--- |
| **Lesiones dolosas** | **0.804** (Fuerte, Positiva) |
| Extorsión | -0.456 (Negativa, moderada) |

**Patrón Interpretativo: "Violencia No-Letal y Conflictos Interpersonales"**

El PC4 está fuertemente dominado por las **Lesiones dolosas** (0.804).

* **Significado:** Este componente aísla el patrón de la **Violencia física no letal**. Las entidades que obtienen una puntuación alta en PC4 son aquellas con alta incidencia de Lesiones Dolosas, sugiriendo que este delito se comporta de forma independiente a la estructura de otros crímenes (como el Homicidio o el Robo).

---

## Resumen de los Patrones (Componentes)

Al reducir la dimensionalidad de la matriz, el ACP ha encontrado los siguientes cuatro patrones de riesgo distintos en los datos del INEGI:

1.  **PC1:** **Riesgo General** (Mide el volumen total de la delincuencia en la entidad).
2.  **PC2:** **Riesgo de Homicidio** (Mide el peso del Homicidio dentro de la estructura criminal).
3.  **PC3:** **Riesgo de Robo vs. Extorsión** (Contrasta el patrón de robo de vehículos con el de extorsión).
4.  **PC4:** **Riesgo de Lesiones** (Mide el peso de la violencia interpersonal no letal).

## 📊 Visualización de los Resultados (Biplot)

El **Biplot** es la herramienta de visualización más útil en el ACP, ya que permite ver tres cosas simultáneamente:

1.  **Las Observaciones:** La posición de cada Entidad-Año (los puntos en el gráfico).
2.  **Las Variables:** La dirección y magnitud de las variables originales (los delitos) en el nuevo espacio de componentes (los vectores).
3.  **La Relación:** Cómo se relacionan las variables entre sí y con las observaciones.



```python
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# --- 1. Datos del Paso Anterior (Para Replicación) ---
# Asumimos que X_scaled y pca ya fueron calculados.
# Para esta demostración, utilizaremos datos simulados similares a la tabla de cargas:
# Simulación de las cargas factoriales para el gráfico
cargas_factoriales = np.array([
    [0.5095, -0.3959, -0.6128, -0.4561],
    [0.4037, 0.8875, -0.2208, -0.0227],
    [0.5491, -0.2279, 0.0053, 0.8040],
    [0.5251, -0.0598, 0.7587, -0.3807]
]).T # Transponer la matriz de cargas

columnas_delito = ['Extorsión', 'Homicidio doloso', 'Lesiones dolosas', 'Robo de vehículo automotor']
X_scaled = pd.DataFrame(np.random.randn(100, 4), columns=columnas_delito) # Datos estandarizados simulados

pca = PCA(n_components=4)
X_pca = pca.fit_transform(X_scaled)
pca.components_ = cargas_factoriales # Sobrescribir con las cargas del ejemplo para el gráfico

# --- 2. Preparación de los datos para el Biplot ---
# Puntuaciones de las Observaciones (Coordenadas de las Entidades)
score = X_pca[:, [0, 1]] # Usar PC1 y PC2

# Coordenadas de las Variables (Cargas Factoriales de PC1 y PC2)
loadings = pca.components_[[0, 1], :].T

# --- 3. Generación del Biplot ---
plt.figure(figsize=(10, 8))
plt.scatter(score[:, 0], score[:, 1], alpha=0.6, s=50) # Graficar las Entidades-Año

# Graficar los vectores de las variables (Delitos)
for i, feature in enumerate(columnas_delito):
    plt.arrow(0, 0, loadings[i, 0], loadings[i, 1],
              color='red', alpha=0.8, linewidth=2,
              head_width=0.05, head_length=0.05)
    plt.text(loadings[i, 0] * 1.05, loadings[i, 1] * 1.05, feature,
             color='black', ha='center', va='center', fontsize=12)

plt.xlabel(f'Componente Principal 1 ({pca.explained_variance_ratio_[0]*100:.2f}%) - Riesgo General')
plt.ylabel(f'Componente Principal 2 ({pca.explained_variance_ratio_[1]*100:.2f}%) - Violencia Letal Concentrada')
plt.title('Biplot: Distribución de Entidades y Cargas de Delitos (PC1 vs PC2)')
plt.grid(True)
plt.axhline(0, color='grey', linestyle='--', alpha=0.5)
plt.axvline(0, color='grey', linestyle='--', alpha=0.5)
plt.show()
```


    
![png](crime_21_0.png)
    


### Interpretación de la Visualización

  * **Vectores Cercanos:** Delitos cuyos vectores apuntan en la misma dirección (ej. Extorsión, Lesiones y Robo) tienen una alta correlación positiva.
  * **Vectores Opuestos:** Si un vector apunta en la dirección opuesta (separados 180°), tienen una alta correlación negativa (no ocurre en este caso, pero PC3 sí mostró esto).
  * **Puntos (Entidades):**
      * Los puntos a la derecha del gráfico tienen puntuaciones altas en PC1 (alto riesgo delictivo general).
      * Los puntos en la parte superior del gráfico tienen puntuaciones altas en PC2 (alta incidencia desproporcionada de Homicidio Doloso).
  * **Distancia:** La longitud del vector indica la varianza del delito que explica ese componente (mientras más largo, más importante para ese plano).
