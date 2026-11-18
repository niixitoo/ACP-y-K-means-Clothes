# Perfiles de producto en el catálogo de verano de Wish

Taller individual de ACP y K-Means · Machine Learning aplicado a Economía y Finanzas
Pontificia Universidad Javeriana Cali · Facultad de Ciencias Económicas y Administrativas

**Autor:** George David Muñoz Rodríguez

- **Aplicación desplegada:** _pega aquí el enlace de Streamlit Community Cloud_
- **Video de sustentación:** _pega aquí el enlace del video_

## Contexto y pregunta

Wish es un marketplace de precios bajos en el que miles de vendedores compiten con productos muy parecidos.
Este proyecto explora qué distingue a unos productos de otros y busca responder dos preguntas:

1. ¿Cuáles son las dimensiones que explican las diferencias entre productos?
2. ¿Existen perfiles de producto reconocibles, o el catálogo es un continuo sin grupos claros?

La primera se aborda con Análisis de Componentes Principales (ACP) y la segunda con K-Means.

**Por qué estas técnicas son pertinentes.** Varias variables miden casi lo mismo (unidades vendidas y número de
reseñas; precio y costo de envío), así que el ACP permite resumir esa redundancia en pocas dimensiones con
significado. La base no trae ninguna categoría de desempeño, por lo que K-Means sirve para explorar si aparecen
perfiles sin imponerlos de antemano, y la silueta indica qué tan separados están.

## Datos

- **Fuente:** [Sales of summer clothes in E-commerce Wish](https://www.kaggle.com/datasets/jmmvutu/summer-products-and-sales-in-ecommerce-wish),
  publicada en Kaggle por Jeffrey Mvutu Mabilama.
- **Archivo:** `data/summer-products-with-rating-and-performance_2020-08.csv` (1.573 filas, 43 columnas).
- **Observación:** cada fila es un producto que aparecía en Wish al buscar la palabra *summer*.
- **Cobertura:** una sola captura, de agosto de 2020, con precios en euros.
- **Licencia:** [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). El archivo se incluye sin modificaciones.

El archivo está incluido en el repositorio. También se puede descargar desde el enlace de Kaggle y guardar en la
carpeta `data/` con el mismo nombre.

### Variables del modelo

| Variable | Qué mide | Logaritmo |
|---|---|---|
| `price` | Precio de venta (EUR) | Sí |
| `retail_price` | Precio de referencia que Wish muestra tachado (EUR) | Sí |
| `units_sold` | Unidades vendidas, reportadas en escalones | Sí |
| `rating` | Calificación promedio del producto (1 a 5) | No |
| `rating_count` | Número de reseñas del producto | Sí |
| `merchant_rating` | Calificación promedio del vendedor (1 a 5) | No |
| `merchant_rating_count` | Número de reseñas del vendedor | Sí |
| `shipping_option_price` | Costo de envío (EUR) | No |
| `countries_shipped_to` | Países a los que se despacha | No |
| `product_variation_inventory` | Inventario por variación (tope de 50) | No |

## Metodología

1. **Limpieza.**
   - Se eliminan 232 filas con `product_id` repetido (quedan 1.341 productos).
   - Se excluyen 35 productos sin reseñas: tienen `rating_count = 0` pero `rating = 5,0`, un valor de relleno.
   - Quedan **1.306 productos**.
2. **Selección de variables.** Se descartan identificadores, texto, columnas constantes o casi constantes, variables
   binarias y el desglose de reseñas por estrellas (correlación de 0,82 a 1,00 con `rating_count`).
3. **Transformación.** `log(1 + x)` en las variables de cola larga, para que unos pocos productos con valores
   enormes no dominen el análisis.
4. **Estandarización.** `StandardScaler` (media 0, desviación 1), necesaria porque ACP y K-Means dependen de escalas
   y distancias.
5. **ACP.** Varianza explicada, gráfico de sedimentación, cargas y criterio de Kaiser para elegir componentes.
6. **K-Means.** Se ejecuta sobre los componentes conservados. K se compara de 2 a 10 con el método del codo y el
   coeficiente de silueta (`n_init = 20`, semilla 42).
7. **Interpretación.** Los grupos se describen con las medianas de las variables originales.

## Resultados principales

Con la configuración recomendada (10 variables, 3 componentes, K = 4, semilla 42):

| Componente | Varianza | Acumulada | Variables que más pesan | Lectura |
|---|---|---|---|---|
| CP1 | 26,9 % | 26,9 % | Reseñas, unidades vendidas, reseñas del vendedor | Tracción comercial |
| CP2 | 21,6 % | 48,4 % | Precio, costo de envío, precio de referencia | Nivel de precio |
| CP3 | 12,4 % | 60,9 % | Países de envío, calificación del vendedor y del producto | Reputación y alcance |

Se conservan tres componentes: el cuarto tiene un valor propio de 1,00 y agregarlo reduce la silueta
(de 0,266 a 0,229 con K = 4).

| Grupo | Productos | Precio | Unidades vendidas | Reseñas | Calif. producto | Reseñas del vendedor | Perfil |
|---|---|---|---|---|---|---|---|
| A | 494 (37,8 %) | 8,00 | 10.000 | 1.337 | 3,90 | 18.029 | Productos consolidados |
| B | 353 (27,0 %) | 11,00 | 1.000 | 124 | 3,96 | 6.383 | Gama alta del catálogo |
| C | 303 (23,2 %) | 5,00 | 100 | 48 | 3,79 | 6.187 | Económicos de baja rotación |
| D | 156 (11,9 %) | 8,00 | 100 | 18 | 3,42 | 276 | Vendedores pequeños con baja reputación |

Valores en medianas. Las letras son solo nombres y no indican orden.

- **Silueta promedio: 0,266.** Los grupos se solapan: son zonas de un continuo, no categorías nítidas.
- El mejor valor de silueta está en K = 2 (0,295), que solo separa productos con tracción de productos sin ella.
  Se eligió K = 4 porque pierde 0,03 de silueta y entrega perfiles más útiles.
- El 2,8 % de los productos tiene silueta negativa (más cerca de otro grupo que del suyo).
- **Hallazgo central:** precio, tracción y reputación son dimensiones casi independientes. La correlación de
  Spearman del precio con las unidades vendidas es 0,02 y con la calificación es 0,05.

## Limitaciones

- Corte transversal de un solo mes: los resultados describen asociaciones, no causas.
- `units_sold` viene en escalones (100, 1.000, 5.000...), no como conteo exacto.
- Solo incluye productos que aparecían al buscar *summer*: no representa todo Wish.
- `retail_price` es poco confiable: en muchos productos es menor que el precio de venta.
- Tres componentes dejan por fuera el 39,1 % de la variación original.
- K-Means depende de las variables elegidas y de la transformación; otras decisiones producen otros grupos.
- La captura es de 2020.

## Cómo ejecutar el proyecto

Requiere Python 3.10 o superior.

```bash
git clone <URL-de-este-repositorio>
cd <carpeta-del-repositorio>
pip install -r requirements.txt
streamlit run app.py
```

La aplicación se abre en `http://localhost:8501`.

## Estructura del repositorio

```
├── app.py                 # Aplicación: carga, limpieza, ACP, K-Means y visualizaciones
├── requirements.txt       # Dependencias
├── README.md
├── .streamlit/config.toml # Tema visual
└── data/
    └── summer-products-with-rating-and-performance_2020-08.csv
```

Todas las tablas, métricas y gráficos se calculan en `app.py` cada vez que se abre la aplicación o se cambia un
parámetro en la barra lateral (variables, limpieza, logaritmo, número de componentes, K y semilla). No hay
resultados fijos.

| Función en `app.py` | Qué genera |
|---|---|
| `clean_data` | Base limpia y bitácora de pasos de limpieza |
| `transform` | Variables con transformación logarítmica |
| `fit_pca` | Puntajes, cargas, valores propios y varianza explicada |
| `evaluate_k` | Inercia y silueta para cada K (codo y silueta) |
| `fit_kmeans` | Grupo de cada producto y su silueta individual |
| `describe_groups` | Descripción de cada grupo frente al promedio |
