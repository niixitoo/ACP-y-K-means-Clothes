"""
Taller individual: ACP y K-Means
Machine Learning aplicado a Economía y Finanzas - Pontificia Universidad Javeriana Cali

Caso de estudio: perfiles de producto en el catálogo de verano de Wish (agosto de 2020).

Todo lo que muestra la aplicación (tablas, métricas y gráficos) se calcula en vivo
a partir del CSV de la carpeta data/. No hay resultados fijos ni capturas.

Ejecutar en local:  streamlit run app.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.preprocessing import StandardScaler

# --------------------------------------------------------------------------------------
# Configuración general
# --------------------------------------------------------------------------------------

st.set_page_config(page_title="Perfiles de producto en Wish | ACP y K-Means", page_icon="🛍️", layout="wide")

DATA_PATH = Path(__file__).parent / "data" / "summer-products-with-rating-and-performance_2020-08.csv"
KAGGLE_URL = "https://www.kaggle.com/datasets/jmmvutu/summer-products-and-sales-in-ecommerce-wish"

# Variables cuantitativas candidatas: etiqueta, descripción y si se transforman con logaritmo
VARIABLES = {
    "price": ("Precio de venta (EUR)", "Precio al que se ofrece el producto.", True),
    "retail_price": ("Precio de referencia (EUR)", "Precio 'de mercado' que Wish muestra tachado junto al precio de venta.", True),
    "units_sold": ("Unidades vendidas", "Ventas acumuladas, reportadas por Wish en escalones (100, 1.000, 5.000...).", True),
    "rating": ("Calificación del producto", "Promedio de estrellas del producto (1 a 5).", False),
    "rating_count": ("Número de reseñas", "Cantidad de calificaciones recibidas por el producto.", True),
    "merchant_rating": ("Calificación del vendedor", "Promedio de estrellas del vendedor (1 a 5).", False),
    "merchant_rating_count": ("Reseñas del vendedor", "Cantidad de calificaciones del vendedor: aproxima su tamaño y trayectoria.", True),
    "shipping_option_price": ("Costo de envío (EUR)", "Precio de la opción de envío mostrada.", False),
    "countries_shipped_to": ("Países de envío", "Número de países a los que se despacha el producto.", False),
    "product_variation_inventory": ("Inventario por variación", "Unidades disponibles por variación (tope de 50 en la plataforma).", False),
}
DEFAULT_VARS = list(VARIABLES.keys())
DEFAULT_SEED = 42

# Paleta: tonos categóricos en orden fijo (el color sigue al grupo, no a su posición)
CATEGORICAL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
DIVERGING = [[0.0, "#d95926"], [0.5, "#f0efec"], [1.0, "#1c5cab"]]
BLUE = "#2a78d6"
INK, INK_2, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#fcfcfb"
LETTERS = "ABCDEFGH"


def label(col: str) -> str:
    return VARIABLES[col][0] if col in VARIABLES else col


def low(col: str) -> str:
    """Etiqueta en minúscula para usar dentro de una frase (conserva 'EUR')."""
    return label(col).lower().replace("(eur)", "(EUR)")


def num(x: float, d: int = 1) -> str:
    """Formato numérico en español: punto para miles y coma decimal."""
    return f"{x:,.{d}f}".replace(",", "§").replace(".", ",").replace("§", ".")


def style(fig: go.Figure, title: str, x: str = "", y: str = "", height: int = 420) -> go.Figure:
    fig.update_layout(
        title=dict(text=title, font=dict(size=16, color=INK), x=0, xanchor="left"),
        paper_bgcolor=SURFACE,
        plot_bgcolor=SURFACE,
        font=dict(color=INK_2, size=13),
        height=height,
        margin=dict(l=10, r=10, t=60, b=10),
        legend=dict(title_text="", orientation="h", yanchor="bottom", y=1.0, xanchor="right", x=1),
    )
    fig.update_xaxes(title_text=x, gridcolor=GRID, linecolor="#c3c2b7", zeroline=False, title_font=dict(color=INK_2))
    fig.update_yaxes(title_text=y, gridcolor=GRID, linecolor="#c3c2b7", zeroline=False, title_font=dict(color=INK_2))
    return fig


def show(fig: go.Figure) -> None:
    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})


# --------------------------------------------------------------------------------------
# Datos y preparación
# --------------------------------------------------------------------------------------


@st.cache_data
def load_data() -> pd.DataFrame:
    return pd.read_csv(DATA_PATH)


def clean_data(df: pd.DataFrame, drop_dupes: bool, drop_no_reviews: bool) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Aplica las decisiones de calidad y devuelve la base limpia y la bitácora de pasos."""
    steps = [{"Paso": "Base original", "Filas": len(df), "Filas retiradas": 0, "Motivo": "Archivo tal como se descarga de Kaggle."}]
    out = df.copy()
    if drop_dupes:
        before = len(out)
        out = out.drop_duplicates(subset="product_id", keep="first")
        steps.append({
            "Paso": "Eliminar productos repetidos",
            "Filas": len(out),
            "Filas retiradas": before - len(out),
            "Motivo": "El mismo product_id aparece varias veces; repetirlo le daría más peso a esos productos.",
        })
    if drop_no_reviews:
        before = len(out)
        out = out[out["rating_count"] > 0]
        steps.append({
            "Paso": "Excluir productos sin reseñas",
            "Filas": len(out),
            "Filas retiradas": before - len(out),
            "Motivo": "Tienen 0 reseñas pero calificación 5,0: ese valor es un relleno, no una opinión real.",
        })
    return out.reset_index(drop=True), pd.DataFrame(steps)


def transform(data: pd.DataFrame, cols: list[str], use_log: bool) -> pd.DataFrame:
    """Logaritmo (log(1 + x)) para las variables de cola larga; el resto queda igual."""
    X = data[cols].astype(float).copy()
    if use_log:
        for c in cols:
            if VARIABLES[c][2]:
                X[c] = np.log1p(X[c])
    return X


@st.cache_data
def fit_pca(Z: np.ndarray):
    pca = PCA().fit(Z)
    return pca.transform(Z), pca.components_, pca.explained_variance_, pca.explained_variance_ratio_


@st.cache_data
def evaluate_k(S: np.ndarray, k_max: int, seed: int) -> pd.DataFrame:
    """Inercia (codo) y silueta promedio para cada K."""
    rows = []
    for k in range(2, k_max + 1):
        km = KMeans(n_clusters=k, n_init=20, random_state=seed).fit(S)
        rows.append({"K": k, "Inercia": km.inertia_, "Silueta": silhouette_score(S, km.labels_)})
    return pd.DataFrame(rows)


@st.cache_data
def fit_kmeans(S: np.ndarray, k: int, seed: int):
    km = KMeans(n_clusters=k, n_init=20, random_state=seed).fit(S)
    # Las etiquetas numéricas de K-Means son arbitrarias: se renombran con letras,
    # asignadas de mayor a menor tamaño solo para que el nombre sea estable.
    order = np.argsort(-np.bincount(km.labels_, minlength=k))
    mapping = {old: LETTERS[new] for new, old in enumerate(order)}
    groups = np.array([f"Grupo {mapping[l]}" for l in km.labels_])
    return groups, silhouette_samples(S, km.labels_)


def describe_groups(Zdf: pd.DataFrame, groups: np.ndarray, threshold: float = 0.4) -> dict[str, str]:
    """Describe cada grupo con las variables en las que más se aleja del promedio general."""
    means = Zdf.groupby(groups).mean()
    out = {}
    for g, row in means.iterrows():
        top = row[row.abs() >= threshold].sort_values(key=np.abs, ascending=False).head(4)
        high = [low(c) for c, v in top.items() if v > 0]
        below = [low(c) for c, v in top.items() if v < 0]
        parts = []
        if high:
            parts.append("por encima del promedio en " + ", ".join(high))
        if below:
            parts.append("por debajo del promedio en " + ", ".join(below))
        out[g] = "; ".join(parts) if parts else "cercano al promedio general en todas las variables"
    return out


# --------------------------------------------------------------------------------------
# Barra lateral: navegación y parámetros (los algoritmos se recalculan con cada cambio)
# --------------------------------------------------------------------------------------

raw = load_data()

st.sidebar.title("🛍️ Perfiles de producto en Wish")
st.sidebar.caption("Taller individual · ACP y K-Means")
PAGES = ["1. Contexto", "2. Exploración de datos", "3. Preparación", "4. Componentes principales", "5. Segmentación K-Means", "6. Conclusiones"]
page = st.sidebar.radio("Sección", PAGES)

st.sidebar.divider()
st.sidebar.subheader("Parámetros del análisis")
sel_vars = st.sidebar.multiselect("Variables del modelo", DEFAULT_VARS, default=DEFAULT_VARS, format_func=label,
                                  help="Se necesitan al menos cuatro variables cuantitativas.")
opt_dupes = st.sidebar.checkbox("Eliminar productos repetidos", value=True)
opt_reviews = st.sidebar.checkbox("Excluir productos sin reseñas", value=True)
opt_log = st.sidebar.checkbox("Logaritmo en variables de cola larga", value=True,
                              help="Aplica log(1 + x) a precios, ventas y conteos de reseñas.")

if len(sel_vars) < 4:
    st.warning("Selecciona al menos cuatro variables en la barra lateral para ejecutar el análisis.")
    st.stop()

data, steps = clean_data(raw, opt_dupes, opt_reviews)
X = transform(data, sel_vars, opt_log)
Z = StandardScaler().fit_transform(X)
Zdf = pd.DataFrame(Z, columns=sel_vars)

scores_all, components, eigenvalues, var_ratio = fit_pca(Z)
cum_ratio = var_ratio.cumsum()
kaiser = max(2, int((eigenvalues > 1).sum()))

n_comp = st.sidebar.slider("Componentes a conservar", 2, len(sel_vars), min(kaiser, len(sel_vars)),
                           key=f"ncomp_{len(sel_vars)}_{kaiser}",
                           help="Valor inicial: componentes con valor propio mayor que 1 (criterio de Kaiser).")
k = st.sidebar.slider("Número de grupos (K)", 2, 8, 4)
seed = st.sidebar.number_input("Semilla aleatoria", 0, 9999, DEFAULT_SEED, help="Fija la inicialización de K-Means para que el resultado sea reproducible.")

S = np.ascontiguousarray(scores_all[:, :n_comp])
pc_names = [f"CP{i + 1}" for i in range(len(sel_vars))]
retained = cum_ratio[n_comp - 1] * 100

k_table = evaluate_k(S, 10, int(seed))
groups, sil_samples = fit_kmeans(S, k, int(seed))
group_names = sorted(set(groups))
color_map = {g: CATEGORICAL[LETTERS.index(g[-1])] for g in group_names}
sil_k = float(k_table.loc[k_table["K"] == k, "Silueta"].iloc[0])
best_k = int(k_table.loc[k_table["Silueta"].idxmax(), "K"])
descriptions = describe_groups(Zdf, groups)

is_default = (sel_vars == DEFAULT_VARS and opt_dupes and opt_reviews and opt_log and n_comp == 3 and k == 4 and seed == DEFAULT_SEED)

# --------------------------------------------------------------------------------------
# 1. Contexto
# --------------------------------------------------------------------------------------

if page == PAGES[0]:
    st.title("¿Qué perfiles de producto existen en el catálogo de verano de Wish?")
    st.markdown(
        """
Wish es un marketplace de precios bajos en el que miles de vendedores, casi todos desde China, compiten con
productos muy parecidos. Para alguien que administra una tienda de moda en línea, la pregunta práctica es
**qué distingue a unos productos de otros**: ¿el precio?, ¿la reputación del vendedor?, ¿las reseñas acumuladas?

Este proyecto usa aprendizaje no supervisado para responder dos preguntas:

1. **¿Cuáles son las dimensiones que explican las diferencias entre productos?** (Análisis de Componentes Principales)
2. **¿Existen perfiles de producto reconocibles, o el catálogo es un continuo sin grupos claros?** (K-Means)
"""
    )
    c1, c2, c3 = st.columns(3)
    c1.metric("Productos en la base original", num(len(raw), 0))
    c2.metric("Productos después de la limpieza", num(len(data), 0))
    c3.metric("Variables en el modelo", len(sel_vars))

    st.subheader("Los datos")
    st.markdown(
        f"""
- **Fuente:** [Sales of summer clothes in E-commerce Wish]({KAGGLE_URL}), publicada en Kaggle por Jeffrey Mvutu Mabilama.
- **Qué es cada observación:** un producto que aparecía en Wish al buscar la palabra *summer*.
- **Cobertura:** una sola captura, de agosto de 2020, con precios en euros. La mayoría son prendas y accesorios de mujer.
- **Tamaño:** {num(raw.shape[0], 0)} filas y {raw.shape[1]} columnas en el archivo original.
"""
    )
    st.dataframe(
        pd.DataFrame([{"Variable": c, "Nombre": v[0], "Qué mide": v[1], "Logaritmo": "Sí" if v[2] else "No"} for c, v in VARIABLES.items()]),
        hide_index=True, width="stretch",
    )

    st.subheader("Por qué ACP y K-Means")
    st.markdown(
        """
- **ACP:** varias variables miden casi lo mismo (las unidades vendidas y el número de reseñas se mueven juntas, igual que
  el precio y el costo de envío). El ACP resume esa redundancia en pocas dimensiones y permite ponerles nombre.
- **K-Means:** la base no trae ninguna categoría que clasifique los productos por desempeño. K-Means permite explorar
  si aparecen perfiles sin imponerlos de antemano, y la silueta indica qué tan reales son esos perfiles.
- **En conjunto:** agrupar sobre los componentes, y no sobre las diez variables, evita que las variables redundantes
  pesen doble en el cálculo de distancias.
"""
    )
    st.subheader("Limitaciones de la fuente")
    st.markdown(
        """
- Es una **foto de un mes y de una sola búsqueda**: no representa todo Wish ni el mercado actual.
- Las **unidades vendidas vienen en escalones**, no como conteo exacto.
- El **precio de referencia no es confiable**: en muchos productos es menor que el precio de venta.
- Los datos se obtuvieron por *scraping*; no hay documentación oficial de Wish sobre cómo se calcula cada campo.
"""
    )

# --------------------------------------------------------------------------------------
# 2. Exploración
# --------------------------------------------------------------------------------------

elif page == PAGES[1]:
    st.title("Exploración de datos")
    st.caption("Esta sección describe la base original, antes de cualquier limpieza.")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Filas", num(len(raw), 0))
    c2.metric("Columnas", raw.shape[1])
    c3.metric("Productos repetidos", num(raw.duplicated("product_id").sum(), 0))
    c4.metric("Columnas con faltantes", int((raw.isna().sum() > 0).sum()))

    st.subheader("Estructura de la base")
    structure = pd.DataFrame({
        "Columna": raw.columns,
        "Tipo": ["Numérica" if pd.api.types.is_numeric_dtype(raw[c]) else "Texto" for c in raw.columns],
        "Faltantes": raw.isna().sum().values,
        "Valores distintos": raw.nunique().values,
    })
    st.dataframe(structure, hide_index=True, width="stretch", height=280)
    with st.expander("Ver las primeras filas"):
        st.dataframe(raw.head(20), width="stretch")

    st.subheader("Estadísticas descriptivas de las variables del modelo")
    desc = raw[sel_vars].describe().T[["mean", "std", "min", "50%", "max"]]
    desc["Asimetría"] = raw[sel_vars].skew()
    desc.index = [label(c) for c in desc.index]
    desc.columns = ["Media", "Desv. estándar", "Mínimo", "Mediana", "Máximo", "Asimetría"]
    st.dataframe(desc.round(2), width="stretch")
    st.info("Una asimetría muy por encima de 1 indica cola larga: pocos productos con valores enormes. "
            "Por eso esas variables entran al modelo en logaritmo.")

    st.subheader("Distribución de cada variable")
    c1, c2 = st.columns([2, 1])
    var = c1.selectbox("Variable", sel_vars, format_func=label)
    as_log = c2.toggle("Ver en logaritmo", value=False)
    serie = np.log1p(raw[var]) if as_log else raw[var]
    fig = go.Figure(go.Histogram(x=serie, nbinsx=40, marker=dict(color=BLUE, line=dict(color=SURFACE, width=1)),
                                 hovertemplate="Valor: %{x}<br>Productos: %{y}<extra></extra>"))
    show(style(fig, f"Distribución de {low(var)}", f"{label(var)}{' (log(1 + x))' if as_log else ''}", "Número de productos", 380))

    st.subheader("Matriz de correlaciones")
    corr = raw[sel_vars].corr(method="spearman")
    names = [label(c) for c in sel_vars]
    fig = go.Figure(go.Heatmap(z=corr.values, x=names, y=names, zmin=-1, zmax=1, colorscale=DIVERGING,
                               text=corr.round(2).values, texttemplate="%{text}", textfont=dict(size=11),
                               xgap=2, ygap=2, colorbar=dict(title="ρ"),
                               hovertemplate="%{y} y %{x}<br>Correlación: %{z:.2f}<extra></extra>"))
    fig.update_yaxes(autorange="reversed")
    show(style(fig, "Correlación de Spearman entre las variables del modelo", height=560))
    st.markdown(
        "Se usa Spearman porque las variables son muy asimétricas. Las correlaciones altas (azul intenso) señalan "
        "**información redundante**, que es justamente lo que el ACP puede resumir en un solo componente."
    )

# --------------------------------------------------------------------------------------
# 3. Preparación
# --------------------------------------------------------------------------------------

elif page == PAGES[2]:
    st.title("Preparación de los datos")
    st.markdown("Cada decisión se puede activar o desactivar en la barra lateral para ver cómo cambia el resultado.")

    st.subheader("1. Decisiones de calidad")
    st.dataframe(steps, hide_index=True, width="stretch")
    st.metric("Productos que entran al análisis", num(len(data), 0))
    st.caption(f"{num(len(raw) - len(data), 0)} filas menos que la base original.")

    st.subheader("2. Variables que se dejan por fuera")
    st.markdown(
        """
| Columnas | Motivo |
|---|---|
| `product_id`, `merchant_id`, URLs, títulos, `tags` | Identificadores y texto: no miden nada sobre el producto. |
| `currency_buyer`, `theme`, `crawl_month` | Tienen un único valor en toda la base. |
| `inventory_total`, `shipping_is_express` | Casi constantes: no aportan variación. |
| `rating_five_count` ... `rating_one_count` | Correlación casi perfecta con el número de reseñas; repetirían la misma información. |
| `has_urgency_banner`, insignias, `uses_ad_boosts` | Variables binarias; K-Means trabaja con distancias sobre variables continuas. |
"""
    )

    st.subheader("3. Transformación logarítmica")
    logged = [label(c) for c in sel_vars if VARIABLES[c][2]]
    if opt_log:
        st.markdown("Se aplica `log(1 + x)` a: **" + ", ".join(logged) + "**. Así un producto con 100.000 ventas no domina por sí solo el análisis.")
    else:
        st.warning("La transformación logarítmica está desactivada: los valores extremos pesarán mucho en el ACP y en K-Means.")
    comp = pd.DataFrame({"Antes": data[sel_vars].skew(), "Después": X.skew()})
    comp.index = [label(c) for c in comp.index]
    fig = go.Figure()
    fig.add_bar(y=comp.index, x=comp["Antes"], name="Antes", orientation="h", marker_color="#86b6ef",
                hovertemplate="%{y}<br>Asimetría antes: %{x:.2f}<extra></extra>")
    fig.add_bar(y=comp.index, x=comp["Después"], name="Después", orientation="h", marker_color="#184f95",
                hovertemplate="%{y}<br>Asimetría después: %{x:.2f}<extra></extra>")
    fig.update_layout(barmode="group", bargap=0.3)
    fig.update_yaxes(autorange="reversed")
    show(style(fig, "Asimetría de cada variable antes y después de la transformación", "Coeficiente de asimetría (0 = simétrica)", "", 440))

    st.subheader("4. Estandarización")
    st.markdown(
        "Las variables están en unidades distintas (euros, estrellas, unidades, países). ACP y K-Means dependen de "
        "varianzas y distancias, así que cada variable se lleva a **media 0 y desviación estándar 1** con `StandardScaler`."
    )
    check = pd.DataFrame({"Media original": X.mean(), "Desv. original": X.std(), "Media estandarizada": Zdf.mean(), "Desv. estandarizada": Zdf.std(ddof=0)})
    check.index = [label(c) for c in check.index]
    st.dataframe(check.round(3), width="stretch")

# --------------------------------------------------------------------------------------
# 4. ACP
# --------------------------------------------------------------------------------------

elif page == PAGES[3]:
    st.title("Análisis de Componentes Principales")
    c1, c2, c3 = st.columns(3)
    c1.metric("Componentes conservados", n_comp)
    c2.metric("Varianza conservada", f"{num(retained)} %")
    c3.metric("Componentes con valor propio > 1", int((eigenvalues > 1).sum()))

    st.subheader("Varianza explicada")
    var_table = pd.DataFrame({"Componente": pc_names, "Valor propio": eigenvalues, "Varianza (%)": var_ratio * 100, "Acumulada (%)": cum_ratio * 100})
    left, right = st.columns([1, 1.4])
    left.dataframe(var_table.round(2), hide_index=True, width="stretch", height=390)
    fig = go.Figure()
    fig.add_bar(x=pc_names, y=var_ratio * 100, name="Por componente",
                marker_color=[BLUE if i < n_comp else "#b7d3f6" for i in range(len(pc_names))],
                hovertemplate="%{x}<br>Explica: %{y:.1f} %<extra></extra>")
    fig.add_scatter(x=pc_names, y=cum_ratio * 100, name="Acumulada", mode="lines+markers",
                    line=dict(color=INK_2, width=2), marker=dict(size=8),
                    hovertemplate="%{x}<br>Acumulada: %{y:.1f} %<extra></extra>")
    fig.update_yaxes(range=[0, 105])
    with right:
        show(style(fig, "Gráfico de sedimentación", "Componente principal", "Varianza explicada (%)", 390))
    st.markdown(
        f"Las barras oscuras son los componentes conservados. Con **{n_comp} componentes** se retiene el "
        f"**{num(retained)} %** de la variación original; el {num(100 - retained)} % restante se pierde al resumir. "
        f"El criterio de Kaiser (valor propio mayor que 1) sugiere {int((eigenvalues > 1).sum())}: un componente con valor propio "
        "menor que 1 resume menos información que una sola variable original."
    )

    st.subheader("Cargas: qué significa cada componente")
    load = pd.DataFrame(components[:n_comp].T, index=[label(c) for c in sel_vars], columns=pc_names[:n_comp])
    load_low = dict(zip(load.index, [low(c) for c in sel_vars]))
    lim = float(np.abs(load.values).max())
    fig = go.Figure(go.Heatmap(z=load.values, x=load.columns, y=load.index, zmin=-lim, zmax=lim, colorscale=DIVERGING,
                               text=load.round(2).values, texttemplate="%{text}", xgap=2, ygap=2, colorbar=dict(title="Carga"),
                               hovertemplate="%{y} en %{x}<br>Carga: %{z:.2f}<extra></extra>"))
    fig.update_yaxes(autorange="reversed")
    show(style(fig, "Cargas de las variables en los componentes conservados", height=460))
    st.markdown("Una carga alta (positiva o negativa) significa que la variable pesa mucho en ese componente. "
                "El signo indica en qué dirección se mueve.")
    for i in range(n_comp):
        col = load.iloc[:, i]
        strong = col[col.abs() >= 0.35].sort_values(key=np.abs, ascending=False)
        pos = [load_low[n] for n, v in strong.items() if v > 0]
        neg = [load_low[n] for n, v in strong.items() if v < 0]
        text = f"**{pc_names[i]} ({num(var_ratio[i] * 100)} % de la varianza):** aumenta con " + (", ".join(pos) if pos else "ninguna variable dominante")
        if neg:
            text += "; disminuye con " + ", ".join(neg)
        st.markdown("- " + text + ".")

    st.subheader("Los productos en el espacio de componentes")
    c1, c2 = st.columns(2)
    ax_x = c1.selectbox("Eje horizontal", pc_names[:n_comp], index=0)
    ax_y = c2.selectbox("Eje vertical", pc_names[:n_comp], index=1)
    ix, iy = pc_names.index(ax_x), pc_names.index(ax_y)
    hover = [f"{t[:60]}<br>Precio: {p} EUR · Ventas: {u:,}".replace(",", ".") for t, p, u in zip(data["title_orig"], data["price"], data["units_sold"])]
    fig = go.Figure(go.Scattergl(x=scores_all[:, ix], y=scores_all[:, iy], mode="markers", text=hover,
                                 marker=dict(color=BLUE, size=7, opacity=0.55, line=dict(color=SURFACE, width=0.5)),
                                 hovertemplate="%{text}<extra></extra>"))
    show(style(fig, f"Productos proyectados sobre {ax_x} y {ax_y}",
               f"{ax_x} ({num(var_ratio[ix] * 100)} % de la varianza)", f"{ax_y} ({num(var_ratio[iy] * 100)} % de la varianza)", 520))
    st.info(f"Este gráfico muestra solo dos dimensiones y resume el **{num((var_ratio[ix] + var_ratio[iy]) * 100)} %** de la "
            "información original. Dos productos que se ven cercanos aquí pueden estar separados en las dimensiones que no se ven.")

# --------------------------------------------------------------------------------------
# 5. K-Means
# --------------------------------------------------------------------------------------

elif page == PAGES[4]:
    st.title("Segmentación con K-Means")
    st.markdown(f"K-Means se ejecuta sobre los **{n_comp} componentes conservados** ({num(retained)} % de la varianza), "
                "no sobre las variables originales, para que la información redundante no pese doble en las distancias.")

    st.subheader("¿Cuántos grupos?")
    c1, c2 = st.columns(2)
    fig = go.Figure(go.Scatter(x=k_table["K"], y=k_table["Inercia"], mode="lines+markers", line=dict(color=BLUE, width=2),
                               marker=dict(size=[13 if v == k else 8 for v in k_table["K"]], color=BLUE),
                               hovertemplate="K = %{x}<br>Inercia: %{y:,.0f}<extra></extra>"))
    with c1:
        show(style(fig, "Método del codo", "Número de grupos (K)", "Inercia (suma de distancias al centro)", 360))
    fig = go.Figure(go.Scatter(x=k_table["K"], y=k_table["Silueta"], mode="lines+markers", line=dict(color=BLUE, width=2),
                               marker=dict(size=[13 if v == k else 8 for v in k_table["K"]], color=BLUE),
                               hovertemplate="K = %{x}<br>Silueta: %{y:.3f}<extra></extra>"))
    with c2:
        show(style(fig, "Coeficiente de silueta", "Número de grupos (K)", "Silueta promedio (de -1 a 1)", 360))
    st.caption("El punto grande marca el K seleccionado en la barra lateral.")
    kt = k_table.copy()
    drop = -kt["Inercia"].pct_change() * 100
    kt["Caída de inercia frente al K anterior"] = ["—" if pd.isna(v) else f"{num(v)} %" for v in drop]
    st.dataframe(kt.round({"Inercia": 1, "Silueta": 3}), hide_index=True, width="stretch")
    st.markdown(
        f"- **Codo:** la inercia siempre baja al aumentar K; interesa el punto donde deja de bajar con fuerza.\n"
        f"- **Silueta:** el valor más alto está en **K = {best_k}** ({num(k_table['Silueta'].max(), 3)}). "
        f"Con el K elegido (**{k}**) la silueta es **{num(sil_k, 3)}**.\n"
        "- **Lectura:** valores por debajo de 0,5 indican grupos que se solapan. Aquí ningún K produce grupos nítidos, "
        "así que la elección combina estas métricas con qué tan interpretables y útiles son los perfiles."
    )

    st.subheader(f"Los {k} grupos")
    sizes = pd.Series(groups).value_counts().reindex(group_names)
    cols = st.columns(k)
    for col, g in zip(cols, group_names):
        col.metric(g, num(sizes[g], 0))
        col.caption(f"{num(sizes[g] / len(groups) * 100)} % de los productos")
    st.caption("Las letras son solo nombres: no indican orden, calidad ni jerarquía entre grupos.")

    c1, c2 = st.columns(2)
    ax_x = c1.selectbox("Eje horizontal", pc_names[:n_comp], index=0, key="kx")
    ax_y = c2.selectbox("Eje vertical", pc_names[:n_comp], index=1, key="ky")
    ix, iy = pc_names.index(ax_x), pc_names.index(ax_y)
    fig = go.Figure()
    for g in group_names:
        m = groups == g
        hover = [f"{t[:60]}<br>Precio: {p} EUR · Ventas: {u:,}".replace(",", ".") for t, p, u in zip(data.loc[m, "title_orig"], data.loc[m, "price"], data.loc[m, "units_sold"])]
        fig.add_trace(go.Scattergl(x=scores_all[m, ix], y=scores_all[m, iy], mode="markers", name=g, text=hover,
                                   marker=dict(color=color_map[g], size=7, opacity=0.6, line=dict(color=SURFACE, width=0.5)),
                                   hovertemplate=f"<b>{g}</b><br>%{{text}}<extra></extra>"))
    show(style(fig, f"Grupos sobre {ax_x} y {ax_y}", f"{ax_x} ({num(var_ratio[ix] * 100)} % de la varianza)",
               f"{ax_y} ({num(var_ratio[iy] * 100)} % de la varianza)", 540))

    st.subheader("Qué caracteriza a cada grupo, en las variables originales")
    profile = data[sel_vars].groupby(groups).median()
    profile.columns = [label(c) for c in profile.columns]
    profile.insert(0, "Productos", sizes)
    total = data[sel_vars].median()
    total.index = [label(c) for c in total.index]
    profile.loc["Toda la base"] = pd.concat([pd.Series({"Productos": len(data)}), total])
    st.dataframe(profile.round(2), width="stretch")
    st.caption("Mediana de cada variable por grupo, en sus unidades originales. La última fila sirve de referencia.")

    zmeans = Zdf.groupby(groups).mean()
    lim = float(np.abs(zmeans.values).max())
    fig = go.Figure(go.Heatmap(z=zmeans.values.T, x=zmeans.index, y=[label(c) for c in sel_vars], zmin=-lim, zmax=lim,
                               colorscale=DIVERGING, text=zmeans.round(2).values.T, texttemplate="%{text}", xgap=2, ygap=2,
                               colorbar=dict(title="Desv. est."),
                               hovertemplate="%{x} · %{y}<br>%{z:.2f} desviaciones frente al promedio<extra></extra>"))
    fig.update_yaxes(autorange="reversed")
    show(style(fig, "Distancia de cada grupo frente al promedio general", "", "", 480))
    st.caption("Azul: el grupo está por encima del promedio en esa variable. Naranja: por debajo. Gris: similar al promedio.")
    for g in group_names:
        st.markdown(f"- **{g}** ({num(sizes[g], 0)} productos): {descriptions[g]}.")

    st.subheader("¿Qué tan bien separados están?")
    sil_by = pd.Series(sil_samples).groupby(groups).mean().reindex(group_names)
    neg_by = pd.Series(sil_samples < 0).groupby(groups).mean().reindex(group_names) * 100
    fig = go.Figure(go.Bar(x=group_names, y=sil_by.values, marker_color=[color_map[g] for g in group_names],
                           text=[num(v, 2) for v in sil_by.values], textposition="outside", textfont=dict(color=INK),
                           hovertemplate="%{x}<br>Silueta promedio: %{y:.3f}<extra></extra>"))
    fig.update_yaxes(range=[0, max(0.6, float(sil_by.max()) + 0.1)])
    show(style(fig, "Silueta promedio por grupo", "", "Silueta promedio", 360))
    st.markdown(
        f"El **{num((sil_samples < 0).mean() * 100)} %** de los productos tiene silueta negativa: están más cerca de otro grupo "
        "que del suyo y son los casos difíciles de clasificar. Por grupo: "
        + ", ".join(f"{g} {num(neg_by[g])} %" for g in group_names) + "."
    )

    export = data[["product_id", "title_orig"] + sel_vars].copy()
    export["grupo"] = groups
    st.download_button("Descargar los productos con su grupo (CSV)", export.to_csv(index=False).encode("utf-8"),
                       "productos_con_grupo.csv", "text/csv")

# --------------------------------------------------------------------------------------
# 6. Conclusiones
# --------------------------------------------------------------------------------------

else:
    st.title("Conclusiones")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Productos analizados", num(len(data), 0))
    c2.metric("Varianza conservada", f"{num(retained)} %")
    c2.caption(f"con {n_comp} componentes")
    c3.metric("Grupos", k)
    c4.metric("Silueta promedio", num(sil_k, 3))

    st.subheader("Resultados con la configuración actual")
    st.markdown(
        f"- El ACP resume {len(sel_vars)} variables en **{n_comp} componentes** que conservan el **{num(retained)} %** de la variación.\n"
        f"- K-Means con **K = {k}** obtiene una silueta de **{num(sil_k, 3)}**; el mejor valor entre K = 2 y K = 10 está en K = {best_k}.\n"
        f"- El **{num((sil_samples < 0).mean() * 100)} %** de los productos queda más cerca de otro grupo que del suyo."
    )
    for g in group_names:
        st.markdown(f"- **{g}** ({num((groups == g).sum(), 0)} productos): {descriptions[g]}.")

    st.subheader("Lectura del caso")
    if not is_default:
        st.warning("La interpretación que sigue corresponde a la configuración recomendada: las diez variables, limpieza y "
                   "logaritmo activados, 3 componentes, K = 4 y semilla 42. Cambiaste algún parámetro, así que las cifras "
                   "de arriba pueden no coincidir con este texto.")
    st.markdown(
        """
**Tres dimensiones explican el 61 % de las diferencias entre productos.**
La primera es la *tracción comercial*: unidades vendidas, número de reseñas y tamaño del vendedor se mueven juntos.
La segunda es el *nivel de precio*: precio de venta, precio de referencia y costo de envío.
La tercera es la *reputación y el alcance*: calificación del producto, calificación del vendedor y países de envío.
Lo más interesante es que estas dimensiones son independientes entre sí: **un producto más caro no vende más ni está
mejor calificado, y uno muy vendido no tiene mejor calificación que el resto.**

**Se conservaron tres componentes y no cuatro** porque el cuarto tiene un valor propio prácticamente igual a 1
(no resume más que una variable sola) y porque agregarlo empeora la separación de los grupos.
El costo es real: cerca del 39 % de la variación queda por fuera.

**Aparecen cuatro perfiles, pero no cuatro grupos nítidos.** Con K = 4:

- **Productos consolidados** (el grupo más grande, cerca del 38 %): precio medio, unas 10.000 unidades vendidas,
  más de mil reseñas y vendedores grandes.
- **Gama alta del catálogo** (cerca del 27 %): los más caros, con precio de referencia y envío más altos y ventas intermedias.
- **Económicos de baja rotación** (cerca del 23 %): los más baratos, envío mínimo, pocas ventas y poco inventario por variación.
- **Vendedores pequeños con baja reputación** (cerca del 12 %): la peor calificación de producto y de vendedor,
  vendedores con muy pocas reseñas y ventas bajas.

**La separación es débil.** La silueta ronda 0,27 y el mejor valor está en K = 2, que solo distingue productos con
tracción de productos sin ella. Se eligió K = 4 porque pierde apenas 0,03 de silueta, es estable al cambiar la semilla
y entrega perfiles que sirven para tomar decisiones. Los grupos son zonas de un continuo, no categorías naturales.

**Qué significa para el problema inicial.** Lo que distingue a un producto en Wish no es el precio sino la tracción
acumulada (ventas y reseñas) y la reputación del vendedor. Bajar el precio, por sí solo, no ubica a un producto entre
los consolidados: el grupo más barato es también uno de los que menos vende.
"""
    )

    st.subheader("Limitaciones")
    st.markdown(
        """
- **Corte transversal.** Los datos son de un solo mes: no se puede saber si las reseñas generan ventas o las ventas generan
  reseñas. Las conclusiones describen asociaciones, no causas.
- **Ventas en escalones.** `units_sold` no es un conteo exacto, lo que limita la precisión del componente de tracción.
- **Sesgo de selección.** Solo están los productos que aparecían al buscar *summer*; los perfiles no se pueden
  extender a todo Wish ni a otras plataformas.
- **Pérdida de información.** Los tres componentes dejan por fuera cerca del 39 % de la variación.
- **Sensibilidad del método.** K-Means supone grupos compactos y de tamaño similar, y depende de las variables elegidas
  y de la transformación logarítmica. Cambiar esas decisiones cambia los grupos.
- **Vigencia.** La captura es de 2020; el mercado y la plataforma han cambiado.
"""
    )
