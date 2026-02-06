import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt

st.set_page_config(page_title="ETF Analyzer", layout="wide")

DEFAULT_UNIVERSE_FILE = "etf_universe.csv"

st.title("🧭 ETF Analyzer (5Y) — Sector & Style Rankings + Individual Analysis")
st.caption("Datos: Yahoo Finance (precios ajustados). Este app es educativo y NO garantiza ganancias.")

@st.cache_data(ttl=6*60*60)
def load_universe(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["ticker"] = df["ticker"].astype(str).str.upper().str.strip()
    df["group"] = df["group"].astype(str).str.strip()
    df["category"] = df["category"].astype(str).str.strip()
    if "notes" not in df.columns:
        df["notes"] = ""
    df["notes"] = df["notes"].astype(str)
    return df

@st.cache_data(ttl=6*60*60)
def get_prices(tickers, period="5y") -> pd.DataFrame:
    data = yf.download(tickers, period=period, auto_adjust=True, progress=False)
    if "Close" not in data:
        raise ValueError("No se pudieron descargar precios. Revisa tickers.")
    close = data["Close"]
    if isinstance(close, pd.Series):
        close = close.to_frame()
    close = close.dropna(how="all")
    close.columns = [c.upper() for c in close.columns]
    return close

def compute_metrics(prices: pd.DataFrame, rf_annual: float = 0.0) -> pd.DataFrame:
    rets = prices.pct_change().dropna()
    total_return = prices.iloc[-1] / prices.iloc[0] - 1

    n_days = (prices.index[-1] - prices.index[0]).days
    years = n_days / 365.25 if n_days > 0 else np.nan
    cagr = (1 + total_return) ** (1 / years) - 1 if years and years > 0 else np.nan

    ann_vol = rets.std() * np.sqrt(252)
    ann_ret = (1 + rets.mean()) ** 252 - 1
    rf_daily = (1 + rf_annual) ** (1/252) - 1
    sharpe = (rets.mean() - rf_daily) / rets.std()
    sharpe_ann = sharpe * np.sqrt(252)

    growth = (1 + rets).cumprod()
    peak = growth.cummax()
    dd = growth / peak - 1
    mdd = dd.min()

    out = pd.DataFrame({
        "Total Return": total_return,
        "CAGR": cagr,
        "Ann. Return (est.)": ann_ret,
        "Ann. Vol": ann_vol,
        "Sharpe (ann)": sharpe_ann,
        "Max Drawdown": mdd
    })
    return out

def normalize_score(series: pd.Series) -> pd.Series:
    s = series.replace([np.inf, -np.inf], np.nan)
    valid = s.dropna()
    if valid.empty:
        return s * np.nan
    mn, mx = valid.min(), valid.max()
    if mx == mn:
        return pd.Series(50.0, index=s.index)
    return (s - mn) / (mx - mn) * 100

def recommendation_text(row: pd.Series, notes: str, horizon_years: float, risk: str) -> str:
    cagr = row.get("CAGR", np.nan)
    vol = row.get("Ann. Vol", np.nan)
    mdd = row.get("Max Drawdown", np.nan)
    sharpe = row.get("Sharpe (ann)", np.nan)

    pros, cons = [], []

    if np.isfinite(sharpe) and sharpe > 0.7:
        pros.append("Buen retorno ajustado por riesgo (Sharpe relativamente alto).")
    elif np.isfinite(sharpe) and sharpe < 0.3:
        cons.append("Retorno ajustado por riesgo débil (Sharpe bajo).")

    if np.isfinite(mdd) and mdd > -0.25:
        pros.append("Caídas históricas (drawdown) moderadas para un ETF de mercado.")
    elif np.isfinite(mdd) and mdd <= -0.40:
        cons.append("Ha tenido caídas fuertes (drawdown alto); puede ser duro psicológicamente.")

    if np.isfinite(vol) and vol < 0.15:
        pros.append("Volatilidad relativamente baja.")
    elif np.isfinite(vol) and vol > 0.25:
        cons.append("Volatilidad alta; puede moverse fuerte en periodos cortos.")

    if isinstance(notes, str) and notes.strip():
        pros.append(f"Contexto: {notes.strip()}")

    if horizon_years <= 2:
        cons.append("Horizonte corto: el mercado puede estar abajo justo cuando necesites el dinero.")
        if risk.lower().startswith("conserv"):
            pros.append("Si priorizas estabilidad, combina con bonos/cash (no sólo equity ETFs).")

    if risk.lower().startswith("agres"):
        pros.append("Perfil agresivo: puedes tolerar más volatilidad si tu plan lo permite.")
    if risk.lower().startswith("conserv") and np.isfinite(vol) and vol > 0.20:
        cons.append("Para perfil conservador, este ETF puede ser demasiado volátil.")

    if not pros:
        pros.append("Estructura simple (ETF) y diversificación relativa vs acciones individuales.")

    return "**Por qué SÍ:**\n- " + "\n- ".join(pros) + "\n\n**Por qué NO / Riesgos:**\n- " + "\n- ".join(cons)

universe = load_universe(DEFAULT_UNIVERSE_FILE)

st.sidebar.header("⚙️ Configuración")
rf = st.sidebar.number_input("Tasa libre de riesgo anual (Sharpe)", min_value=0.0, max_value=0.15, value=0.03, step=0.005, format="%.3f")
period = st.sidebar.selectbox("Ventana histórica", ["5y", "3y", "1y"], index=0)
risk = st.sidebar.selectbox("Perfil de riesgo (para la explicación)", ["Conservador", "Moderado", "Agresivo"], index=1)
horizon = st.sidebar.slider("Horizonte (años) — para la explicación", min_value=1, max_value=10, value=2)

tickers = sorted(universe["ticker"].unique().tolist())

st.sidebar.divider()
st.sidebar.subheader("➕ Agregar cualquier ETF")
st.sidebar.caption("Escribe tickers separados por coma. Ej: VOO, IVV, QQQ, XLF, IWM")

user_input = st.sidebar.text_input("Tickers adicionales")
extra = []
if user_input.strip():
    extra = [x.strip().upper() for x in user_input.split(",") if x.strip()]
tickers = sorted(list(set(tickers + extra)))


with st.spinner("Descargando precios…"):
    prices = get_prices(tickers, period=period)

metrics = compute_metrics(prices, rf_annual=rf).reset_index().rename(columns={"index":"ticker"})
df = universe.merge(metrics, on="ticker", how="left")

df["Score CAGR"] = normalize_score(df["CAGR"])
df["Score Sharpe"] = normalize_score(df["Sharpe (ann)"])
df["Score Drawdown"] = normalize_score(-df["Max Drawdown"])
df["Score Vol"] = normalize_score(-df["Ann. Vol"])
df["Composite Score"] = (0.40*df["Score CAGR"] + 0.35*df["Score Sharpe"] + 0.15*df["Score Drawdown"] + 0.10*df["Score Vol"])

tab1, tab2, tab3 = st.tabs(["🏆 Rankings (Sector/Style)", "🔍 Análisis Individual", "🧩 Plan (NO garantiza)"])

with tab1:
    st.subheader("🏆 Mejores y peores ETFs por grupo")
    st.caption("Ranking por score compuesto (CAGR, Sharpe, drawdown y volatilidad).")

    group_choice = st.selectbox("Ver por:", ["group", "category"], index=0)
    groups = sorted(df[group_choice].dropna().unique().tolist())

    for g in groups:
        sub = df[df[group_choice] == g].dropna(subset=["Composite Score"]).copy()
        if sub.empty:
            continue
        sub = sub.sort_values("Composite Score", ascending=False)

        st.markdown(f"### {g}")
        colA, colB = st.columns(2)

        with colA:
            st.markdown("**Top 5 (mejores)**")
            st.dataframe(
                sub[["ticker","name","CAGR","Sharpe (ann)","Max Drawdown","Ann. Vol","Composite Score"]].head(5),
                use_container_width=True
            )
        with colB:
            st.markdown("**Bottom 5 (peores)**")
            st.dataframe(
                sub[["ticker","name","CAGR","Sharpe (ann)","Max Drawdown","Ann. Vol","Composite Score"]].tail(5).sort_values("Composite Score"),
                use_container_width=True
            )

    st.divider()
    st.subheader("📊 Comparación rápida")
    compare = st.multiselect("Selecciona ETFs para comparar", tickers, default=tickers[:4])
    if compare:
        comp_prices = prices[compare].dropna()
        comp_rets = comp_prices.pct_change().dropna()
        growth = (1 + comp_rets).cumprod()

        fig, ax = plt.subplots()
        growth.plot(ax=ax)
        ax.set_title("Crecimiento de $1 (precios ajustados)")
        ax.set_ylabel("Growth")
        st.pyplot(fig, use_container_width=True)

with tab2:
    st.subheader("🔍 Análisis individual")
    t = st.selectbox("Selecciona un ETF", tickers, index=0)
    meta = universe[universe["ticker"] == t].iloc[0].to_dict()
    m = df[df["ticker"] == t].iloc[0]

    st.markdown(f"### {t} — {meta.get('name','')}")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("CAGR", f"{m['CAGR']*100:.2f}%" if pd.notna(m["CAGR"]) else "—")
    c2.metric("Sharpe (ann)", f"{m['Sharpe (ann)']:.2f}" if pd.notna(m["Sharpe (ann)"]) else "—")
    c3.metric("Vol (ann)", f"{m['Ann. Vol']*100:.2f}%" if pd.notna(m["Ann. Vol"]) else "—")
    c4.metric("Max Drawdown", f"{m['Max Drawdown']*100:.2f}%" if pd.notna(m["Max Drawdown"]) else "—")
    c5.metric("Total Return", f"{m['Total Return']*100:.2f}%" if pd.notna(m["Total Return"]) else "—")

    st.caption(f"Grupo: **{meta.get('group','')}** | Categoría: **{meta.get('category','')}**")
    if meta.get("notes"):
        st.info(meta.get("notes"))

    p = prices[[t]].dropna()
    ret = p.pct_change().dropna()

    colL, colR = st.columns([2,1])
    with colL:
        fig, ax = plt.subplots()
        (p / p.iloc[0]).plot(ax=ax)
        ax.set_title(f"{t} — Precio ajustado (normalizado)")
        ax.set_ylabel("Index (Start=1)")
        st.pyplot(fig, use_container_width=True)

    with colR:
        g = (1 + ret).cumprod()
        peak = g.cummax()
        dd = g/peak - 1
        fig2, ax2 = plt.subplots()
        dd.plot(ax=ax2, legend=False)
        ax2.set_title("Drawdown")
        ax2.set_ylabel("DD")
        st.pyplot(fig2, use_container_width=True)

    st.subheader("✅ Recomendación (heurística)")
    st.warning("Esto NO es asesoría financiera. Es una explicación basada en historial y tipo de ETF.")
    st.markdown(recommendation_text(m, meta.get("notes",""), float(horizon), risk))

with tab3:
    st.subheader("🧩 Plan de inversión (NO garantiza ganancias)")
    st.caption("No existe forma responsable de garantizar ganancias en 2 años con ETFs de mercado. "
               "Aquí armamos un plan por perfil con reglas claras.")
    amount = st.number_input("¿Cuánto vas a invertir? (USD)", min_value=100.0, value=10000.0, step=500.0)

    presets = {
        "Conservador (más estabilidad)": ["BND", "IEF", "SHY", "VOO"],
        "Moderado (balanceado)": ["VOO", "VTI", "VXUS", "BND"],
        "Agresivo (más crecimiento)": ["VTI", "QQQ", "VXUS", "AVUV"]
    }
    preset_name = st.selectbox("Preset", list(presets.keys()), index=1)
    chosen = st.multiselect("ETFs en tu plan", tickers, default=[x for x in presets[preset_name] if x in tickers])

    if chosen:
        st.markdown("#### Pesos objetivo")
        cols = st.columns(min(4, len(chosen)))
        targets = {}
        for i, etf in enumerate(chosen):
            with cols[i % len(cols)]:
                targets[etf] = st.number_input(f"{etf} peso", min_value=0.0, max_value=1.0, value=1/len(chosen), step=0.05, format="%.2f")
        w = pd.Series(targets)
        if w.sum() > 0:
            w = w / w.sum()
            plan = pd.DataFrame({
                "ETF": w.index,
                "Peso": w.values,
                "Monto ($)": (w.values * amount)
            }).sort_values("Monto ($)", ascending=False)
            st.dataframe(plan, use_container_width=True)

            st.markdown("#### Reglas simples")
            st.write("- Aporta cada mes (DCA) si puedes.")
            st.write("- Rebalancea cada 3 meses (o cuando un ETF se desvíe > 5% del objetivo).")
            st.write("- Mantén fondo de emergencia aparte.")
            st.write("- Si tu horizonte es 2 años y necesitas el dinero sí o sí, reduce riesgo (más bonos/cash).")

            sub_prices = prices[w.index].dropna()
            sub_rets = sub_prices.pct_change().dropna()
            port_ret = (sub_rets * w).sum(axis=1)
            growth = (1 + port_ret).cumprod()

            fig, ax = plt.subplots()
            growth.plot(ax=ax, legend=False)
            ax.set_title("Crecimiento histórico de $1 (no predice futuro)")
            ax.set_ylabel("Growth")
            st.pyplot(fig, use_container_width=True)
        else:
            st.info("Asigna al menos un peso > 0.")
    else:
        st.info("Selecciona al menos 1 ETF.")
