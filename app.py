import streamlit as st
import requests
import pandas as pd
import numpy as np

# -----------------------------
# CONFIG GLOBALE
# -----------------------------
st.set_page_config(
    page_title="Simulateur d'économies photovoltaïques",
    page_icon="☀️",
    layout="wide"
)

# -----------------------------
# FONCTIONS UTILITAIRES
# -----------------------------

@st.cache_data(show_spinner=False)
def geocode_address(address: str):
    """
    Géocode une adresse avec Nominatim (OpenStreetMap)
    Retourne (lat, lon) ou lève une exception si non trouvé.
    """
    url = "https://nominatim.openstreetmap.org/search"
    params = {"q": address, "format": "json", "limit": 1}
    headers = {"User-Agent": "solar-simulator-streamlit/1.0"}
    r = requests.get(url, params=params, headers=headers, timeout=10)
    r.raise_for_status()
    data = r.json()
    if not data:
        raise ValueError("Adresse non trouvée par le service de géocodage.")
    lat = float(data[0]["lat"])
    lon = float(data[0]["lon"])
    return lat, lon


@st.cache_data(show_spinner=True)
def get_pvgis_hourly(lat, lon, peakpower_kw, angle, aspect):
    """
    Appelle l'API PVGIS 'seriescalc', convertit la série (10 min ou autre)
    en série HORAIRE, et renvoie un DataFrame avec :
        index : DatetimeIndex (heure pleine, sans timezone)
        colonne : 'pv_kwh' (énergie produite sur chaque heure)
    """
    url = "https://re.jrc.ec.europa.eu/api/v5_3/seriescalc"
    params = {
        "lat": lat,
        "lon": lon,
        "startyear": 2025,
        "endyear": 2025,
        "pvcalculation": 1,          # production PV activée
        "peakpower": peakpower_kw,   # kWc
        "loss": 14,                  # pertes %
        "mountingplace": "building",
        "outputformat": "json",
    }

    if auto_optimal:
        params["optimalangles"] = 1
    else:
        params["angle"] = angle
        params["aspect"] = aspect

    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    data = r.json()

    if "outputs" not in data or "hourly" not in data["outputs"]:
        raise ValueError(
            f"Réponse PVGIS inattendue. Clés outputs = {list(data.get('outputs', {}).keys())}"
        )

    df = pd.DataFrame(data["outputs"]["hourly"])

    if "time" not in df.columns or "P" not in df.columns:
        raise ValueError("Colonnes 'time' ou 'P' manquantes dans la réponse PVGIS.")

    # 1) Parsing du format de date PVGIS : 20200101:0010
    df["time"] = pd.to_datetime(df["time"], format="%Y%m%d:%H%M", errors="raise")
    df = df.sort_values("time").set_index("time")

    # 2) Calcul du pas de temps réel (en heures)
    if len(df) > 1:
        dt_hours = (df.index[1] - df.index[0]).total_seconds() / 3600.0
    else:
        dt_hours = 1.0  # fallback si un seul point

    # 3) P en W → énergie sur l'intervalle (kWh)
    #    E = P(W) * Δt(h) / 1000
    df["pv_kwh_interval"] = df["P"] * dt_hours / 1000.0

    # 4) On agrège par heure pleine pour matcher la conso (load_df)
    hourly = (
        df["pv_kwh_interval"]
        .resample("H")
        .sum()
        .to_frame(name="pv_kwh")
        .sort_index()
    )

    # index naïf (sans timezone) comme build_load_timeseries
    hourly.index = hourly.index.tz_localize(None)

    return hourly





def daily_profile_base(params, date):
    """
    Construit un profil horaire (24 valeurs) de consommation pour une journée donnée.
    Modèle simple basé sur :
    - conso de base
    - chauffage
    - clim
    - piscine
    - véhicule électrique
    - présence (augmentation conso matin/soir)
    """
    month = date.month
    h = np.arange(24)
    profile = np.zeros(24)

    # --- Conso de base (frigo, veille, box, etc.) ---
    base_kwh_day = params.get("base_kwh_day", 5.0)
    profile += base_kwh_day / 24.0

    # --- Chauffage électrique ou PAC chauffage (hiver) ---
    if params.get("chauffage", False) and month in [11, 12, 1, 2, 3]:
        chauffage_kwh_day = params.get("chauffage_kwh_day_winter", 20.0)
        prof_ch = np.zeros(24)
        # Montées le matin (6-9) et le soir (17-22)
        prof_ch[6:9] = 1
        prof_ch[17:22] = 1
        prof_ch = prof_ch / prof_ch.sum()
        profile += chauffage_kwh_day * prof_ch

    # --- Clim (été) ---
    if params.get("clim", False) and month in [6, 7, 8]:
        clim_kwh_day = params.get("clim_kwh_day_summer", 6.0)
        prof_clim = np.zeros(24)
        # Apres-midi et début de soirée
        prof_clim[12:21] = 1
        prof_clim = prof_clim / prof_clim.sum()
        profile += clim_kwh_day * prof_clim

    # --- Piscine (pompe de filtration, été large) ---
    if params.get("piscine", False) and month in [5, 6, 7, 8, 9]:
        piscine_kwh_day = params.get("piscine_kwh_day_summer", 6.0)
        prof_pisc = np.zeros(24)
        prof_pisc[8:12] = 1
        prof_pisc[14:18] = 1
        prof_pisc = prof_pisc / prof_pisc.sum()
        profile += piscine_kwh_day * prof_pisc

    # --- Véhicule électrique (charge surtout la nuit) ---
    if params.get("ve", False):
        ve_kwh_day = params.get("ve_kwh_day", 8.0)
        prof_ve = np.zeros(24)
        prof_ve[22:24] = 1
        prof_ve[0:6] = 1
        prof_ve = prof_ve / prof_ve.sum()
        profile += ve_kwh_day * prof_ve

    # --- Effet "présence" : on augmente la conso aux heures où les gens sont là ---
    presence_factor = np.ones(24)
    morning_start, morning_end = params.get("presence_morning", (6, 9))
    evening_start, evening_end = params.get("presence_evening", (17, 22))
    presence_factor[morning_start:morning_end] += params.get("presence_boost", 0.4)
    presence_factor[evening_start:evening_end] += params.get("presence_boost", 0.6)

    profile *= presence_factor

    return profile


def build_load_timeseries(year, params):
    """
    Construit un DataFrame horaire sur une année avec la consommation électrique (kWh/h)
    Index = DatetimeIndex naïf (sans timezone) pour matcher pv_df.
    """
    rng = pd.date_range(f"{year}-01-01 00:00", f"{year}-12-31 23:00", freq="H")
    days = sorted(set(rng.date))

    load_values = []
    for d in days:
        day_profile = daily_profile_base(params, pd.Timestamp(d))
        load_values.extend(day_profile.tolist())

    load_values = load_values[:len(rng)]

    df = pd.DataFrame({"load_kwh": load_values}, index=rng)
    return df



def simulate_savings(pv_df, load_df, price_hp, price_hc=None, inj_price=0.13, hc_hours=(22, 6)):
    """
    Calcule autoconsommation, injection, achat réseau, et économies.
    Si price_hc est fourni, on applique HP/HC selon les heures.
    """
    df = pv_df.join(load_df, how="inner")
    df = df.sort_index()

    pv = df["pv_kwh"].values
    load = df["load_kwh"].values

    autocons = np.minimum(pv, load)
    injection = np.maximum(pv - load, 0)
    reseau = np.maximum(load - pv, 0)

    df["autocons_kwh"] = autocons
    df["injection_kwh"] = injection
    df["reseau_kwh"] = reseau

    # Facture SANS PV
    if price_hc is None:
        # Tarif unique
        facture_sans_pv = load.sum() * price_hp
        facture_avec_pv = reseau.sum() * price_hp
    else:
        # HP/HC
        hours = df.index.hour
        is_hc = (hours >= hc_hours[0]) | (hours < hc_hours[1])
        prix_sans = np.where(is_hc, price_hc, price_hp)
        prix_avec = prix_sans  # même structure tarifaire

        facture_sans_pv = np.sum(load * prix_sans)
        facture_avec_pv = np.sum(reseau * prix_avec)

    recette_injection = injection.sum() * inj_price
    economie_nette = facture_sans_pv - facture_avec_pv + recette_injection

    resume = {
        "conso_totale_kwh": float(load.sum()),
        "prod_totale_kwh": float(pv.sum()),
        "autocons_kwh": float(autocons.sum()),
        "taux_autocons": float(autocons.sum() / pv.sum()) if pv.sum() > 0 else 0,
        "taux_autonomie": float(autocons.sum() / load.sum()) if load.sum() > 0 else 0,
        "injection_kwh": float(injection.sum()),
        "reseau_kwh": float(reseau.sum()),
        "facture_sans_pv": float(facture_sans_pv),
        "facture_avec_pv": float(facture_avec_pv),
        "recette_injection": float(recette_injection),
        "economie_nette": float(economie_nette),
    }
    return df, resume


# -----------------------------
# INTERFACE STREAMLIT
# -----------------------------

st.title("☀️ Simulateur d’économies photovoltaïques")
st.markdown(
    """
Ce simulateur estime la **production solaire**, la **consommation de votre maison** et les **économies réalisables**
grâce à l'autoconsommation et à la revente du surplus.
"""
)

with st.sidebar:
    st.header("📍 Localisation & installation")

    address = st.text_input(
        "Adresse postale",
        placeholder="Ex : 10 rue de la Paix, 75002 Paris, France"
    )

    auto_optimal = st.checkbox("Laisser le simulateur optimiser l'angle et l'orientation", True)
    
    col_p1, col_p2 = st.columns(2)
    with col_p1:
        kwc = st.number_input("Puissance installée (kWc)", 1.0, 36.0, 6.0, 0.5)
    with col_p2:
        angle = st.slider("Inclinaison (°)", 0, 60, 30, 1)

    aspect_label = st.selectbox(
        "Orientation",
        ["Sud", "Sud-Est", "Sud-Ouest", "Est", "Ouest"]
    )
    aspect_map = {
        "Sud": 0,
        "Sud-Est": -45,
        "Sud-Ouest": 45,
        "Est": -90,
        "Ouest": 90
    }
    aspect = aspect_map[aspect_label]

    st.header("🏠 Profil de la maison")

    base_kwh_day = st.slider("Conso de base (kWh/jour)", 2.0, 15.0, 5.0, 0.5)

    nb_personnes = st.slider("Nombre de personnes", 1, 6, 3)

    st.markdown("**Équipements**")
    chauffage = st.checkbox("Chauffage électrique / PAC chauffage", True)
    clim = st.checkbox("Climatisation l'été", False)
    piscine = st.checkbox("Piscine (pompe de filtration)", False)
    ve = st.checkbox("Véhicule électrique", False)

    st.header("⏱️ Présence & usage")
    presence_boost = st.slider("Impact de la présence sur la conso", 0.0, 1.0, 0.4, 0.05)

    st.header("💶 Tarifs et investissement")

    price_hp = st.number_input("Prix kWh (HP ou tarif unique) €", 0.10, 0.40, 0.22, 0.01)
    use_hc = st.checkbox("Utiliser heures creuses", False)
    price_hc = None
    if use_hc:
        price_hc = st.number_input("Prix kWh (HC) €", 0.05, 0.30, 0.16, 0.01)

    inj_price = st.number_input("Prix de rachat du surplus (€/kWh)", 0.00, 0.30, 0.13, 0.01)

    st.markdown("---")
    st.markdown("**Investissement (optionnel)**")
    capex = st.number_input("Coût de l'installation (€)", 0, 50000, 9000, 500)

    simulate_button = st.button("Lancer la simulation 🚀")

# -----------------------------
# LOGIQUE DE SIMULATION
# -----------------------------
if simulate_button:
    if not address.strip():
        st.error("Merci de renseigner une adresse postale.")
    else:
        try:
            with st.spinner("Géocodage de l'adresse..."):
                lat, lon = geocode_address(address)

            st.success(f"Adresse géocodée : lat={lat:.4f}, lon={lon:.4f}")

            with st.spinner("Calcul de la production photovoltaïque (PVGIS)..."):
                pv_df = get_pvgis_hourly(lat, lon, kwc, angle, aspect)

            st.success("Production PV récupérée.")

            # Paramètres de conso
            params_load = {
                "base_kwh_day": base_kwh_day + (nb_personnes - 1) * 1.0,
                "chauffage": chauffage,
                "chauffage_kwh_day_winter": 6.0 * nb_personnes,
                "clim": clim,
                "clim_kwh_day_summer": 2.0 * nb_personnes,
                "piscine": piscine,
                "piscine_kwh_day_summer": 4.0,
                "ve": ve,
                "ve_kwh_day": 7.0,
                "presence_morning": (6, 9),
                "presence_evening": (17, 22),
                "presence_boost": presence_boost,
            }

            with st.spinner("Construction du profil de consommation..."):
                year = pv_df.index[0].year
                load_df = build_load_timeseries(year, params_load)

            with st.spinner("Simulation de l'autoconsommation et des économies..."):
                df, resume = simulate_savings(
                    pv_df, load_df,
                    price_hp=price_hp,
                    price_hc=price_hc,
                    inj_price=inj_price
                )

            # -----------------------------
            # AFFICHAGE DES RÉSULTATS
            # -----------------------------
            st.subheader("📊 Résultats annuels")

            col1, col2, col3 = st.columns(3)
            col1.metric("Production annuelle", f"{resume['prod_totale_kwh']:.0f} kWh")
            col2.metric("Consommation annuelle", f"{resume['conso_totale_kwh']:.0f} kWh")
            col3.metric("Taux d'autoconsommation", f"{resume['taux_autocons']*100:.0f} %")

            col4, col5, col6 = st.columns(3)
            col4.metric("Taux d'autonomie", f"{resume['taux_autonomie']*100:.0f} %")
            col5.metric("Énergie injectée", f"{resume['injection_kwh']:.0f} kWh")
            col6.metric("Énergie achetée réseau", f"{resume['reseau_kwh']:.0f} kWh")

            st.subheader("💶 Économies financières")

            col7, col8, col9 = st.columns(3)
            col7.metric("Facture sans PV", f"{resume['facture_sans_pv']:.0f} € / an")
            col8.metric("Facture avec PV", f"{resume['facture_avec_pv']:.0f} € / an")
            col9.metric("Recette injection", f"{resume['recette_injection']:.0f} € / an")

            economie = resume["economie_nette"]
            st.metric("Économie nette annuelle", f"{economie:.0f} € / an")

            if capex > 0 and economie > 0:
                tri = capex / economie
                st.info(f"⏳ Temps de retour sur investissement estimé : **{tri:.1f} ans**")

            # -----------------------------
            # GRAPHIQUES
            # -----------------------------
            st.subheader("📈 Production vs consommation (exemple sur 7 jours)")

            df_plot = df[["pv_kwh", "load_kwh", "autocons_kwh"]].copy()
            df_plot = df_plot.rename(columns={
                "pv_kwh": "Production PV (kWh)",
                "load_kwh": "Consommation (kWh)",
                "autocons_kwh": "Autoconsommation (kWh)"
            })
            
            week_sample = df_plot.iloc[:24*7].copy()
            week_sample = week_sample.reset_index().rename(columns={"index": "Date"})
            
            st.line_chart(week_sample, x="Date")

            with st.expander("Voir le détail horaire complet (table)"):
                st.dataframe(df.head(200))

        except Exception as e:
            st.error(f"Une erreur est survenue : {e}")
else:
    st.info("Renseigne les paramètres dans la barre latérale, puis clique sur **Lancer la simulation 🚀**.")








