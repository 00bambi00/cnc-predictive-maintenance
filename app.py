import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go

from scipy.fft import fft
from scipy.stats import kurtosis, skew


# =========================
# PAGE CONFIG
# =========================

st.set_page_config(
    page_title="CNC Predictive Maintenance AI",
    page_icon="⚙️",
    layout="wide"
)


# =========================
# CONFIG
# =========================

FAILURE_WEAR = 220


# =========================
# STYLE
# =========================

st.markdown("""
<style>

.stApp {
    background-color:#0f172a;
}

h1,h2,h3 {
    color:white;
}

[data-testid="metric-container"]{
    background:#1e293b;
    padding:15px;
    border-radius:15px;
}

</style>

""", unsafe_allow_html=True)



# =========================
# FEATURE EXTRACTION
# เหมือน Notebook
# =========================

def extract(x):

    x = np.array(x, dtype=float)

    f = np.abs(fft(x))[:len(x)//2]


    return [

        np.mean(x),
        np.std(x),
        np.max(x),
        np.min(x),
        np.median(x),
        np.var(x),

        np.sqrt(np.mean(x**2)),

        kurtosis(x),
        skew(x),

        np.ptp(x),

        np.percentile(x,25),
        np.percentile(x,75),

        np.mean(f),
        np.max(f),
        np.sum(f**2),
        np.argmax(f)

    ]



# =========================
# LOAD MODEL
# =========================

@st.cache_resource
def load_model():

    model = joblib.load(
        "model.pkl"
    )

    scaler = joblib.load(
        "scaler.pkl"
    )

    return model, scaler



model, scaler = load_model()



# =========================
# UPLOAD DATA
# =========================

st.title(
    "⚙️ CNC Predictive Maintenance AI"
)

st.write(
"""
AI system for CNC tool wear prediction using
FFT feature extraction and Random Forest regression.
"""
)


uploaded_file = st.file_uploader(
    "Upload CNC sensor CSV file",
    type=["csv"]
)



if uploaded_file:


    df = pd.read_csv(
        uploaded_file,
        header=None
    )


    st.subheader(
        "Sensor Data Preview"
    )


    st.dataframe(
        df.head()
    )


    # ----------------------
    # FEATURE GENERATION
    # ----------------------

    features=[]


    for col in range(df.shape[1]):

        signal=df.iloc[:,col].values

        features.extend(
            extract(signal)
        )


    X_new=np.array(
        [features]
    )


    # DEBUG

    st.write(
        "Generated Features:",
        X_new.shape[1]
    )

    st.write(
        "Expected Features:",
        scaler.n_features_in_
    )


    # ----------------------
    # CHECK FEATURE SIZE
    # ----------------------

    if X_new.shape[1] != scaler.n_features_in_:


        st.error(
        f"""
        Feature mismatch!

        Model expects:
        {scaler.n_features_in_}

        But uploaded data generated:
        {X_new.shape[1]}

        Please upload the same sensor format used during training.
        """
        )

        st.stop()



    # ----------------------
    # PREDICT
    # ----------------------

    X_scaled=scaler.transform(
        X_new
    )


    wear=model.predict(
        X_scaled
    )[0]



    # ----------------------
    # HEALTH
    # ----------------------

    health=max(
        0,
        100*(1-wear/FAILURE_WEAR)
    )


    remaining=max(
        0,
        FAILURE_WEAR-wear
    )


    days=remaining/10



    if days < 3:

        status="🔴 CRITICAL"

    elif days <14:

        status="🟡 WARNING"

    else:

        status="🟢 NORMAL"



    # ======================
    # DASHBOARD
    # ======================


    st.divider()


    c1,c2,c3,c4=st.columns(4)


    c1.metric(
        "Current Wear",
        f"{wear:.2f} μm"
    )


    c2.metric(
        "Tool Health",
        f"{health:.1f}%"
    )


    c3.metric(
        "Remaining Life",
        f"{days:.1f} Days"
    )


    c4.metric(
        "Status",
        status
    )



    st.divider()



    fig=go.Figure(
        go.Indicator(

            mode="gauge+number",

            value=health,

            title={
                "text":"Tool Health"
            },

            gauge={
                "axis":{
                    "range":[0,100]
                }
            }

        )
    )


    st.plotly_chart(
        fig,
        use_container_width=True
    )



    st.subheader(
        "AI Recommendation"
    )


    st.info(

    f"""
    Current tool wear:
    {wear:.2f} μm


    Wear level:
    {(wear/FAILURE_WEAR)*100:.1f}%


    Estimated remaining life:
    {days:.1f} days


    Condition:
    {status}

    """

    )


else:

    st.warning(
        "Please upload CNC sensor CSV file."
    )