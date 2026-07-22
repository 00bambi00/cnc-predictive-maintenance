import streamlit as st
import pandas as pd
import numpy as np

import os
import glob
import joblib

from scipy.fft import fft
from scipy.stats import kurtosis, skew

import plotly.graph_objects as go


# ==========================
# CONFIG
# ==========================

st.set_page_config(
    page_title="CNC Predictive Maintenance AI",
    page_icon="⚙️",
    layout="wide"
)


ROOT_PATH = "archive"

FAILURE_WEAR = 220



# ==========================
# STYLE
# ==========================

st.markdown("""
<style>

.stApp {
    background:#0F172A;
}

h1,h2,h3 {
    color:white;
}

[data-testid="metric-container"] {

background:#1E293B;
padding:15px;
border-radius:15px;

}

</style>

""", unsafe_allow_html=True)



# ==========================
# FEATURE EXTRACTION
# ==========================

def extract(x):

    x=np.array(x,dtype=float)


    if len(x)>2000:
        x=x[:2000]


    f=np.abs(fft(x))[:len(x)//2]


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



# ==========================
# LOAD MODEL
# ==========================

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



# ==========================
# LOAD LAST DATA
# ==========================

@st.cache_data
def load_latest():

    X=[]


    cases=[1,4,6]


    for case in cases:


        folder=os.path.join(

            ROOT_PATH,
            f"c{case}",
            f"c{case}"

        )


        files=sorted(

            glob.glob(
                os.path.join(
                    folder,
                    "*.csv"
                )
            )

        )


        if len(files)==0:
            continue


        # เอาไฟล์ล่าสุด

        file=files[-1]


        df=pd.read_csv(
            file,
            header=None
        )


        feature=[]


        for col in range(df.shape[1]):

            feature.extend(

                extract(
                    df.iloc[:,col]
                )

            )


        X.append(feature)



    return np.array(X)



X_new=load_latest()



X_scaled=scaler.transform(
    X_new
)



wear=model.predict(
    X_scaled
)[0]



# ==========================
# CALCULATE HEALTH
# ==========================

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


elif days < 14:

    status="🟡 WARNING"


else:

    status="🟢 NORMAL"




# ==========================
# DASHBOARD
# ==========================


st.title(
    "⚙️ CNC Predictive Maintenance AI"
)


st.caption(
    "AI Tool Wear Prediction using FFT Feature Extraction + Random Forest"
)



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



# Gauge

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


fig.update_layout(
height=350
)


st.plotly_chart(
fig,
use_container_width=True
)



# REPORT


st.subheader(
"🧠 AI Recommendation"
)


st.info(

f"""
Current tool wear:
{wear:.2f} μm


Wear percentage:
{wear/FAILURE_WEAR*100:.1f}%


Estimated remaining life:
{days:.1f} days


Condition:
{status}

"""

)