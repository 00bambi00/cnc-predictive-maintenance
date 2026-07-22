import streamlit as st
import pandas as pd
import numpy as np

from scipy.fft import fft
from scipy.stats import kurtosis, skew

from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

import os
import glob

import plotly.graph_objects as go
import plotly.express as px


# ==========================
# CONFIG
# ==========================

st.set_page_config(
    page_title="CNC Predictive Maintenance AI",
    page_icon="⚙️",
    layout="wide"
)


ROOT_PATH = "archive"

CASES = [1,4,6]

FAILURE_WEAR = 220



# ==========================
# STYLE
# ==========================

st.markdown("""
<style>

.stApp {
    background-color:#0b1220;
}

h1,h2,h3 {
    color:white;
}

div[data-testid="metric-container"] {
    background:#162033;
    padding:20px;
    border-radius:15px;
}

</style>
""", unsafe_allow_html=True)



# ==========================
# FEATURE EXTRACTION
# ==========================

def extract(x):

    x=np.array(x)

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
# LOAD DATA
# ==========================

@st.cache_data
def load_data():

    X=[]
    Y=[]


    for case in CASES:


        folder = os.path.join(
            ROOT_PATH,
            f"c{case}",
            f"c{case}"
        )


        wear_file=os.path.join(
            ROOT_PATH,
            f"c{case}",
            f"c{case}_wear.csv"
        )


        if not os.path.exists(wear_file):
            continue


        wear=pd.read_csv(wear_file)


        wear_values = (
            wear[
            [
            "flute_1",
            "flute_2",
            "flute_3"
            ]
            ]
            .mean(axis=1)
            .values
        )


        files=sorted(
            glob.glob(
                os.path.join(folder,"*.csv")
            )
        )


        n=min(
            len(files),
            len(wear_values)
        )


        for i in range(n):

            df=pd.read_csv(
                files[i],
                header=None
            )


            features=[]


            for col in range(df.shape[1]):

                features.extend(
                    extract(
                        df.iloc[:,col]
                    )
                )


            X.append(features)

            Y.append(
                wear_values[i]
            )


    return np.array(X),np.array(Y)




# ==========================
# TRAIN MODEL
# ==========================

@st.cache_resource
def create_model(X,Y):


    scaler=RobustScaler()


    X_scaled=scaler.fit_transform(X)


    X_train,X_test,y_train,y_test=train_test_split(
        X_scaled,
        Y,
        test_size=0.2,
        random_state=42
    )


    model=RandomForestRegressor(
        n_estimators=500,
        max_depth=15,
        random_state=42,
        n_jobs=-1
    )


    model.fit(
        X_train,
        y_train
    )


    pred=model.predict(
        X_test
    )


    mae=mean_absolute_error(
        y_test,
        pred
    )


    r2=r2_score(
        y_test,
        pred
    )


    return (
        model,
        scaler,
        X_scaled,
        y_test,
        pred,
        mae,
        r2
    )



# ==========================
# MAIN
# ==========================


st.title(
    "⚙️ CNC Predictive Maintenance AI"
)

st.write(
    "Tool Wear Prediction using FFT + Random Forest"
)


with st.spinner("Training AI Model..."):

    X,Y=load_data()


    if len(X)==0:

        st.error(
            "ไม่พบ Dataset กรุณาวาง archive folder ไว้ข้าง app.py"
        )

        st.stop()


    (
        model,
        scaler,
        X_scaled,
        y_test,
        pred,
        mae,
        r2

    )=create_model(X,Y)



# Prediction

current_wear = model.predict(
    X_scaled[-1].reshape(1,-1)
)[0]


health=max(
    0,
    100*(1-current_wear/FAILURE_WEAR)
)


remaining_cuts=max(
    0,
    FAILURE_WEAR-current_wear
)


remaining_days=remaining_cuts/10



if remaining_days <3:
    status="🔴 CRITICAL"

elif remaining_days <14:
    status="🟡 WARNING"

else:
    status="🟢 NORMAL"



# ==========================
# DASHBOARD
# ==========================


c1,c2,c3,c4=st.columns(4)


c1.metric(
    "Current Wear",
    f"{current_wear:.2f} μm"
)

c2.metric(
    "Health",
    f"{health:.1f}%"
)

c3.metric(
    "Remaining Days",
    f"{remaining_days:.1f}"
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


st.plotly_chart(
    fig,
    use_container_width=True
)



# Prediction chart

st.subheader(
    "Actual vs Prediction"
)


chart=pd.DataFrame({

    "Actual":y_test,

    "Prediction":pred

})


fig=px.line(
    chart,
    markers=True
)


st.plotly_chart(
    fig,
    use_container_width=True
)



# Performance

st.subheader(
    "Model Performance"
)


a,b,c=st.columns(3)


a.metric(
    "R² Score",
    f"{r2:.4f}"
)

b.metric(
    "MAE",
    f"{mae:.4f}"
)

c.metric(
    "Dataset Size",
    len(Y)
)



# Report

st.subheader(
    "AI Recommendation"
)


st.info(
f"""
Current tool wear:
{current_wear:.2f} μm

Tool health:
{health:.1f} %

Estimated remaining life:
{remaining_days:.1f} days

Status:
{status}
"""
)

cd /Users/sheribam/Desktop/cnc_predictive_maintenance