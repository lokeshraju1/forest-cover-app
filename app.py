import joblib
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Forest Cover Type Prediction", page_icon="🌲", layout="centered")

@st.cache_resource
def load_artifacts():
    model = joblib.load("best_random_forest_model.pkl")
    label_encoder = joblib.load("new_label_encoder.pkl")
    pt = joblib.load("power_transformer.pkl")
    scaler = joblib.load("scaler.pkl")
    return model, label_encoder, pt, scaler

model, label_encoder, pt, scaler = load_artifacts()

TOP_FEATURES = [
    "Elevation", "Soil_Type", "Horizontal_Distance_To_Roadways", "Wilderness_Area",
    "Horizontal_Distance_To_Fire_Points", "Horizontal_Distance_To_Hydrology",
    "Hillshade_9am", "Vertical_Distance_To_Hydrology", "Aspect", "Hillshade_Noon",
]
PT_COLS = ["Vertical_Distance_To_Hydrology", "Hillshade_9am", "Hillshade_Noon"]
SCALER_COLS = ["Elevation", "Horizontal_Distance_To_Roadways", "Hillshade_3pm", "Horizontal_Distance_To_Fire_Points"]

COVER_NAMES = {1: "Spruce/Fir", 2: "Lodgepole Pine", 3: "Ponderosa Pine", 4: "Cottonwood/Willow",
               5: "Aspen", 6: "Douglas-fir", 7: "Krummholz"}

def preprocess(raw):
    pt_in = pd.DataFrame([[raw[c] for c in PT_COLS]], columns=PT_COLS)
    pt_out = pt.transform(pt_in)[0]

    sc_vals = dict(raw)
    sc_vals["Hillshade_3pm"] = float(scaler.mean_[SCALER_COLS.index("Hillshade_3pm")])
    sc_in = pd.DataFrame([[sc_vals[c] for c in SCALER_COLS]], columns=SCALER_COLS)
    sc_out = scaler.transform(sc_in)[0]

    row = {
        "Elevation": sc_out[SCALER_COLS.index("Elevation")],
        "Soil_Type": raw["Soil_Type"],
        "Horizontal_Distance_To_Roadways": sc_out[SCALER_COLS.index("Horizontal_Distance_To_Roadways")],
        "Wilderness_Area": raw["Wilderness_Area"],
        "Horizontal_Distance_To_Fire_Points": sc_out[SCALER_COLS.index("Horizontal_Distance_To_Fire_Points")],
        "Horizontal_Distance_To_Hydrology": np.log1p(raw["Horizontal_Distance_To_Hydrology"]),
        "Hillshade_9am": pt_out[PT_COLS.index("Hillshade_9am")],
        "Vertical_Distance_To_Hydrology": pt_out[PT_COLS.index("Vertical_Distance_To_Hydrology")],
        "Aspect": np.log1p(raw["Aspect"]),
        "Hillshade_Noon": pt_out[PT_COLS.index("Hillshade_Noon")],
    }
    return pd.DataFrame([row])[TOP_FEATURES]

st.title("🌲 Forest Cover Type Prediction")
st.write("Enter the normal (original) values. Scaling and transforms are applied automatically.")

with st.form("input_form"):
    c1, c2 = st.columns(2)
    with c1:
        elevation = st.number_input("Elevation (m)", 1800.0, 4000.0, 2900.0, 10.0)
        aspect = st.number_input("Aspect (degrees)", 0.0, 360.0, 150.0, 1.0)
        h_hydro = st.number_input("Horizontal Distance To Hydrology", 0.0, 1500.0, 250.0, 10.0)
        v_hydro = st.number_input("Vertical Distance To Hydrology", -400.0, 700.0, 40.0, 5.0)
        h_road = st.number_input("Horizontal Distance To Roadways", 0.0, 7500.0, 2500.0, 50.0)
    with c2:
        h_fire = st.number_input("Horizontal Distance To Fire Points", 0.0, 7500.0, 2000.0, 50.0)
        hs9 = st.number_input("Hillshade 9am (0-255)", 0.0, 255.0, 212.0, 1.0)
        hs_noon = st.number_input("Hillshade Noon (0-255)", 0.0, 255.0, 223.0, 1.0)
        wilderness = st.selectbox("Wilderness Area", [1, 2, 3, 4])
        soil = st.selectbox("Soil Type", list(range(1, 41)))
    submit = st.form_submit_button("🔍 Predict Forest Cover")

if submit:
    raw = {
        "Elevation": elevation, "Aspect": aspect,
        "Horizontal_Distance_To_Hydrology": h_hydro, "Vertical_Distance_To_Hydrology": v_hydro,
        "Horizontal_Distance_To_Roadways": h_road, "Horizontal_Distance_To_Fire_Points": h_fire,
        "Hillshade_9am": hs9, "Hillshade_Noon": hs_noon,
        "Wilderness_Area": wilderness, "Soil_Type": soil,
    }
    X_in = preprocess(raw)

    pred_label = label_encoder.inverse_transform(model.predict(X_in))[0]
    st.success(f"🌲 Predicted Forest Cover Type: **{pred_label}**")

    proba = model.predict_proba(X_in)[0]
    labels = label_encoder.inverse_transform(model.classes_)
    prob_df = pd.DataFrame({
        "Cover Type": labels,
        "Probability (%)": (proba * 100).round(2),
    }).sort_values("Probability (%)", ascending=False)
    st.dataframe(prob_df, hide_index=True, use_container_width=True)
