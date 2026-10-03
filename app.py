import streamlit as st
from src.alerts import create_environmental_alert
from src.predict_service import predict_risk
from src.training_ui import render_training_page

st.set_page_config(
    page_title="Smart Environmental Intelligence",
    page_icon=":bar_chart:",
    layout="wide",
)
st.title("Smart Environmental Intelligence Platform")
st.caption("Train, evaluate, and monitor environmental risk models.")

training_tab, dashboard_tab = st.tabs(["Model training", "Risk dashboard"])
with training_tab:
    render_training_page()

with dashboard_tab:
    st.subheader("Environmental risk prediction")
    col1, col2 = st.columns(2)
    with col1:
        temperature = st.number_input("Temperature (deg C)", value=35.0)
        humidity = st.number_input("Humidity (%)", value=60.0)
        wind_speed = st.number_input("Wind Speed (km/h)", value=10.0)
        pm25 = st.number_input("PM2.5", value=55.0)
    with col2:
        pm10 = st.number_input("PM10", value=90.0)
        no2 = st.number_input("NO2", value=40.0)
        aqi = st.number_input("AQI", value=130.0)
    if st.button("Analyze Environmental Risk", type="primary"):
        sensor_values = {
            "Temperature": temperature,
            "Humidity": humidity,
            "WindSpeed": wind_speed,
            "PM2_5": pm25,
            "PM10": pm10,
            "NO2": no2,
            "AQI": aqi,
        }
        result = predict_risk(sensor_values)
        alert = create_environmental_alert(
            result["risk"],
            result["confidence"],
            aqi,
        )
        st.metric("Predicted Risk", result["risk"])
        st.metric("Confidence", f"{result['confidence']:.1%}")
        st.write("Class probabilities", result["probabilities"])
        if alert["level"] == "CRITICAL":
            st.error(alert["message"])
        elif alert["level"] in {"WARNING", "REVIEW"}:
            st.warning(alert["message"])
        else:
            st.success(alert["message"])
