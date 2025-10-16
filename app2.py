import streamlit as st
import pandas as pd
import numpy as np
import pickle


model = pickle.load(open('personality_model.pkl', 'rb'))


st.set_page_config(page_title="Personality Classifier", page_icon="🧠")
st.title("🧠 Personality Classifier")
st.markdown("Predict whether a person is more likely to be an **Introvert** or **Extrovert** using their behavioral traits.")


with st.form("input_form"):
    Time_spent_Alone = st.slider("🕒 Time spent alone per day (hours)", 0, 12, 3)
    Social_event_attendance = st.slider("🎉 Social events attended per month", 0, 10, 3)
    Going_outside = st.slider("🌳 Times going outside per week", 0, 7, 3)
    Friends_circle_size = st.slider("👥 Number of close friends", 0, 20, 5)
    Post_frequency = st.slider("📱 Social media posts per week", 0, 10, 2)
    Stage_fear = st.selectbox("🎤 Do you have stage fear?", ["Yes", "No"])
    Drained_after_socializing = st.selectbox("😩 Do you feel drained after socializing?", ["Yes", "No"])
    submit = st.form_submit_button("🔍 Predict Personality")


if submit:
    input_data = np.array([[
        Time_spent_Alone,
        Social_event_attendance,
        Going_outside,
        Friends_circle_size,
        Post_frequency,
        1 if Stage_fear == "Yes" else 0,
        1 if Drained_after_socializing == "Yes" else 0,
    ]])

    prediction = model.predict(input_data)[0]
    result = "Introvert" if prediction == 1 else "Extrovert"

    st.success(f"✅ This person is likely an **{result}**")


st.markdown("---")
st.markdown("<div style='text-align: center; color: gray;'>🔧 Built by <b>Dev Vaghani</b> using Python, Scikit-learn & Streamlit</div>", unsafe_allow_html=True)
