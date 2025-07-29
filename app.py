import numpy as np
import pandas as pd
import joblib
import base64
import streamlit as st
import pyttsx3
import xgboost as xgb
import os
from fpdf import FPDF

def set_background_image(image_file):
    if not os.path.exists(image_file):
        st.error(f"Background image not found: {image_file}")
        return

    with open(image_file, "rb") as image:
        encoded_string = base64.b64encode(image.read()).decode()
    background_image = f"""
    <style>
    .stApp {{
        background-image: url('data:image/jpg;base64,{encoded_string}');
        background-size: cover;
    }}
    </style>
    """
    st.markdown(background_image, unsafe_allow_html=True)

def load_model(file_path):
    try:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Model file not found at: {file_path}")

        if file_path.endswith('scaler.pkl'):
            model = joblib.load(file_path)
        elif file_path.endswith('new_model.json'):
            model = xgb.XGBClassifier()
            model.load_model(file_path)
        else:
            raise ValueError("Unsupported model format. Please use .pkl or .json")
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.stop()

def predict_stage(inputs):
    scaler = load_model('scaler.pkl')
    model = load_model('new_model.json')

    feature_names = scaler.feature_names_in_
    data_df = pd.DataFrame([inputs], columns=feature_names)

    if not all(col in feature_names for col in data_df.columns):
        missing_cols = [col for col in data_df.columns if col not in feature_names]
        raise ValueError(f"Missing or invalid columns: {missing_cols}")

    data_scaled = scaler.transform(data_df)
    prediction = model.predict(data_scaled)

    return int(prediction[0])

def provide_recommendations(stage):
    recommendations = {
        0: "**Stage 0:** Maintain a balanced diet, avoid alcohol, and undergo regular health check-ups.",
        1: "**Stage 1:** Reduce salt intake, monitor liver function, and take prescribed medications.",
        2: "**Stage 2:** Follow a low-sodium diet, take diuretics as prescribed, and manage symptoms promptly.",
        3: "**Stage 3:** Hospitalization may be required. Strict dietary control and close monitoring are essential.",
        4: "**Stage 4:** Advanced liver disease. Consider liver transplantation and manage complications with medical support."
    }
    return recommendations.get(stage, "Invalid Stage Predicted.")

def provide_diet_plan(stage):
    diet_plans = {
        0: "**Stage 0:** Focus on a balanced diet rich in fruits, vegetables, whole grains, lean proteins, and low-fat dairy. Stay hydrated and avoid alcohol.",
        1: "**Stage 1:** Reduce salt intake. Eat fresh fruits, vegetables, lean meats, fish, and low-fat dairy. Limit processed foods.",
        2: "**Stage 2:** Follow a low-sodium diet. Increase intake of leafy greens, berries, and high-fiber foods. Avoid alcohol and fried foods.",
        3: "**Stage 3:** Low-protein, low-sodium diet. Eat boiled vegetables, rice, and lean poultry. Avoid red meat and alcohol.",
        4: "**Stage 4:** Nutrient-dense, low-sodium foods. Consume soft-cooked vegetables, soups, and low-fat yogurt. Protein intake may be restricted."
    }
    return diet_plans.get(stage, "No specific diet plan available.")

def generate_pdf(inputs, stage, recommendation_text, diet_plan_text):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)

    pdf.cell(200, 10, txt="Liver Cirrhosis Stage Prediction Report", ln=True, align='C')
    pdf.ln(10)

    pdf.set_font("Arial", size=10)
    pdf.cell(200, 10, txt="Patient Details and Input Values:", ln=True)
    for key, value in zip(inputs.keys(), inputs.values()):
        pdf.cell(200, 8, txt=f"{key}: {value}", ln=True)
    
    pdf.ln(10)
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt=f"Predicted Stage: {stage}", ln=True)

    pdf.ln(10)
    pdf.set_font("Arial", size=10)
    pdf.multi_cell(0, 8, f"Recommendations: {recommendation_text}")
    pdf.ln(5)
    pdf.multi_cell(0, 8, f"Diet Plan: {diet_plan_text}")

    pdf_file = "Liver_Cirrhosis_Report.pdf"
    pdf.output(pdf_file)
    st.success("PDF report generated successfully!")
    st.download_button(label="Download Report", data=open(pdf_file, "rb"), file_name=pdf_file, mime="application/pdf")

def speak(text):
    engine = pyttsx3.init()
    voices = engine.getProperty('voices')

    # Select a female voice, preferably Indian
    for voice in voices:
        if 'female' in voice.name.lower() and ('india' in voice.name.lower() or 'indian' in voice.name.lower()):
            engine.setProperty('voice', voice.id)
            break
        elif 'female' in voice.name.lower():
            engine.setProperty('voice', voice.id)

    engine.setProperty('rate', 140)  # Adjust speech speed
    engine.say(text)
    engine.runAndWait()

def main():
    st.title('Liver Cirrhosis Stage Prediction')
    set_background_image('images/img.jpg')

    # Sidebar navigation
    st.sidebar.title("Navigation")
    option = st.sidebar.radio("Select a section", ["Home", "Symptoms", "Medicines", "Prediction"])

    # Display content based on selected option
    if option == "Home":
        st.header("Welcome to the Liver Cirrhosis Prediction System")
        st.write("This system predicts the stage of liver cirrhosis based on patient data.")
    
    elif option == "Symptoms":
        st.header("Symptoms of Liver Cirrhosis")
        st.write("""
            - Fatigue
            - Jaundice (yellowing of skin or eyes)
            - Abdominal swelling
            - Nausea and vomiting
            - Weight loss
            - Dark urine and pale stools
            """)
    
    elif option == "Medicines":
        st.header("Medicines for Liver Cirrhosis")
        st.write("""
            - Diuretics (to reduce fluid buildup)
            - Lactulose (to reduce the risk of hepatic encephalopathy)
            - Beta-blockers (to reduce the risk of bleeding)
            - Antiviral medications (for chronic hepatitis B)
            """)
    
    elif option == "Prediction":
        inputs = {}
        scaler = load_model('scaler.pkl')
        for feature in scaler.feature_names_in_:
            if feature == 'Drug':
                value = st.selectbox(f"Select {feature}", ['Placebo', 'D-penicillamine'])
                inputs[feature] = 0 if value == 'Placebo' else 1
            elif feature == 'Sex':
                value = st.selectbox(f"Select {feature}", ['Female', 'Male'])
                inputs[feature] = 0 if value == 'Female' else 1
            else:
                value = st.text_input(f"Enter {feature}")
                if value.strip() == "":
                    st.warning(f"Please enter a value for {feature}")
                    return
                try:
                    inputs[feature] = float(value) if '.' in value or 'e' in value else int(value)
                except ValueError:
                    st.error(f"Invalid input for {feature}. Please enter a valid number.")
                    return

        if st.button('Predict Stage'):
            try:
                result = predict_stage(list(inputs.values()))
                st.success(f"Predicted Stage: {result}")
                recommendation_text = provide_recommendations(result)
                diet_plan_text = provide_diet_plan(result)
                st.info(recommendation_text)
                st.info(f"**Diet Plan:** {diet_plan_text}")
                speak(f"The predicted stage is {result}. {recommendation_text} {diet_plan_text}")
                generate_pdf(inputs, result, recommendation_text, diet_plan_text)
            except Exception as e:
                st.error(f"Prediction Error: {e}")

if __name__ == "__main__":
    main()
