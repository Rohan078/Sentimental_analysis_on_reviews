import streamlit as st
import pickle
import pandas as pd
from preprocess import clean


st.set_page_config(
    page_title="Flipkart Sentiment Analyzer",
    layout="centered",
    initial_sidebar_state="collapsed"
)

st.markdown("""
    <style>
    .main {
        background-color: #f0f2f6;
    }
    .stTextArea textarea {
        background-color: #ffffff !important;
        color: #333333 !important;
        caret-color: #333333 !important;
        border-radius: 10px;
        border: 1px solid #e0e0e0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
    }
    .stTextArea textarea:focus {
        border-color: #4b6cb7;
        box-shadow: 0 0 0 2px rgba(75, 108, 183, 0.2);
    }
    .stButton>button {
        background: linear-gradient(90deg, #4b6cb7 0%, #182848 100%);
        color: white;
        border: none;
        border-radius: 25px;
        height: 50px;
        width: 100%;
        font-weight: bold;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 7px 14px rgba(0,0,0,0.1);
    }
    .result-card {
        padding: 20px;
        border-radius: 15px;
        background-color: white;
        color: #333333;
        box-shadow: 0 10px 20px rgba(0,0,0,0.1);
        text-align: center;
        margin-top: 20px;
        animation: fadeIn 0.5s;
    }
    .result-card p {
        color: #666666;
    }
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }
    .positive { color: #2ecc71 !important; }
    .negative { color: #e74c3c !important; }
    .neutral { color: #f39c12 !important; }
    </style>
""", unsafe_allow_html=True)

# Application Header
st.markdown("<h1 style='text-align: center; color: #182848;'> Product Sentiment Analysis</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #666;'>Analyze customer reviews instantly using AI.</p>", unsafe_allow_html=True)

# Load Model
@st.cache_resource
def load_model():
    try:
        model = pickle.load(open("model.pkl", "rb"))
        vectorizer = pickle.load(open("vector.pkl", "rb"))
        return model, vectorizer
    except FileNotFoundError:
        return None, None

model, vectorizer = load_model()

if model is None:
    st.error("🚨 Model not found! Please run the training script first.")
    st.stop()


review_text = st.text_area(" Enter your review sentiment here...", height=150, placeholder="Example: The product quality is amazing, but the delivery was late.")

if st.button("Analyze Sentiment"):
    if review_text.strip() == "":
        st.warning("Please enter some text to analyze.")
    else:
        with st.spinner("Analyzing..."):
            
            cleaned_text = clean(review_text)
            
           
            vectorized_text = vectorizer.transform([cleaned_text])
            
           
            prediction = model.predict(vectorized_text)[0]
            probability = model.predict_proba(vectorized_text).max() * 100
            
            
            sentiment_color = "neutral"
            if prediction == "Positive":
                sentiment_color = "positive"
            elif prediction == "Negative":
                sentiment_color = "negative"
            
            st.markdown(f"""
                <div class="result-card">
                    <h2 class="{sentiment_color}">{emoji} {prediction}</h2>
                    <p>Confidence Score: <strong>{probability:.2f}%</strong></p>
                </div>
            """, unsafe_allow_html=True)


st.markdown("---")
st.markdown("<p style='text-align: center; font-size: 12px; color: #999;'>Powered by Machine Learning • Deployed on AWS</p>", unsafe_allow_html=True)
