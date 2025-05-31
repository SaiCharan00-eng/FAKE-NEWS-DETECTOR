import streamlit as st  # type: ignore
import numpy as np # type: ignore
import pandas as pd  # type: ignore
from sklearn.model_selection import train_test_split  # type: ignore
from sklearn.feature_extraction.text import TfidfVectorizer  # type: ignore
from sklearn.linear_model import PassiveAggressiveClassifier  # type: ignore
from sklearn.metrics import accuracy_score  # type: ignore
from sklearn.metrics.pairwise import cosine_similarity  # type: ignore

# Inject custom CSS for light theme with hover and animation effects
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Rubik:wght@400;600;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Rubik', sans-serif;
        background-color: #f4f4f9;
        color: #333;
        transition: all 0.3s ease-in-out;
    }

    h1, h2, h3, h4 {
        color: #4CAF50;
        font-weight: 800;
    }

    .stTextArea textarea {
        background-color: #ffffff;
        color: #333;
        border: 2px solid #4CAF50;
        border-radius: 10px;
        font-size: 16px;
        font-style: italic;
        padding: 10px;
        transition: all 0.3s ease;
    }

    .stTextArea textarea:hover {
        border-color: #ff4081;
        background-color: #f1f1f1;
    }

    .stButton button {
        background-color: #4CAF50;
        color: white;
        border-radius: 10px;
        padding: 12px 24px;
        font-size: 16px;
        font-weight: bold;
        border: none;
        box-shadow: 0 4px 8px rgba(76,175,80,0.3);
        transition: transform 0.2s ease, background-color 0.3s ease;
    }

    .stButton button:hover {
        background-color: #388e3c;
        color: white;
        transform: scale(1.05);
        cursor: pointer;
        box-shadow: 0 0 12px #388e3c;
    }

    .stCheckbox > label {
        font-size: 16px;
        color: #555555;
    }

    .st-expanderHeader {
        font-size: 18px;
        font-weight: bold;
        color: #4CAF50;
    }

    .stAlert, .stInfo {
        background-color: #ffffff !important;
        color: #333333 !important;
        border-left: 5px solid #4CAF50 !important;
    }

    .stSuccess {
        background-color: #dcedc8 !important;
        border-left: 5px solid #2ecc71 !important;
    }

    .stError {
        background-color: #ffe0e0 !important;
        border-left: 5px solid #e74c3c !important;
    }
    </style>
""", unsafe_allow_html=True)

# Load dataset
@st.cache_data
def load_data():
    fake = pd.read_csv("Fake.csv")
    true = pd.read_csv("True.csv")
    fake["label"] = 0
    true["label"] = 1
    df = pd.concat([fake, true], axis=0)
    df = df.sample(frac=1).reset_index(drop=True)
    return df

df = load_data()

# Preprocessing
X = df['text']
y = df['label']

vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
X_train_text, X_test_text, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train = vectorizer.fit_transform(X_train_text)
X_test = vectorizer.transform(X_test_text)

model = PassiveAggressiveClassifier(max_iter=50)
model.fit(X_train, y_train)

# App layout
st.markdown("<h1 style='text-align: center;'>📰 <b>Fake News Detector</b></h1>", unsafe_allow_html=True)
st.markdown("""
<p style='text-align: center; font-size:18px;'>
    <i>Paste a news article below to detect whether it's</i> 
    <span style='color:#ff4081; font-weight:bold;'>FAKE</span> or 
    <span style='color:#4CAF50; font-weight:bold;'>REAL</span>.
</p>
""", unsafe_allow_html=True)

user_input = st.text_area("✏️ *Enter News Text*", height=200, placeholder="Paste news content here...")

col1, col2 = st.columns([1, 5])
with col1:
    detect = st.button("🔍 Detect")

if detect:
    if user_input.strip() == "":
        st.warning("⚠️ Please enter some news text to analyze.")
    else:
        input_vec = vectorizer.transform([user_input])
        similarities = cosine_similarity(input_vec, X_train)
        max_sim = np.max(similarities)
        threshold = 0.3

        if max_sim < threshold:
            st.error("🤷‍♂️ Can't Predict: This content is too different from training data.")
        else:
            prediction = model.predict(input_vec)
            if prediction[0] == 1:
                st.success("✅ This news appears to be **REAL**.")
            else:
                st.error("🚨 This news appears to be **FAKE**.")

# Accuracy in expander
with st.expander("📊 *Show model accuracy on test set*"):
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    st.info(f"📈 Model Accuracy: **{acc * 100:.2f}%**")
