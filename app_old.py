# ----- LIBRARIES -----
import streamlit as st
import pandas as pd
import numpy as np
import re, string, random, joblib
import matplotlib.pyplot as plt
import seaborn as sns
import nltk

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from nltk.corpus import stopwords

# ----- NLTK -----
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# ----- HOME PAGE HEADER -----
st.title("Toxic Comments Classification System")
# st.markdown("""
# <h5>Hello World</h5>
# """, unsafe_allow_html=True)
st.markdown("---")

# ----- LOAD MODELS -----
@st.cache_resource
def load_models():

    lr = joblib.load("../Trained_Models/logistic_model.pkl")
    svm = joblib.load("../Trained_Models/svm_model.pkl")
    tfidf = joblib.load("../Trained_Models/tfidf_vectorizer.pkl")

    try:
        lstm = load_model("../Trained_Models/lstm_model.keras")
        tok = joblib.load("../Trained_Models/lstm_tokenizer.pkl")
    except:
        lstm, tok = None, None


    return lr, svm, tfidf, lstm, tok


lr_model, svm_model, tfidf, lstm_model, tokenizer_lstm = load_models()

df = None

# ----- FILE UPLOAD -----
st.sidebar.header("📁 Upload Dataset")
uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
else:
    pass#st.warning("Please upload file first")
# ----- PREPROCESS FUNCTION -----
def preprocess(text):

    text = str(text)

    text = text.lower()

    text = text.translate(str.maketrans('', '', string.punctuation))

    text = re.sub(r'\d+', '', text)

    text = " ".join([w for w in text.split() if w not in stop_words])

    text = re.sub(r'\s+', ' ', text).strip()

    return text

# ----- SIDEBAR -----
pages = st.sidebar.radio(
    "Modes",
    [
        "Data Preparation",
        "Single",
        "Batch",
        "Models Comparison",
        "About Project"
    ]
)

# =========================================================
# DATA PREPARATION
# =========================================================
if pages == "Data Preparation":

    st.header("📊 Dataset Analysis & Preparation")

    if df is None:

        st.warning("Upload dataset first")

    else:
        text_column = st.selectbox(
            "Text Column",
            df.columns
        )

        label_column = st.selectbox(
            "Label Column",
            df.columns
        )
    
        df[text_column] = (
                                    df[text_column]
                                    .astype(str)
                                    .apply(preprocess)
                                    )
        
        if text_column == label_column:
            st.error("Text column and Label column cannot be same")
            st.stop()

        # ----- ANALYZE -----
        if st.button("Analyze Dataset"):

           # sample_text = " ".join(df[text_column].astype(str).head(100))

        
            st.subheader("🧹 Cleaned Dataset Preview")

            st.dataframe(df.head())

            csv_clean = (
                df
                .to_csv(index=False)
                .encode("utf-8")
            )

            st.download_button(
                label="📥 Download Cleaned Dataset",
                data=csv_clean,
                file_name="cleaned_dataset.csv",
                mime="text/csv"
            )

# =========================================================
# SINGLE PREDICTION
# =========================================================
elif pages == "Single":

    st.header("Single Prediction")

    user_text = st.text_area("Enter Comment")

    model_list = [
        "Logistic Regression",
        "SVM",
        "LSTM"
    ]


    model_choice = st.selectbox(
        "Choose Model",
        model_list
    )

    if st.button("Run Model"):
        MIN_LENGTH = 3
        MAX_LENGTH = 500
        if user_text.strip() == "":
            st.warning("Comment cannot be empty. Please enter a comment")
        
        elif len(user_text.strip()) < MIN_LENGTH:
            st.warning(f"Comment must be at least {MIN_LENGTH} characters.")

        elif len(user_text.strip()) > MAX_LENGTH:
            st.warning(f"Comment cannot exceed {MAX_LENGTH} characters.")

        else:

            text = preprocess(user_text)

            with st.spinner("Running model..."):

                if model_choice in ["Logistic Regression", "SVM"]:

                    X_tfidf = tfidf.transform([text])

                    model = (lr_model if model_choice == "Logistic Regression" else svm_model)

                    prob = model.predict_proba(X_tfidf)[0][1]

                    pred = model.predict(X_tfidf)[0]

                else:

                    seq = tokenizer_lstm.texts_to_sequences([text])

                    X_pad = pad_sequences(seq, maxlen=200)

                    prob = lstm_model.predict(X_pad)[0][0]

                    pred = int(prob > 0.5)

            confidence = (prob * 100 if pred == 1 else (1 - prob) * 100)

            if pred == 1:
                st.error("Toxic Comment")
            else:
                st.success("Non-Toxic Comment")

            st.write(f"### Confidence: {confidence:.2f}%")

# =========================================================
# BATCH
# =========================================================
elif pages == "Batch":

    st.header("Batch Processing")

    if df is None:

        st.warning("Upload dataset from sidebar")

    else:
        df["comment_text"] = (
                                            df["comment_text"]
                                            .astype(str)
                                            .apply(preprocess)
                                            )
        st.dataframe(df.head())

        if "processed_df" in st.session_state:

            text_col = st.session_state.get("text_column")
            label_col = st.session_state.get("label_column")

        else:

            text_col = st.selectbox(
                "Text Column",
                df.columns
            )

            label_col = st.selectbox(
                "Label Column",
                df.columns
            )

        X = df[text_col].astype(str)

        y = df[label_col]

        best_model = st.session_state.get(
            "best_model",
            "Logistic Regression"
        )

        model_choice = st.selectbox(
            "Choose Model",
            ["Logistic Regression", "SVM", "LSTM"]
        )

        if st.button("Run Model"):

            with st.spinner("Running predictions..."):

                if model_choice in ["Logistic Regression", "SVM"]:

                    X_tfidf = tfidf.transform(X)

                    model = (
                        lr_model
                        if model_choice == "Logistic Regression"
                        else svm_model
                    )

                    y_pred = model.predict(X_tfidf)

                    y_prob = model.predict_proba(X_tfidf)[:, 1]


                else:

                    seq = tokenizer_lstm.texts_to_sequences(X)

                    X_pad = pad_sequences(seq, maxlen=200)

                    y_prob = lstm_model.predict(X_pad).flatten()

                    y_pred = (y_prob > 0.5).astype(int)

            results_df = df.copy()

            results_df["prediction"] = y_pred
            results_df["probability"] = y_prob

            st.write(
                f"Accuracy: {accuracy_score(y, y_pred):.2%}"
            )

            st.text(
                classification_report(y, y_pred)
            )

            cm = confusion_matrix(y, y_pred)

            fig, ax = plt.subplots(figsize=(6, 5))

            sns.heatmap(
                cm,
                annot=True,
                fmt="d",
                cmap="Blues",
                cbar=True,
                linewidths=1,
                linecolor="black",
                xticklabels=["Non-Toxic", "Toxic"],
                yticklabels=["Non-Toxic", "Toxic"],
                ax=ax
            )

            # Axis labels
            ax.set_xlabel("Predicted Label", fontsize=12)
            ax.set_ylabel("Actual Label", fontsize=12)

            # Title
            ax.set_title("Confusion Matrix" + "-" + model_choice, fontsize=14)

            # Rotate labels if needed
            ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
            ax.set_yticklabels(ax.get_yticklabels(), rotation=0)

            plt.tight_layout()
                        

            st.pyplot(fig)

            csv = (
                results_df
                .to_csv(index=False)
                .encode('utf-8')
            )

            st.download_button(
                label="📥 Download Results CSV",
                data=csv,
                file_name="prediction_results.csv",
                mime="text/csv"
            )