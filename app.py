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

# ----- SETTING WIDTH OF STREAMLIT MAIN CONTAINER WIDTH -----
st.markdown(
    """
    <style>
    .stMainBlockContainer {
        max-width: 900px;
        padding-left: 2rem;
        padding-right: 2rem;
    }
    </style>
    """,
    unsafe_allow_html=True
)



# ----- HOME PAGE HEADER -----
st.markdown("""
<div style='text-align: center; padding: 10px;'>
    <h1 style='color:rgb(128 12 85); font-size:38px;'>
        Toxic Comment Classification System
    </h1>
    <h5>
        AI-Based Detection of Toxic Online Comments Using ML & Deep Learning
    </h5>
    <p style='font-size:18px;'>
        Analyze, preprocess, classify and compare toxic comment detection models.
    </p>
    <hr style="margin: 0; padding: 0; border: none; border-top: 2px solid green;">
</div>
""", unsafe_allow_html=True)
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

if pages in ["Data Preparation",
        "Batch",
        "Models Comparison"]:
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
    else:
        st.warning("👈Upload dataset from sidebar")

# =========================================================
# DATA PREPARATION
# =========================================================
if pages == "Data Preparation":

    st.header("📊 Dataset Analysis & Preparation")

    if df is None:

        pass

    else:
        text_column = st.selectbox(
            "Text Column",
            df.columns
        )

        label_column = st.selectbox(
            "Label Column",
            df.columns
        )
    
        df[text_column] = (df[text_column].astype(str).apply(preprocess))
        if text_column == label_column:
            st.error("Text column and Label column cannot be same")
            st.stop()

        # ----- ANALYZE -----
        if st.button("Analyze Dataset"):

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


    model_choice = st.selectbox("Choose Model",model_list)

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

        pass

    else:
        df["comment_text"] = (df["comment_text"].astype(str).apply(preprocess))
        st.dataframe(df.head())

        if "processed_df" in st.session_state:
            text_col = st.session_state.get("text_column")
            label_col = st.session_state.get("label_column")

        else:

            text_col = st.selectbox("Text Column",df.columns)
            label_col = st.selectbox("Label Column",df.columns)

        X = df[text_col].astype(str)
        y = df[label_col]

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

            st.write(f"Accuracy: {accuracy_score(y, y_pred):.2%}")

            st.text(classification_report(y, y_pred))

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

            csv = (results_df.to_csv(index=False).encode('utf-8'))

            st.download_button(
                label="📥 Download Results CSV",
                data=csv,
                file_name="prediction_results.csv",
                mime="text/csv"
            )


# =========================================================
# MODEL COMPARISON
# =========================================================
elif pages == "Models Comparison":

    st.header("📊 Model Comparison")
    # =====================================================
    # PREPROCESS TEXT FOR DISPLAY
    # =====================================================
    if df is None:
        pass
    else:
        text_col = st.selectbox("Text Column",df.columns)
        label_col = st.selectbox("Label Column",df.columns)
        display_df = df.copy()
        display_df[text_col] = (display_df[text_col].astype(str).apply(preprocess))

        st.subheader("🧹 Preprocessed Dataset Preview")

        st.dataframe(display_df.head())

        # =====================================================
        # RUN COMPARISON
        # =====================================================

        if st.button("Run Comparison"):

            with st.spinner("Comparing models..."):

                X = (df[text_col].astype(str).apply(preprocess))
                y = pd.to_numeric(df[label_col],errors="coerce")

                # Remove rows with invalid/missing labels
                valid_idx = y.notna()

                X = X[valid_idx]
                y = y[valid_idx].astype(int)

                # Reset indexes so X and y stay aligned
                X = X.reset_index(drop=True)
                y = y.reset_index(drop=True)

                X_tfidf = tfidf.transform(X)

                results = {}
                models_preds = {}

                # =================================================
                # METRICS FUNCTION
                # =================================================

                def metrics(y_true, y_pred, y_prob):

                    return {

                        "accuracy": accuracy_score(
                            y_true,
                            y_pred
                        ),

                        "precision": precision_score(
                            y_true,
                            y_pred,
                            zero_division=0
                        ),

                        "recall": recall_score(
                            y_true,
                            y_pred,
                            zero_division=0
                        ),

                        "f1": f1_score(
                            y_true,
                            y_pred,
                            zero_division=0
                        ),

                        "roc_auc": roc_auc_score(
                            y_true,
                            y_prob
                        )
                    }

                # =================================================
                # LOGISTIC REGRESSION
                # =================================================

                prob_lr = (
                    lr_model
                    .predict_proba(X_tfidf)[:, 1]
                )

                pred_lr = (
                    prob_lr > 0.5
                ).astype(int)

                results["Logistic Regression"] = metrics(
                    y,
                    pred_lr,
                    prob_lr
                )

                models_preds["Logistic Regression"] = pred_lr

                # =================================================
                # SVM
                # =================================================

                prob_svm = (
                    svm_model
                    .predict_proba(X_tfidf)[:, 1]
                )

                pred_svm = (
                    prob_svm > 0.5
                ).astype(int)

                results["SVM"] = metrics(
                    y,
                    pred_svm,
                    prob_svm
                )

                models_preds["SVM"] = pred_svm

                # =================================================
                # LSTM
                # =================================================

                if lstm_model:

                    try:

                        seq = (tokenizer_lstm.texts_to_sequences(X))

                        X_pad = pad_sequences(seq,maxlen=200)

                        prob_lstm = (lstm_model.predict(X_pad).flatten())

                        pred_lstm = (prob_lstm > 0.5).astype(int)

                        results["LSTM"] = metrics(y,pred_lstm,prob_lstm)

                        models_preds["LSTM"] = pred_lstm

                    except Exception as e:

                        st.warning(
                            f"LSTM skipped: {e}"
                        )

                # =====================================================
                # RESULTS
                # =====================================================

                df_res = pd.DataFrame(results).T

                df_res["combined"] = ((df_res["f1"] * 0.6) + (df_res["roc_auc"] * 0.4))

                st.subheader("📊 Model Performance")

                st.dataframe(df_res.style.highlight_max(axis=0))

                # =====================================================
                # DOWNLOAD CSV
                # =====================================================

                csv = (df_res.to_csv().encode("utf-8"))

                st.download_button("⬇️ Download Results",csv,"model_comparison.csv","text/csv")

                # =====================================================
                # BAR CHART
                # =====================================================

                fig, ax = plt.subplots()

                df_res.drop(
                    columns=["combined"]
                ).plot(
                    kind="bar",
                    ax=ax
                )

                ax.set_title(
                    "Model Metrics Comparison"
                )

                ax.set_ylabel("Score")

                ax.set_xticklabels(
                    df_res.index,
                    rotation=0
                )

                for container in ax.containers:

                    ax.bar_label(
                        container,
                        fmt="%.2f",
                        fontsize=8,
                        label_type="center",
                        rotation=90
                    )

                st.pyplot(fig)

                # =====================================================
                # HEATMAP
                # =====================================================

                fig, ax = plt.subplots()

                sns.heatmap(
                    df_res,
                    annot=True,
                    fmt=".2f",
                    cmap="Oranges",
                    ax=ax
                )

                ax.set_title(
                    "Performance Heatmap"
                )

                st.pyplot(fig)

                # =====================================================
                # CONFUSION MATRICES
                # =====================================================

                import math

                n_models = len(models_preds) # 3

                cols = 2

                rows = math.ceil(
                    n_models / cols #3/2= 1.5 = 2
                )

                fig, axes = plt.subplots(
                    rows,
                    cols,
                    figsize=(6 * cols, 4 * rows)
                )

                # Make axes always iterable
                axes = np.array(axes).reshape(-1)

                for ax, (name, preds) in zip(
                    axes,
                    models_preds.items()
                ):

                    cm = confusion_matrix(
                        y,
                        preds
                    )

                    sns.heatmap(
                        cm,
                        annot=True,
                        fmt="d",
                        ax=ax,
                        cbar=False,
                        xticklabels=[
                            "Non-Toxic",
                            "Toxic"
                        ],
                        yticklabels=[
                            "Non-Toxic",
                            "Toxic"
                        ]
                    )

                    ax.set_title(name)

                    ax.set_xlabel(
                        "Predicted"
                    )

                    ax.set_ylabel(
                        "Actual"
                    )

                # Remove empty plots

                for i in range(
                    len(models_preds),
                    len(axes)
                ):

                    fig.delaxes(axes[i])

                plt.tight_layout()

                st.pyplot(fig)
# =========================================================
# ABOUT PAGE
# =========================================================
elif pages == "About Project":

    st.header("ℹ️ About TCC System")

    st.markdown("""
    ## 📌 Project Overview

    The Toxic Comment Classification System is an AI-powered web application
    developed to detect and classify toxic comments using multiple Machine
    Learning and Deep Learning models.

    This system helps in:
    - Detecting harmful or abusive comments
    - Improving online community moderation
    - Comparing ML/DL model performance
    - Automating toxic content analysis

    ---
                """)
        # ----- FEATURES -----
    st.subheader("🚀 System Features")

    col1, col2 = st.columns(2)

    with col1:
        st.info("""
        ✅ Dataset Analysis  
        ✅ Data Cleaning  
        ✅ NLP Preprocessing  
        ✅ Augmentation Techniques  
        """)

    with col2:
        st.info("""
        ✅ Single Prediction  
        ✅ Batch Prediction  
        ✅ Model Comparison  
        ✅ Performance Visualization  
        """)

    st.markdown("---")

    # ----- WORKFLOW -----
    st.subheader("⚙️ System Workflow")

    st.success("""
    1️⃣ Upload Dataset  
    2️⃣ Analyze & Prepare Data  
    3️⃣ Select Prediction Mode  
    4️⃣ Run AI Models  
    5️⃣ View Results & Metrics  
    6️⃣ Download Reports  
    """)
    
    st.markdown("""

    ## 🤖 Models Used

    - Logistic Regression
    - Support Vector Machine (SVM)
    - LSTM Deep Learning Model

    ---

    ## 🛠 Technologies Used

    - Python
    - Streamlit
    - Scikit-learn
    - TensorFlow / Keras
    - NLTK
    - Pandas & NumPy
    - Matplotlib & Seaborn

    ---

    ## 👨‍💻 Developer

    **Muhammad Mohsin Murtaza**  
    Software Engineer

    ---

    ## 🎯 Purpose

    This project has been developed for educational and research purposes
    to explore toxic comment detection using Artificial Intelligence
    and Natural Language Processing (NLP).
    """)


# ----- FOOTER -----
st.markdown("---")

st.markdown(
"""
<div style='text-align: center; color: gray; font-size: 14px;'>
&copy 2026 Toxic Comment Classification System | Developed by Muhammad Mohsin Murtaza
</div>
""",
unsafe_allow_html=True
)