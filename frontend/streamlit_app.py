import streamlit as st
import requests
import os

# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Fraud Detection | Fintech ML",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

API_URL = os.getenv("API_URL", "http://localhost:8000")
THRESHOLD = 0.95

# --------------------------------------------------
# CUSTOM CSS
# --------------------------------------------------

st.markdown("""

<style>
/* Main application */
.stApp {
    background-color: #F4F6FA;
    color: #172B4D;
}

[data-testid="stAppViewContainer"] {
    background-color: #F4F6FA;
}

.block-container {
    max-width: 1200px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}

/* Force readable main-page text */
[data-testid="stAppViewContainer"] .main p,
[data-testid="stAppViewContainer"] .main label,
[data-testid="stAppViewContainer"] .main span,
[data-testid="stAppViewContainer"] .main li {
    color: #26364F;
}

h1, h2, h3, h4 {
    color: #172B4D !important;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background-color: #101D35;
}

[data-testid="stSidebar"] * {
    color: #FFFFFF;
}

/* Sidebar secondary text */
[data-testid="stSidebar"] [data-testid="stCaptionContainer"] p {
    color: #CBD5E1 !important;
}

/* Input labels */
.stSelectbox label,
.stNumberInput label {
    color: #26364F !important;
    font-weight: 600;
}

/* Input boxes */
[data-baseweb="input"] input {
    color: #172B4D !important;
    background-color: #FFFFFF !important;
}

[data-baseweb="select"] * {
    color: #172B4D !important;
}

/* Metrics */
[data-testid="stMetric"] {
    background-color: #FFFFFF;
    border: 1px solid #DCE3ED;
    border-radius: 12px;
    padding: 20px;
}

[data-testid="stMetricLabel"] {
    color: #52627A !important;
}

[data-testid="stMetricValue"] {
    color: #172B4D !important;
}

/* Buttons */
.stFormSubmitButton button {
    background-color: #2457D6 !important;
    color: #FFFFFF !important;
    border: none;
    border-radius: 8px;
    min-height: 45px;
    font-weight: 600;
}

/* Result cards */
.result-warning {
    background-color: #FFF5E9;
    border-left: 5px solid #E58A24;
    padding: 20px;
    border-radius: 8px;
}

.result-normal {
    background-color: #EAF7EF;
    border-left: 5px solid #29965B;
    padding: 20px;
    border-radius: 8px;
}

.result-warning h3 {
    color: #914B0B !important;
}

.result-warning p {
    color: #694214 !important;
}

.result-normal h3 {
    color: #207447 !important;
}

.result-normal p {
    color: #315D45 !important;
}

/* Captions */
[data-testid="stCaptionContainer"] p {
    color: #64748B !important;
}

/* Divider */
hr {
    border-color: #DCE3ED;
}
</style>
""", unsafe_allow_html=True)


# --------------------------------------------------
# API CONNECTION CHECK
# --------------------------------------------------

def check_api():
    try:
        response = requests.get(
            f"{API_URL}/health",
            timeout=3
        )

        if response.status_code == 200:
            return response.json()

        return None

    except requests.exceptions.RequestException:
        return None


api_status = check_api()


# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------

with st.sidebar:

    st.markdown("# 🛡️")
    st.markdown("## Fraud Detection")
    st.caption("Digital Payment Risk Assessment")

    st.divider()

    st.markdown("### MODEL INFORMATION")

    st.markdown("**Algorithm**")
    st.write("XGBoost Classifier")

    st.markdown("**Model version**")
    st.write("Version 1")

    st.markdown("**Decision threshold**")
    st.write("0.95")

    st.divider()

    st.markdown("### SYSTEM STATUS")

    if api_status:
        st.success("API Connected")
    else:
        st.error("API Disconnected")

    st.divider()

    st.caption("Built with:")
    st.caption("Python · XGBoost · MLflow")
    st.caption("FastAPI · Streamlit · Docker")


# --------------------------------------------------
# MAIN HEADER
# --------------------------------------------------


st.title("Transaction Risk Assessment")

st.markdown(
    "Evaluate digital payment transactions using a trained "
    "machine learning model."
)

st.write(
    "Enter the transaction details below to obtain a model-generated "
    "risk score and review recommendation."
)

st.divider()


# --------------------------------------------------
# TRANSACTION INPUT FORM
# --------------------------------------------------

st.subheader("Transaction details")

st.caption(
    "Provide the transaction information using the values "
    "available before the transaction."
)

with st.form("transaction_form"):

    col1, col2 = st.columns(2)

    with col1:

        transaction_type = st.selectbox(
            "Transaction type",
            [
                "CASH_IN",
                "CASH_OUT",
                "DEBIT",
                "PAYMENT",
                "TRANSFER"
            ]
        )

        amount = st.number_input(
            "Transaction amount",
            min_value=0.0,
            value=1000.0,
            step=100.0,
            help="The amount involved in the transaction."
        )

    with col2:

        oldbalance_org = st.number_input(
            "Origin balance before transaction",
            min_value=0.0,
            value=1000.0,
            step=100.0
        )

        oldbalance_dest = st.number_input(
            "Destination balance before transaction",
            min_value=0.0,
            value=0.0,
            step=100.0
        )

    st.write("")

    submitted = st.form_submit_button(
        "Assess transaction",
        use_container_width=True
    )


# --------------------------------------------------
# PREDICTION
# --------------------------------------------------

if submitted:

    payload = {
        "type": transaction_type,
        "amount": amount,
        "oldbalanceOrg": oldbalance_org,
        "oldbalanceDest": oldbalance_dest
    }

    try:

        with st.spinner("Evaluating transaction..."):

            response = requests.post(
                f"{API_URL}/predict",
                json=payload,
                timeout=30
            )

        if response.status_code == 200:

            result = response.json()

            prediction = result["fraud_prediction"]
            risk_score = result["risk_score"]
            threshold = result["threshold"]

            st.divider()

            st.subheader("Assessment result")

            # Result summary
            score_col, threshold_col = st.columns(2)

            with score_col:
                st.metric(
                    "Model risk score",
                    f"{risk_score:.4f}"
                )

            with threshold_col:
                st.metric(
                    "Decision threshold",
                    f"{threshold:.2f}"
                )

            # Visual score indicator
            st.markdown("**Model score indicator**")

            st.progress(
                min(max(float(risk_score), 0.0), 1.0)
            )

            st.caption(
                "The indicator shows the model score on a 0–1 scale. "
                "It is not a calibrated probability."
            )

            st.write("")

            # Prediction outcome
            if prediction == 1:

                st.markdown(
                    f"""
                    <div class="result-warning">
                        <h3 style="color:#A65B12;">
                            ⚠️ Flag for review
                        </h3>
                        <p style="color:#694214;">
                            The model score meets or exceeds the
                            selected decision threshold.
                            Further investigation is recommended.
                        </p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            else:

                st.markdown(
                    f"""
                    <div class="result-normal">
                        <h3 style="color:#207447;">
                            ✓ No flag at selected threshold
                        </h3>
                        <p style="color:#315D45;">
                            The model score is below the selected
                            decision threshold.
                            This does not guarantee that the
                            transaction is legitimate.
                        </p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            # Transaction summary
            with st.expander("View submitted transaction details"):

                st.json(payload)

        else:

            st.error(
                f"API returned HTTP {response.status_code}: "
                f"{response.text}"
            )

    except requests.exceptions.ConnectionError:

        st.error(
            "Unable to connect to the FastAPI backend. "
            "Check that your Docker container is running."
        )

    except requests.exceptions.Timeout:

        st.error("The prediction request timed out.")

    except requests.exceptions.RequestException as error:

        st.error(f"Request failed: {error}")

    except (KeyError, ValueError, TypeError):

        st.error(
            "The API response could not be processed. "
            "Check the backend response format."
        )


# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.divider()

footer_col1, footer_col2 = st.columns(2)

with footer_col1:
    st.caption("Digital Payment Fraud Detection System")

with footer_col2:
    st.caption("PaySim synthetic dataset · Portfolio project")