import streamlit as st
from werkzeug.security import generate_password_hash, check_password_hash
import joblib
import numpy as np
import pandas as pd

# ------------------ PAGE CONFIG ------------------
st.set_page_config(page_title="AI Job Prediction", layout="centered")

# ------------------ LOAD MODELS (CACHED) ------------------
@st.cache_resource
def load_models():
    model = joblib.load("pred_model.pkl")
    degree_encoder = joblib.load("degree2.pkl")
    spec_encoder = joblib.load("specialization2.pkl")
    job_encoder = joblib.load("job2.pkl")
    return model, degree_encoder, spec_encoder, job_encoder

model, degree_encoder, spec_encoder, job_encoder = load_models()

# ------------------ SESSION STORAGE ------------------
if "users" not in st.session_state:
    st.session_state.users = {}

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if "username" not in st.session_state:
    st.session_state.username = ""

if "page" not in st.session_state:
    st.session_state.page = "login"

# ------------------ OPTIONS ------------------
DEGREE_OPTIONS = ["B.Tech", "M.Tech", "B.Sc", "M.Sc"]

SPECIALIZATION_OPTIONS = [
    "Electrical and Electronics Engineering",
    "Electronics and Communication Engineering",
    "Computer Science and Engineering",
    "Mechanical Engineering",
    "Civil Engineering",
    "Biotechnology",
    "Biomedical Engineering",
    "Aeronautical Engineering",
    "Aerospace Engineering",
    "Metallurgical Engineering",
    "Textile Engineering",
    "Marine Engineering",
    "Chemical Engineering",
    "Information Technology",
    "Petroleum Engineering",
    "Environmental Engineering",
    "Mining Engineering",
    "Physics",
    "Chemistry",
    "Mathematics",
    "Statistics",
    "Computer Science",
    "Electronics",
    "Forensic Science",
    "Nursing",
    "Medical Laboratory Technology",
    "Radiology",
    "Operation Theatre Technology",
    "Optometry",
    "Dialysis Technology",
    "Anesthesia Technology"
]

# ------------------ LOGIN PAGE ------------------
def login_page():
    st.title("Login")

    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        users = st.session_state.users

        if username in users and check_password_hash(users[username]["password"], password):
            st.session_state.logged_in = True
            st.session_state.username = username
            st.session_state.page = "dashboard"
        else:
            st.error("Invalid credentials")

    st.write("Don't have an account?")
    if st.button("Go to Register"):
        st.session_state.page = "register"


# ------------------ REGISTER PAGE ------------------
def register_page():
    st.title("Register")

    name = st.text_input("Full Name")
    email = st.text_input("Email")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Register"):

        if not name or not email or not username or not password:
            st.error("All fields are required")
            return

        users = st.session_state.users

        if username in users:
            st.error("Username already exists")
            return

        users[username] = {
            "name": name,
            "email": email,
            "password": generate_password_hash(password),
            "degree": "",
            "specialization": "",
            "cgpa": "",
            "history": []
        }

        st.success("Registration successful!")
        st.session_state.page = "login"

    if st.button("Back to Login"):
        st.session_state.page = "login"


# ------------------ DASHBOARD ------------------
def dashboard():

    user = st.session_state.users[st.session_state.username]

    st.title(f"Welcome, {user['name']}")

    st.subheader("Profile Info")

    st.write("Email:", user["email"])
    st.write("Degree:", user["degree"] if user["degree"] else "Not set")
    st.write("Specialization:", user["specialization"] if user["specialization"] else "Not set")
    st.write("CGPA:", user["cgpa"] if user["cgpa"] else "Not set")

    col1, col2, col3, col4 = st.columns(4)

    if col1.button("Edit Profile"):
        st.session_state.page = "edit_profile"

    if col2.button("Predict Job"):
        st.session_state.page = "predict"

    if col3.button("History"):
        st.session_state.page = "history"

    if col4.button("Settings"):
        st.session_state.page = "settings"

    if st.button("Logout"):
        st.session_state.logged_in = False
        st.session_state.page = "login"


# ------------------ EDIT PROFILE ------------------
def edit_profile():

    user = st.session_state.users[st.session_state.username]

    st.title("Edit Profile")

    degree = st.selectbox("Degree", DEGREE_OPTIONS)
    specialization = st.selectbox("Specialization", SPECIALIZATION_OPTIONS)
    cgpa = st.number_input("CGPA", min_value=0.0, max_value=10.0)

    if st.button("Save Profile"):

        user["degree"] = degree
        user["specialization"] = specialization
        user["cgpa"] = cgpa

        st.success("Profile Updated")

    if st.button("Back"):
        st.session_state.page = "dashboard"


# ------------------ JOB PREDICTION ------------------
def predict_job():

    user = st.session_state.users[st.session_state.username]

    st.title("Job Prediction")

    if not user["degree"] or not user["specialization"]:
        st.error("Please complete your profile first!")
        return

    degree = user["degree"]
    spec = user["specialization"]
    cgpa = float(user["cgpa"])

    if st.button("Predict Job"):

        with st.spinner("Predicting job..."):

            d_encoded = degree_encoder.transform([degree])[0]
            s_encoded = spec_encoder.transform([spec])[0]

            features = np.array([[d_encoded, s_encoded, cgpa]])

            prediction_num = model.predict(features)

            result = job_encoder.inverse_transform(prediction_num)[0]

        st.success(f"Predicted Job: {result}")

        user["history"].append({
            "Degree": degree,
            "Specialization": spec,
            "CGPA": cgpa,
            "Prediction": result
        })

    if st.button("Back"):
        st.session_state.page = "dashboard"


# ------------------ HISTORY ------------------
def history_page():

    user = st.session_state.users[st.session_state.username]

    st.title("Prediction History")

    if len(user["history"]) == 0:
        st.write("No predictions yet.")
    else:
        df = pd.DataFrame(user["history"])
        st.table(df)

    if st.button("Back"):
        st.session_state.page = "dashboard"


# ------------------ SETTINGS ------------------
def settings_page():

    user = st.session_state.users[st.session_state.username]

    st.title("Change Password")

    current = st.text_input("Current Password", type="password")
    new = st.text_input("New Password", type="password")

    if st.button("Update Password"):

        if check_password_hash(user["password"], current):
            user["password"] = generate_password_hash(new)
            st.success("Password Updated")
        else:
            st.error("Current password incorrect")

    if st.button("Back"):
        st.session_state.page = "dashboard"


# ------------------ PAGE ROUTER ------------------
if not st.session_state.logged_in:

    if st.session_state.page == "login":
        login_page()

    elif st.session_state.page == "register":
        register_page()

else:

    if st.session_state.page == "dashboard":
        dashboard()

    elif st.session_state.page == "edit_profile":
        edit_profile()

    elif st.session_state.page == "predict":
        predict_job()

    elif st.session_state.page == "history":
        history_page()

    elif st.session_state.page == "settings":
        settings_page()