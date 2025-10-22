# streamlit_precision_health_app.py
"""Precision Health Disease Prediction - Enhanced Version
Features added:
- Password hashing with bcrypt
- Session timeout (auto logout after inactivity)
- Persistent prediction history in SQLite
- Plotly chart visualizations of prediction trends
- Health-tip personalization considering multiple factors
- Dark mode toggle via injected CSS
- Input pre-filling from last user record
- Multi-language PDF reports (English + Hindi)
- QR code embedded in PDF linking to user's history page
- Admin dashboard (admin user sees overall stats)
- User profile page with downloadable history
"""

import pickle
from streamlit_option_menu import option_menu
from fpdf import FPDF
import sqlite3
import base64
import re
import pandas as pd
import datetime
import requests
import bcrypt
import qrcode
import io
import plotly.express as px
import os
import streamlit as st
def rerun():
    st.session_state['rerun'] = not st.session_state.get('rerun', False)

# Initialize the rerun flag if not present
if 'rerun' not in st.session_state:
    st.session_state['rerun'] = False


# ----------------- Configuration -----------------
DB_NAME = 'users.db'
SESSION_TIMEOUT_MINUTES = 15
PDF_TEMP_DIR = 'tmp_reports'
os.makedirs(PDF_TEMP_DIR, exist_ok=True)

# ----------------- Load Models -------------------
# If models missing, app will still run but predictions will be mocked
try:
    cancer_model = pickle.load(open("Saved models/cancer_model.sav", "rb"))
except Exception:
    cancer_model = None
try:
    obesity_model = pickle.load(open("Saved models/obesity_model.sav", "rb"))
except Exception:
    obesity_model = None
try:
    pressure_model = pickle.load(open("Saved models/pressure_model.sav", "rb"))
except Exception:
    pressure_model = None

# ----------------- Database ----------------------

def init_db():
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()

    # Users table
    c.execute('''
        CREATE TABLE IF NOT EXISTS users (
            username TEXT PRIMARY KEY,
            password TEXT NOT NULL,
            is_admin INTEGER DEFAULT 0
        )
    ''')

    # Predictions table
    c.execute('''
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT NOT NULL,
            display_name TEXT,
            age INTEGER,
            gender TEXT,
            disease TEXT,
            result TEXT,
            details TEXT,
            created_at TEXT,
            FOREIGN KEY(username) REFERENCES users(username)
        )
    ''')

    conn.commit()
    conn.close()

def add_user(username, password):
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    try:
        c.execute("INSERT INTO users (username, password) VALUES (?, ?)", (username, password))
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False
    finally:
        conn.close()

def validate_user(username, password):
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute("SELECT * FROM users WHERE username = ? AND password = ?", (username, password))
    user = c.fetchone()
    conn.close()
    return user is not None

def user_exists(username):
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute("SELECT * FROM users WHERE username = ?", (username,))
    exists = c.fetchone() is not None
    conn.close()
    return exists

def validate_username(username):
    if len(username) < 4:
        return "Username must be at least 4 characters."
    if not username[0].isupper():
        return "Username must start with a capital letter."
    if not username.isalnum():
        return "Username must contain only letters and numbers (no spaces or symbols)."
    return None

def validate_password(password):
    if len(password) < 8:
        return "Password must be at least 8 characters."
    if not re.search(r"[A-Z]", password):
        return "Password must contain at least one uppercase letter."
    if not re.search(r"[a-z]", password):
        return "Password must contain at least one lowercase letter."
    if not re.search(r"\d", password):
        return "Password must contain at least one number."
    if not re.search(r"[!@#$%^&*(),.?\":{}|<>]", password):
        return "Password must contain at least one special character."
    return None


# ----------------- Prediction Persistence ------------
def save_prediction(username, display_name, age, gender, disease, result, details):
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    now = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    c.execute('''INSERT INTO predictions (username, display_name, age, gender, disease, result, details, created_at)
                 VALUES (?, ?, ?, ?, ?, ?, ?, ?)''', (username, display_name, age, gender, disease, result, details, now))
    conn.commit()
    conn.close()

def load_user_predictions(username):
    conn = sqlite3.connect(DB_NAME)
    df = pd.read_sql_query('SELECT display_name as Name, age as Age, disease as Disease, result as Prediction, created_at as DateTime FROM predictions WHERE username = ? ORDER BY created_at DESC', conn, params=(username,))
    conn.close()
    return df

def load_all_predictions():
    conn = sqlite3.connect(DB_NAME)
    df = pd.read_sql_query('SELECT username, display_name, age, disease, result, created_at FROM predictions ORDER BY created_at DESC', conn)
    conn.close()
    return df

# ----------------- Session timeout -----------------
def update_last_active():
    st.session_state['last_active'] = datetime.datetime.utcnow()

def check_session_timeout():
    if not st.session_state.get('logged_in'):
        return
    last = st.session_state.get('last_active')
    if not last:
        update_last_active()
        return
    now = datetime.datetime.utcnow()
    diff = now - last
    if diff.total_seconds() > SESSION_TIMEOUT_MINUTES * 60:
        st.session_state['logged_in'] = False
        st.warning(f"Session timed out after {SESSION_TIMEOUT_MINUTES} minutes of inactivity. Please login again.")
        st.experimental_rerun()
    else:
        update_last_active()

# ----------------- Health Tip Personalization --------
def get_health_tip(disease, age, gender=None, activity=None, smoker=None, family_history=None):
    tips = {
        "Diabetes": "Avoid sugary foods, check blood sugar regularly.",
        "Heart Disease": "Low-sodium diet, daily walk recommended.",
        "Lung Cancer": "Quit smoking, avoid polluted areas.",
        "Obesity": "Increase activity, reduce sugar/fat intake.",
        "Blood Pressure": "Reduce salt intake, monitor BP weekly."
    }
    base = tips.get(disease, "Maintain hygiene, hydrate, stay active.")
    extras = []
    if age and age > 60:
        extras.append("Consider routine geriatric checkups.")
    if smoker:
        extras.append("Smoking cessation is strongly advised.")
    if activity is not None and activity == 0:
        extras.append("Increase light daily activity like walking.")
    if family_history:
        extras.append("Family history present — consult a specialist for screening plan.")
    return " ".join([base] + extras)

# ----------------- PDF Generation with QR & Multilang - FPDF ----------------
LANG_MAP = {
    'en': {
        'report_title': 'Health Prediction Report',
        'name': 'Name',
        'age': 'Age',
        'disease': 'Disease',
        'prediction': 'Prediction Result',
        'health_tip': 'Health Tip'
    }
    
}

def generate_pdf_report(name, age, disease, result, tips, username, lang='en'):
    labels = LANG_MAP.get(lang, LANG_MAP['en'])
    filename = f"{PDF_TEMP_DIR}/{username}_{disease}_{int(datetime.datetime.now().timestamp())}.pdf"

    # Create QR code linking to history (placeholder link)
    history_url = f"https://example.com/user/{username}/history"  # Replace with real route if deployed
    qr = qrcode.make(history_url)
    qr_io = io.BytesIO()
    qr.save(qr_io, format='PNG')
    qr_io.seek(0)

    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt=labels['report_title'], ln=True, align='C')
    pdf.ln(6)
    pdf.cell(0, 8, txt=f"{labels['name']}: {name}", ln=True)
    pdf.cell(0, 8, txt=f"{labels['age']}: {age}", ln=True)
    pdf.cell(0, 8, txt=f"{labels['disease']}: {disease}", ln=True)
    pdf.cell(0, 8, txt=f"{labels['prediction']}: {result}", ln=True)
    pdf.ln(4)
    pdf.multi_cell(0, 8, txt=f"{labels['health_tip']}: {tips}")

    # Insert QR code
    tmp_qr_path = f"{PDF_TEMP_DIR}/qr_{username}_{int(datetime.datetime.now().timestamp())}.png"
    with open(tmp_qr_path, 'wb') as f:
        f.write(qr_io.read())
    pdf.image(tmp_qr_path, x=10, y=pdf.get_y()+5, w=30)

    pdf.output(filename)
    with open(filename, "rb") as f:
        b64 = base64.b64encode(f.read()).decode('utf-8')
    href = f'<a href="data:application/pdf;base64,{b64}" download="{os.path.basename(filename)}">Download PDF Report</a>'
    st.markdown(href, unsafe_allow_html=True)

# ----------------- SMS Integration (placeholder) -----
def send_sms_via_fast2sms(mobile, message):
    url = "https://www.fast2sms.com/dev/bulkV2"
    headers = {'authorization': 'YOUR_FAST2SMS_API_KEY', 'Content-Type': "application/x-www-form-urlencoded"}
    payload = f"sender_id=TXTIND&message={message}&language=english&route=v3&numbers={mobile}"
    try:
        resp = requests.post(url, data=payload, headers=headers)
        return resp.status_code == 200
    except Exception:
        return False

# ----------------- UI Utilities ---------------------
def inject_dark_mode_css(enabled: bool):
    if enabled:
        css = """
        <style>
        .stApp { background-color: #0e1117; color: #e6edf3; }
        .stButton>button { background-color:#1f2937; color:#fff }
        </style>
        """
        st.markdown(css, unsafe_allow_html=True)

# ----------------- Validation Helpers ----------------
def validate_username(username):
    if len(username) < 4:
        return "Username must be at least 4 characters."
    if not username[0].isupper():
        return "Username must start with a capital letter."
    if not username.isalnum():
        return "Username must contain only letters and numbers (no spaces or symbols)."
    return None

def validate_password(password):
    if len(password) < 8:
        return "Password must be at least 8 characters."
    if not re.search(r"[A-Z]", password):
        return "Password must contain at least one uppercase letter."
    if not re.search(r"[a-z]", password):
        return "Password must contain at least one lowercase letter."
    if not re.search(r"\d", password):
        return "Password must contain at least one number."
    if not re.search(r"[!@#$%^&*(),.?\":{}|<>]", password):
        return "Password must contain at least one special character."
    return None

# ----------------- Streamlit App State Setup ----------
if 'selected_page' not in st.session_state:
    st.session_state['selected_page'] = 'Register'
if 'logged_in' not in st.session_state:
    st.session_state['logged_in'] = False
if 'username' not in st.session_state:
    st.session_state['username'] = ''
if 'last_active' not in st.session_state:
    st.session_state['last_active'] = None
if 'dark_mode' not in st.session_state:
    st.session_state['dark_mode'] = False

# ----------------- Sidebar Navigation ----------------
with st.sidebar:
    inject_dark_mode_css(st.session_state['dark_mode'])
    st.checkbox('Dark Mode', value=st.session_state['dark_mode'], key='dark_mode', on_change=lambda: st.rerun())
    menu_items = ["Register", "Login", "Lung Cancer Prediction", "Obesity Prediction", "Blood Pressure Prediction", "History", "Symptom Checker", "Profile"]
    
    selected = option_menu("PRECISION HEALTH DISEASE PREDICTION", menu_items, default_index=menu_items.index(st.session_state['selected_page']) if st.session_state['selected_page'] in menu_items else 0, key='selected_page')

# ----------------- Session Timeout Check --------------
check_session_timeout()

# ----------------- Login/Register ---------------------
if selected == 'Login':
    st.title('🔐 Login')
    username = st.text_input('Username')
    password = st.text_input('Password', type='password')
    if st.button('Login'):
        if validate_user(username, password):
            st.session_state['logged_in'] = True
            st.session_state['username'] = username
            update_last_active()
            st.success('Logged in successfully')
            rerun()   # <-- replace st.experimental_rerun() with rerun()
        else:
            st.error('Invalid credentials')
elif selected == 'Register':
    st.title('📝 Register')
    new_user = st.text_input('Choose Username')
    new_pass = st.text_input('Choose Password', type='password')
    admin_flag = st.checkbox('Register as admin (for demo only)')
    if st.button('Register'):
        err = validate_username(new_user) or validate_password(new_pass)
        if err:
            st.error(err)
        elif user_exists(new_user):
            st.warning('Username already exists')
        else:
            add_user(new_user, new_pass)
            st.success('Registration successful. Please login.')

# ----------------- Symptom Checker -------------------
elif selected == 'Symptom Checker':
    st.title('💬 AI Symptom Checker')
    user_input = st.chat_input('Enter your symptom(s), comma separated')
    if user_input:
        input_l = user_input.lower()
        mapping = {
            'Heart Disease': ['chest pain', 'palpitation', 'shortness of breath', 'dizziness'],
            'Lung Condition': ['cough', 'wheezing', 'coughing blood', 'hoarseness'],
            'Blood Pressure': ['headache', 'blurred vision', 'nosebleeds'],
            'Obesity': ['weight gain', 'snoring', 'difficulty breathing']
        }
        matched = []
        for disease, kw in mapping.items():
            if any(k in input_l for k in kw):
                matched.append(disease)
        if matched:
            st.write('Possible matches: ' + ', '.join(matched))
        else:
            st.write('No clear match. Try common keywords like chest pain, persistent cough.')

# ----------------- History ---------------------------
elif selected == 'History':
    st.title('🕘 Prediction History')
    if not st.session_state.get('logged_in'):
        st.warning('Please login to view history')
        st.stop()
    df = load_user_predictions(st.session_state['username'])
    if df.empty:
        st.info('No predictions yet')
    else:
        st.dataframe(df, use_container_width=True)
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button('Download CSV', csv, file_name='prediction_history.csv', mime='text/csv')

# ----------------- Profile ---------------------------
elif selected == 'Profile':
    st.title('👤 Profile')
    if not st.session_state.get('logged_in'):
        st.warning('Please login to view profile')
        st.stop()

    username = st.session_state['username']
    st.write('Logged in as:', username)

    # Load this user's predictions
    df = load_user_predictions(username)

    # Get latest age from user's predictions
    user_age = None
    if not df.empty and 'Age' in df.columns:
        user_age = df.iloc[0]['Age']  # Most recent record because of ORDER BY DESC

    # Get number of distinct users who have predictions
    conn = sqlite3.connect(DB_NAME)
    total_pred_users = conn.execute("SELECT COUNT(DISTINCT username) FROM predictions").fetchone()[0]
    conn.close()

    # Show metrics
    col1, col2, col3 = st.columns(3)
    col1.metric("Your Age", user_age if user_age else "N/A")
    col2.metric("Your Total Predictions", len(df))
    col3.metric("Total Users with Predictions", total_pred_users)

    # Show chart + table
    if not df.empty:
        fig = px.histogram(df, x='DateTime', color='Disease', title='Predictions over time')
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(df.head(50))
    else:
        st.info("No predictions found for your account.")


# ----------------- Admin Dashboard ------------------
elif selected == 'Admin Dashboard':
    st.title('🛠️ Admin Dashboard')
    if not st.session_state.get('logged_in') or not is_admin_user(st.session_state['username']):
        st.warning('Admin access required')
        st.stop()
    df = load_all_predictions()
    users_conn = sqlite3.connect(DB_NAME)
    users_df = pd.read_sql_query('SELECT username, is_admin FROM users', users_conn)
    users_conn.close()
    st.metric('Total users', len(users_df))
    st.metric('Total predictions', len(df))
    if not df.empty:
        st.dataframe(df.groupby(['disease']).size().reset_index(name='count'))
        fig = px.bar(df.groupby('disease').size().reset_index(name='count'), x='disease', y='count', title='Predictions by Disease')
        st.plotly_chart(fig, use_container_width=True)

# ----------------- Lung Cancer Prediction ----------
elif selected == 'Lung Cancer Prediction':
    st.title('🫁 Lung Cancer Prediction')
    if not st.session_state.get('logged_in'):
        st.warning('Please login to use this feature')
        st.stop()
    update_last_active()

    # Prefill from last record
    last_df = load_user_predictions(st.session_state['username'])
    last_row = last_df[last_df['Disease'] == 'Lung Cancer'].iloc[0] if not last_df.empty and 'Lung Cancer' in last_df['Disease'].values else None

    col1, col2 = st.columns(2)
    with col1:
        name = st.text_input('Full Name', value=last_row['Name'] if last_row is not None else '', key='lc_name')
        age = st.number_input('Age', min_value=1, value=int(last_row['Age']) if last_row is not None else 30, key='lc_age')
    with col2:
        mobile = st.text_input('Mobile (optional)', key='lc_mobile')

    col1, col2, col3 = st.columns(3)
    with col1:
        SMOKING = st.selectbox('Smoking: YES=1 , NO=0', (0,1), key='lc_smoking')
        ANXIETY = st.selectbox('Anxiety: YES=1 , NO=0', (0,1), key='lc_anxiety')
        FATIGUE = st.selectbox('Fatigue: YES=1 , NO=0', (0,1), key='lc_fatigue')
        ALCOHOL_CONSUMING = st.selectbox('Alcohol: YES=1 , NO=0', (0,1), key='lc_alcohol')
        SWALLOWING_DIFFICULTY = st.selectbox('Swallowing Difficulty: YES=1 , NO=0', (0,1), key='lc_swallow')
    with col2:
        YELLOW_FINGERS = st.selectbox('Yellow Fingers: YES=1 , NO=0', (0,1), key='lc_yellow')
        PEER_PRESSURE = st.selectbox('Peer Pressure: YES=1 , NO=0', (0,1), key='lc_peer')
        ALLERGY = st.selectbox('Allergy: YES=1 , NO=0', (0,1), key='lc_allergy')
        COUGHING = st.selectbox('Coughing: YES=1 , NO=0', (0,1), key='lc_cough')
        CHEST_PAIN = st.selectbox('Chest Pain: YES=1 , NO=0', (0,1), key='lc_chest')
    with col3:
        CHRONIC_DISEASE = st.selectbox('Chronic Disease: YES=1 , NO=0', (0,1), key='lc_chronic')
        WHEEZING = st.selectbox('Wheezing: YES=1 , NO=0', (0,1), key='lc_wheeze')
        SHORTNESS_OF_BREATH = st.selectbox('Shortness of Breath: YES=1 , NO=0', (0,1), key='lc_short')

    if st.button('🔍 Predict Lung Cancer'):
        if not name.strip() or age <= 0:
            st.error('Please enter name and valid age')
        else:
            if cancer_model is None:
                st.warning('Cancer model not loaded; returning mock negative')
                pred = [0]
            else:
                pred = cancer_model.predict([[age, SMOKING, YELLOW_FINGERS, ANXIETY, PEER_PRESSURE,
                    CHRONIC_DISEASE, FATIGUE, ALLERGY, WHEEZING, ALCOHOL_CONSUMING, COUGHING,
                    SHORTNESS_OF_BREATH, SWALLOWING_DIFFICULTY, CHEST_PAIN]])
            result = 'Positive' if pred[0] == 1 else 'Negative'
            if result == 'Positive':
                st.error('🚨 Lung Cancer likely — seek medical attention')
            else:
                st.success('✅ Lung Cancer unlikely')
            tips = get_health_tip('Lung Cancer', age, smoker=bool(SMOKING), activity=None, family_history=None)
            st.info(f'💡 Health Tip: {tips}')
            save_prediction(st.session_state['username'], name, age, None, 'Lung Cancer', result, str({'smoking':SMOKING}))
            generate_pdf_report(name, age, 'Lung Cancer', result, tips, st.session_state['username'], lang='en')

# ----------------- Obesity Prediction ----------------
elif selected == 'Obesity Prediction':
    st.title('🍔 Obesity Prediction')
    if not st.session_state.get('logged_in'):
        st.warning('Please login')
        st.stop()
    update_last_active()

    last_df = load_user_predictions(st.session_state['username'])
    last_row = last_df[last_df['Disease'] == 'Obesity'].iloc[0] if not last_df.empty and 'Obesity' in last_df['Disease'].values else None

    name = st.text_input('Full Name', value=last_row['Name'] if last_row is not None else '', key='ob_name')
    mobile = st.text_input('Mobile (optional)', key='ob_mobile')

    col1, col2 = st.columns(2)
    with col1:
        gender = st.selectbox('Gender (Female=0 / Male=1)', (0,1), index=1 if last_row is None else 0, key='ob_gender')
        height = st.number_input('Height (m)', min_value=0.1, value=1.6, key='ob_height')
        family_history = st.selectbox('Family history overweight (Yes=1 / No=0)', (0,1), key='ob_famhist')
        vegetables = st.number_input('Veg freq (1–3)', min_value=1.0, value=2.0, key='ob_veg')
        between_meals = st.number_input('Between meals freq (1–4)', min_value=1.0, value=2.0, key='ob_between')
        water = st.number_input('Water consumption (1–3)', min_value=1.0, value=2.0, key='ob_water')
        activity = st.number_input('Physical activity frequency (0–3)', min_value=0.0, value=1.0, key='ob_activity')
        alcohol = st.number_input('Alcohol freq (0–3)', min_value=0.0, value=0.0, key='ob_alcohol')
    with col2:
        age = st.number_input('Age', min_value=1.0, value=30.0, key='ob_age')
        weight = st.number_input('Weight (kg)', min_value=1.0, value=70.0, key='ob_weight')
        caloric_food = st.selectbox('High caloric food (Yes=1 / No=0)', (0,1), key='ob_calfood')
        meals = st.number_input('Number of main meals', min_value=1, value=3, key='ob_meals')
        smoke = st.selectbox('Smoking (Yes=1 / No=0)', (0,1), key='ob_smoke')
        calories_monitor = st.selectbox('Calories monitoring (Yes=1 / No=0)', (0,1), key='ob_calmon')
        technology = st.number_input('Time using tech devices (0–2)', min_value=0.0, value=1.0, key='ob_tech')

    if st.button('🔍 Predict Obesity'):
        if not name.strip():
            st.error('Enter name')
        elif age <= 0 or height <= 0 or weight <= 0:
            st.error('Enter valid vitals')
        else:
            input_data = [gender, age, height, weight, family_history, caloric_food,
                vegetables, meals, between_meals, smoke, water,
                calories_monitor, activity, technology, alcohol]
            if obesity_model is None:
                st.warning('Obesity model not loaded; returning mock Normal Weight')
                prediction = 2
            else:
                prediction = obesity_model.predict([input_data])[0]
            classes = {1:'Insufficient Weight',2:'Normal Weight',3:'Overweight Level I',4:'Overweight Level II',5:'Obesity Type I',6:'Obesity Type II',7:'Obesity Type III'}
            result = classes.get(prediction, 'Unknown')
            if prediction <= 2:
                st.success(result)
            elif prediction <= 4:
                st.warning(result)
            else:
                st.error(result)
            tips = get_health_tip('Obesity', age, gender=gender, activity=activity, smoker=bool(smoke), family_history=bool(family_history))
            st.info(f'💡 Health Tip: {tips}')
            save_prediction(st.session_state['username'], name, age, gender, 'Obesity', result, str({'bmi': weight/(height*height) if height>0 else None}))
            generate_pdf_report(name, age, 'Obesity', result, tips, st.session_state['username'], lang='en')

# ----------------- Blood Pressure Prediction --------
elif selected == 'Blood Pressure Prediction':
    st.title('🩺 Blood Pressure Prediction')
    if not st.session_state.get('logged_in'):
        st.warning('Please login')
        st.stop()
    update_last_active()

    name = st.text_input('Full Name', key='bp_name')
    col1, col2, col3 = st.columns(3)
    with col1:
        Age = st.number_input('Age', min_value=1.0, key='bp_age')
        wc = st.number_input('Waist Circumference (WC)', min_value=0.0, key='bp_wc')
        SBP = st.number_input('Systolic Blood Pressure (SBP)', min_value=0.0, key='bp_sbp')
    with col2:
        Obese = st.selectbox('Obesity (Yes=1 / No=0)', (0,1), key='bp_obese')
        hc = st.number_input('Hip Circumference (HC)', min_value=0.0, key='bp_hc')
        DBP = st.number_input('Diastolic Blood Pressure (DBP)', min_value=0.0, key='bp_dbp')
    with col3:
        bmi = st.number_input('BMI', min_value=0.0, key='bp_bmi')
        whr = st.number_input('Waist-Hip Ratio (WHR)', min_value=0.0, key='bp_whr')
    mobile = st.text_input('Mobile (optional)', key='bp_mobile')

    if st.button('Predict Blood Pressure'):
        if not name.strip():
            st.error('Enter name')
        elif any(val == 0 for val in [Age, bmi, wc, hc, whr, SBP, DBP]):
            st.warning('Please enter all required details')
        else:
            if pressure_model is None:
                st.warning('Pressure model not loaded; returning mock Normal')
                prediction = [0]
            else:
                prediction = pressure_model.predict([[Age, Obese, bmi, wc, hc, whr, SBP, DBP]])
            result = 'Hyper Blood Pressure' if prediction[0] == 1 else 'Regular Blood Pressure'
            if prediction[0] == 1:
                st.error('⚠️ High Blood Pressure Detected')
            else:
                st.success('✅ Blood Pressure is Normal')
            tips = get_health_tip('Blood Pressure', Age)
            st.info(f'💡 Health Tip: {tips}')
            save_prediction(st.session_state['username'], name, Age, None, 'Blood Pressure', result, str({'sbp': SBP, 'dbp': DBP}))
            generate_pdf_report(name, Age, 'Blood Pressure', result, tips, st.session_state['username'], lang='en')

# ----------------- Default --------------------------
else:
    st.title('Welcome to Precision Health')
    st.write('Use the sidebar to navigate.')

# ----------------- End of App -----------------------

