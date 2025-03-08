import streamlit as st
import pandas as pd
import folium
from folium.plugins import HeatMap
from streamlit_folium import folium_static
import random
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import bcrypt
import json

# JSON File Path for local storage
json_file = 'users.json'

# Page Configuration
st.set_page_config(page_title="Crime Prediction Dashboard", page_icon="🕵️‍♀️")

# Session State Initialization
if 'page' not in st.session_state:
    st.session_state['page'] = "login"

# Function to hash password
def hash_password(password):
    return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')

# Function to check password
def check_password(password, hashed):
    return bcrypt.checkpw(password.encode('utf-8'), hashed.encode('utf-8'))

# Function to load JSON data
def load_users():
    try:
        with open(json_file, 'r') as file:
            return json.load(file)
    except FileNotFoundError:
        return []

# Function to save JSON data
def save_users(users):
    with open(json_file, 'w') as file:
        json.dump(users, file, indent=4)

# Function to register user
def register_user(username, password, role):
    users = load_users()
    for user in users:
        if user['username'] == username:
            st.error("Username already exists")
            return False
    hashed_password = hash_password(password)
    users.append({'username': username, 'password': hashed_password, 'role': role})
    save_users(users)
    st.success("User registered successfully. Please login.")
    st.session_state.page = "login"
    st.rerun()

# Function to authenticate user
def authenticate_user(username, password):
    users = load_users()
    for user in users:
        if user['username'] == username and check_password(password, user['password']):
            st.session_state['user_role'] = user['role']
            st.session_state['page'] = user['role']
            st.session_state['logged_in'] = True
            st.rerun()
    st.error("Invalid username or password")

# ✅ LOGIN PAGE HANDLING
if st.session_state.page == "login":
    st.title("Login Page")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        authenticate_user(username, password)

    if st.button("New User? Register Here"):
        st.session_state.page = "register"
        st.rerun()

# ✅ REGISTER PAGE HANDLING
elif st.session_state.page == "register":
    st.title("Register")
    new_username = st.text_input("New Username")
    new_password = st.text_input("New Password", type="password")
    role = st.selectbox("Select Role", ["government", "user"])

    if st.button("Register"):
        if new_username and new_password:
            register_user(new_username, new_password, role)
        else:
            st.error("All fields are required")

    if st.button("Back to Login"):
        st.session_state.page = "login"
        st.rerun()

# ✅ GOVERNMENT DASHBOARD
elif st.session_state.page == "government":
    st.title('Government Crime Management Dashboard')
    df = pd.read_csv("crime_data2.csv")

    # Encode categorical variables
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    encoded_features = encoder.fit_transform(df[["Crime_Type", "Weather_Condition", "Patrol_Presence"]])
    encoded_columns = encoder.get_feature_names_out(["Crime_Type", "Weather_Condition", "Patrol_Presence"])

    df_encoded = pd.DataFrame(encoded_features, columns=encoded_columns)
    df['Hour'] = df['Time'].apply(lambda x: int(x.split(":")[0]))
    df_final = pd.concat([df[["Hour", "Latitude", "Longitude", "Severity_Score"]], df_encoded], axis=1)

    X = df_final.drop(columns=["Severity_Score"])
    y = df_final["Severity_Score"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    def predict_crime(input_time, input_weather, input_patrol):
        input_hour = int(input_time.split(":")[0])
        input_data = pd.DataFrame([[input_hour, 26.9124, 75.7873]], columns=["Hour", "Latitude", "Longitude"])

        input_encoded = encoder.transform(pd.DataFrame([[random.choice(df['Crime_Type'].unique()), input_weather, input_patrol]],
                                                        columns=["Crime_Type", "Weather_Condition", "Patrol_Presence"]))
        input_encoded_df = pd.DataFrame(input_encoded, columns=encoded_columns)
        final_input = pd.concat([input_data, input_encoded_df], axis=1).reindex(columns=X_train.columns, fill_value=0)

        predicted_severity = model.predict(final_input)[0]

        return max(predicted_severity, 1)

    # User Inputs
    input_time = st.time_input("Select Time")
    input_weather = st.selectbox("Weather Condition", ["Clear", "Rainy", "Cloudy", "Foggy", "Stormy"])
    input_patrol = st.selectbox("Patrol Presence", ["Yes", "No"])

    if st.button("Predict Severity"):
        severity = predict_crime(str(input_time), input_weather, input_patrol)
        st.write(f"### Predicted Crime Severity: {severity:.2f}")

        # Generate Heatmap
        df['Predicted_Severity'] = df.apply(lambda row: predict_crime(row['Time'], row['Weather_Condition'], row['Patrol_Presence']), axis=1)
        crime_map = folium.Map(location=[df['Latitude'].mean(), df['Longitude'].mean()], zoom_start=10)
        HeatMap(df[['Latitude', 'Longitude', 'Predicted_Severity']].values.tolist(), radius=15, blur=10, max_zoom=1).add_to(crime_map)

        # Embed the map into Streamlit
        folium_static(crime_map)

        # Provide download option
        st.download_button("Download Heatmap CSV", df.to_csv(index=False).encode('utf-8'), "heatmap_data.csv")


# ✅ USER DASHBOARD
elif st.session_state.page == "user":
    st.title('User Crime Dashboard')
    df = pd.read_csv("crime_data2.csv")
    crime_map = folium.Map(location=[df['Latitude'].mean(), df['Longitude'].mean()], zoom_start=12)
    HeatMap(df[['Latitude', 'Longitude', 'Severity_Score']].values.tolist(), radius=15, blur=10, max_zoom=1).add_to(crime_map)
    folium_static(crime_map)

    # Logout Button
    if st.button("Logout"):
        st.session_state.page = "login"
        st.session_state.logged_in = False
        st.rerun()
