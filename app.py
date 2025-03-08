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
places = [
    ("Jaipur", 26.9124, 75.7873),
    ("Amber", 26.9948, 75.8513),
    ("Bassi", 26.8394, 76.0411),
    ("Chaksu", 26.6052, 75.9481),
    ("Chomu", 27.1700, 75.7221),
    ("Mauzmabad", 26.5215, 75.4880),
    ("Jamwa Ramgarh", 26.9640, 75.8760),
    ("Phagi", 26.5873, 75.6671),
    ("Phulera", 26.8732, 75.2433),
    ("Kotputli", 27.7025, 76.1996),
    ("Sanganer", 26.8191, 75.8028),
    ("Shahpura", 27.3913, 75.9595),
    ("Viratnagar", 27.3189, 76.1833),
    ("Dudu", 26.7043, 75.5850),
    ("Sambhar", 26.9131, 75.1914),
    ("Kot Khawada", 26.7972, 75.6700),
    ("Kishangarh-Renwal", 26.7536, 75.5650),
    ("Paota", 27.3763, 76.0405),
    ("Bapu Nagar", 26.8926, 75.8120),
    ("Gopalpura", 26.8815, 75.7890),
    ("Ajmer Road", 26.9117, 75.7437),
    ("Vaishali Nagar", 26.9220, 75.7411),
    ("Malviya Nagar", 26.8526, 75.8180),
    ("Jagatpura", 26.8626, 75.8470),
    ("Tonk Road", 26.8537, 75.8050),
    ("Sindhi Camp", 26.9216, 75.7960),
    ("C-Scheme", 26.9124, 75.7873),
    ("Raja Park", 26.8996, 75.8277),
    ("Mansarovar", 26.8648, 75.7590)
]

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
    import streamlit as st
    import pandas as pd
    import numpy as np
    import random
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import OneHotEncoder

    # Load dataset
    df = pd.read_csv("crime_data2.csv")

    # Encode categorical data
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    encoded_features = encoder.fit_transform(df[["Crime_Type", "Weather_Condition", "Patrol_Presence"]])
    encoded_columns = encoder.get_feature_names_out(["Crime_Type", "Weather_Condition", "Patrol_Presence"])

    df_encoded = pd.DataFrame(encoded_features, columns=encoded_columns)
    df['Hour'] = df['Time'].apply(lambda x: int(x.split(":")[0]))
    df_final = pd.concat([df[["Hour", "Latitude", "Longitude", "Severity_Score"]], df_encoded], axis=1)

    # Train Model
    X = df_final.drop(columns=["Severity_Score"])
    y = df_final["Severity_Score"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Places List
    places = [
        ("Jaipur", 26.9124, 75.7873),
        ("Amber", 26.9948, 75.8513),
        ("Bassi", 26.8394, 76.0411),
        ("Chaksu", 26.6052, 75.9481),
        ("Chomu", 27.1700, 75.7221),
        ("Mansarovar", 26.8648, 75.7590),
        ("Jagatpura", 26.8626, 75.8470),
        ("Vaishali Nagar", 26.9220, 75.7411),
        ("Phagi", 26.5873, 75.6671)
    ]

    # Place-specific crime messages
    crime_messages = {
    "Mansarovar": ("More chances of theft in this area.", 1.4),
    "Jagatpura": ("Burglary may occur at night.", 1.3),
    "Vaishali Nagar": ("Moderate chances of property theft.", 1.2),
    "Chaksu": ("Rare cases of field theft reported.", 0.8),
    "Phagi": ("Lock your vehicles to avoid theft.", 0.9),
    "Amber": ("Be cautious of pickpockets in crowds.", 1.1),
    "Bassi": ("slight chances of mobile theft here.", 0.7),
    "Chomu": ("Avoid walking alone late at night.", 1.2),
    "Kotputli": ("Be careful while traveling at night.", 1.5),
    "Sanganer": ("be careful with your vehicles here.", 1.2)
}


    # Predict Function
    def predict_crime(input_time, input_weather, input_patrol, input_place):
        input_hour = int(input_time.split(":")[0])
        latitude, longitude = next((lat, lon) for place, lat, lon in places if place == input_place)

        input_data = pd.DataFrame([[input_hour, latitude, longitude]], columns=["Hour", "Latitude", "Longitude"])
        input_encoded = encoder.transform(pd.DataFrame([[random.choice(df['Crime_Type'].unique()), input_weather, input_patrol]],
                                                        columns=["Crime_Type", "Weather_Condition", "Patrol_Presence"]))
        input_encoded_df = pd.DataFrame(input_encoded, columns=encoded_columns)
        final_input = pd.concat([input_data, input_encoded_df], axis=1).reindex(columns=X.columns, fill_value=0)

        predicted_severity = model.predict(final_input)[0]

        # Apply Risk Logic
        if 18 <= input_hour < 24 or 0 <= input_hour < 6:
            if input_patrol == "No":
                predicted_severity *= 1.5
            else:
                predicted_severity *= 0.7
        elif 12 <= input_hour < 18:
            if input_patrol == "No":
                predicted_severity *= 1.2
            else:
                predicted_severity *= 0.9
        else:
            if input_patrol == "No":
                predicted_severity *= 1.1
            else:
                predicted_severity *= 0.8

        if input_weather == "Stormy":
            predicted_severity *= 0.6
        elif input_weather == "Rainy":
            predicted_severity *= 0.8
        elif input_weather == "Clear":
            predicted_severity *= 1.1

        # Adjust Severity Based on Place
        if input_place in crime_messages:
            message, multiplier = crime_messages[input_place]
            predicted_severity *= multiplier
        else:
            message = "No specific crime trend reported for this area."

        return max(predicted_severity, 1), message

    # User Inputs
    st.title("🚨 Crime Severity Prediction")
    input_time = st.time_input("Select Time")
    input_weather = st.selectbox("Weather Condition", ["Clear", "Rainy", "Cloudy", "Foggy", "Stormy"])
    input_patrol = st.selectbox("Patrol Presence", ["Yes", "No"])
    input_place = st.selectbox("Select Place", [place[0] for place in places])

    # Predict Button
    if st.button("Predict Severity"):
        severity, message = predict_crime(str(input_time), input_weather, input_patrol, input_place)
        st.write(f"### Predicted Crime Severity: {severity:.2f}")
        st.write(f"**{message}**")

        if severity < 4:
            st.markdown("🟢 **This area seems safe.**")
        elif 4 <= severity < 7:
            st.markdown("🟡 **Moderate Risk Area. Women advised to be careful . Be Cautious.**")
        else:
            st.markdown("🔴 **High-Risk Area.Unsafe for women after 12:00, Avoid If Possible!**")
    # Logout Button
    if st.button("Logout"):
        st.session_state.page = "login"
        st.session_state.logged_in = False
        st.rerun()
