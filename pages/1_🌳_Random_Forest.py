import streamlit as st
import pandas as pd
import pickle

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score
from sklearn.model_selection import train_test_split


st.set_page_config(layout="wide",
                   initial_sidebar_state="collapsed"
                   )

@st.cache_data
def load_model():
    with open('pickle_output/randomforest_confusionmatrix.pickle', 'rb') as file:
        model = pickle.load(file)
    return model

def predict(model, input_data):
    predictions = model.predict(input_data)
    return predictions

st.markdown("<h2 style='text-align: center;'>Random Forest Classifier</h2>", unsafe_allow_html=True)
st.write("<p style='text-align: center;'>This page predicts the alert level (green, yellow, orange, red) of earthquakes.</p>", unsafe_allow_html=True)

df_eq = pd.read_csv('cleaned_earthquake.csv')

st.markdown("""---""")

col1, col2 = st.columns([1, 3])

with col2:
    # dataset for reference
    st.write("For references:")
    st.write(df_eq)

with col1:
    # choose input type menu
    st.write("\n\n")
    st.write("\n\n")
    st.write("\n\n")
    st.write("\n\n")
    st.write("\n\n")
    st.write("\n\n")
    st.write("\n\n")
    st.write("\n\n")
    st.write("\n\n")

    st.write("<h4>Select input type</h4>", unsafe_allow_html=True)
    input_type = st.selectbox("", ["Sample from dataset", "Enter values manually", "Use sliders"], key="input_type")

st.markdown("""---""")
# select a sample option
if input_type == "Sample from dataset":
    sample_index = st.selectbox('Select an index from the dataset:', df_eq.index)
    input_data = df_eq.loc[sample_index, ['cdi', 'mmi', 'tsunami', 'sig', 'nst', 'dmin', 'gap', 'depth', 'latitude', 'longitude']]

# enter values option
elif input_type == "Enter values manually":
    cdi_value = st.text_input("Enter value for cdi:")
    mmi_value = st.text_input("Enter value for mmi:")
    tsunami_value = st.text_input("Enter value for tsunami:")
    sig_value = st.text_input("Enter value for sig:")
    nst_value = st.text_input("Enter value for nst:")
    dmin_value = st.text_input("Enter value for dmin:")
    gap_value = st.text_input("Enter value for gap:")
    depth_value = st.text_input("Enter value for depth:")
    latitude_value = st.text_input("Enter value for latitude:")
    longitude_value = st.text_input("Enter value for longitude:")

    input_data = {
        'cdi': cdi_value,
        'mmi': mmi_value,
        'tsunami': tsunami_value,
        'sig': sig_value,
        'nst': nst_value,
        'dmin': dmin_value,
        'gap': gap_value,
        'depth': depth_value,
        'latitude': latitude_value,
        'longitude': longitude_value
    }
    input_data = pd.DataFrame([input_data])

# slider option
elif input_type == "Use sliders":
    cdi_value = st.slider("Select value for cdi:", min_value=0.0, max_value=9.0, step=0.1)
    mmi_value = st.slider("Select value for mmi:", min_value=0.0, max_value=9.0, step=0.1)
    tsunami_value = st.slider("Select value for tsunami:", min_value=0.0, max_value=1.0, step=0.1)
    sig_value = st.slider("Select value for sig:", min_value=600, max_value=3000, step=1)
    nst_value = st.slider("Select value for nst:", min_value=0, max_value=900, step=1)
    dmin_value = st.slider("Select value for dmin:", min_value=0.0, max_value=11.000, step=0.0001)
    gap_value = st.slider("Select value for gap:", min_value=0, max_value=300, step=1)
    depth_value = st.slider("Select value for depth:", min_value=0, max_value=900, step=1)
    latitude_value = st.slider("Select value for latitude:", min_value=-100.0, max_value=100.0, step=0.001)
    longitude_value = st.slider("Select value for longitude:", min_value=-100.0, max_value=100.0, step=0.001)

    input_data = {
        'cdi': cdi_value,
        'mmi': mmi_value,
        'tsunami': tsunami_value,
        'sig': sig_value,
        'nst': nst_value,
        'dmin': dmin_value,
        'gap': gap_value,
        'depth': depth_value,
        'latitude': latitude_value,
        'longitude': longitude_value
    }
    input_data = pd.DataFrame([input_data])

st.write("\n\n")

if st.button("Predict"):
    if input_data is not None:
        model = load_model()

        predictions = predict(model, input_data.values.reshape(1, -1))
        predicted_alert_level = predictions[0].upper()

        # RFC modeling
        df = pd.read_csv('cleaned_earthquake.csv')

        x = df[['cdi', 'mmi', 'tsunami', 'sig', 'nst', 'dmin', 'gap', 'depth', 'latitude', 'longitude']]
        y = df['alert']

        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

        rf = RandomForestClassifier()
        rf.fit(x_train, y_train)

        y_pred = rf.predict(x_test)


        st.markdown("""---""")

        col3, col4 = st.columns([3, 7])

        with col3:
            # model evaluation
            st.markdown("<h4>Model Evaluation:</h4>", unsafe_allow_html=True)
            st.write("\n\n")

            accuracy = accuracy_score(y_test, y_pred)
            st.write("Accuracy:", accuracy)

            precision = precision_score(y_test, y_pred, average='macro')
            st.write("Precision:", precision)

            recall = recall_score(y_test, y_pred, average='macro')
            st.write("Recall:", recall)

        with col4:
            # feature importance
            st.write("<h4>Feature Importance:</h4>", unsafe_allow_html=True)
            st.write("\n\n")

            feature_importances = pd.Series(model.feature_importances_, index=x_train.columns).sort_values(ascending=False)
            st.bar_chart(feature_importances, color='#99ccff')

        st.markdown("""---""")
        # predicted output
        color_map = {'green': '#65a765', 'yellow': '#F4EA56', 'orange': '#F28C28', 'red': '#E23F44'}
        alert_color = color_map.get(predictions[0].lower(), '#454545')
        font_color = 'black' if alert_color == '#F4EA56' else 'white'
        highlighted_text = f'<div style=" display: flex; align-items: center; justify-content: center; background-color: {alert_color}; color: {font_color}; padding: 10px; border-radius: 5px; width: 100%;">{predicted_alert_level}</div>'
        
        st.write("<p style='text-align: center;'><b>Alert Level:<b></p>", unsafe_allow_html=True)
        st.markdown(f'<p style="font-size: 24px;">{highlighted_text}</p>', unsafe_allow_html=True)