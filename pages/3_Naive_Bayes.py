# Importing necessary libraries
import streamlit as st
import pandas as pd
import pickle
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from nb import accuracy, precision, f1score

# Setting page configuration
st.set_page_config(layout="wide",
                   initial_sidebar_state="collapsed",
                   page_title='Naive Bayes'
                   )
st.markdown("<h1 style='text-align: center; color: black;'>Naive Bayes</h1>", unsafe_allow_html=True)

# Description of Model
con1 = st.container(border=True)
col1 = con1.columns(2)

with col1[0]:
    table_info = pd.DataFrame(
        {"Column's Name" : ['magnitude', 'mmi', 'tsunami', 'dmin', 'gap', 'depth', 'nst', 'latitude', 'longitude'],
        'Description' : ['Size of earthquake.',
                        'Estimated instrument intensity during the event of earthquake.',
                        'Oceanic event.',
                        'Horizontal distance from the surface of the epicentre to the nearest station.',
                        'Largest azimuthally gap between two azimuthally adjacent stations.',
                        'The location of the ruptures (epicentre).',
                        'Number of seismic stations.',
                        'Coordinates',
                        'Coordinates.']}
    )
    st.table(table_info)

with col1[1]:
    st.image('features_importance.png', caption='Importance of Each Feature For Prediction') # How to centralize

# Pickle for prediction and mapping values
nb_pickle = open('nbm.pickle', 'rb')
map_pickle = open('outputnb.pickle', 'rb')

nbm = pickle.load(nb_pickle)
mapping = pickle.load(map_pickle)

nb_pickle.close()
map_pickle.close()

# To show reliability for users
con2 = st.container(border=True)
reliabilty_info = pd.DataFrame(
    {'Evaluation Metrics' : ['Accuracy', 'Precision', 'F1score'],
     'Description' : ['The percentage that correctly predicts an output out of all predictions.',
                     'Quality of correct predictions.',
                     'The percentage that measures accuracy by the total times it predicts correctly throughout the whole dataset.'],
     'Percentage' : [accuracy, precision, f1score]}
)
st.table(reliabilty_info)

st.markdown("""---""")

# User inputs
with st.form('inputs'):
    col2 = st.columns(2)
    magnitude = st.slider('Magnitude of Earthquake', 0.0, 10.0, 5.0, step=0.1)
    with col2[0]:
        tsunami = st.radio('Occurence of Tsunami', ['Yes', 'No'])
        if tsunami == 'Yes':
            tsunami = 1
        else:
            tsunami = 0
        mmi = st.number_input('Estimated Instrument Intensity during Earthquake', min_value=0)
        distance = st.number_input('Horizontal Distance (km) From Surface of Epicentre To Nearest Station', min_value=0)    
        gap = st.number_input('Largest Gap Between Two Azimuthally Stations (degrees)', min_value=0, max_value=360)
    with col2[1]:
        depth = st.number_input('Depth of Epicentre (km)', min_value=0)
        stations = st.number_input('Number of Seismic Stations', min_value=0)
        lat = st.number_input('Latitude', min_value=-90.00, max_value=90.00)
        long = st.number_input('Longitude', min_value=-180.00, max_value=180.00)
    submit = st.form_submit_button(use_container_width=True)

st.write('Alert Level :')

if submit :
    prediction = nbm.predict([[magnitude, mmi, tsunami, distance, gap, depth, stations, lat, long]])
    prediction_alert = mapping[prediction][0] # Takes the highest possibility of the prediction
    
    if prediction_alert == 'unknown':
        st.write('UNKNOWN')
    elif prediction_alert == 'green':
        st.write(':green[GREEN]')
    elif prediction_alert == 'yellow':
        st.write(':yellow[YELLOW]')
    elif prediction_alert == 'orange':
        st.write(':orange[ORANGE]')
    else:
        st.write(':red[RED]')