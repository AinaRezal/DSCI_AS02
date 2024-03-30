import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 
import pickle
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.inspection import permutation_importance
from modelling.LRmodel import r2, accuracy, f1score, precision

st.set_page_config(layout="wide",
                   initial_sidebar_state="collapsed",
                   page_title='Linear Regression'
                   )
st.markdown("<h1 style='text-align: center; color: black;'>Linear Regression</h1>", unsafe_allow_html=True)

# Description of Model
con1 = st.container(border=True)
col1 = con1.columns(2)

with col1[0]:
    st.subheader('Description of Columns')
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
    st.image('Graphs/LRfeatures_importance.png')

lr_pickle = open('pickle_output/linearregression.pickle', 'rb')
map_pickle = open('pickle_output/output_linearregression.pickle', 'rb')

lr_model = pickle.load(lr_pickle)
mapping = pickle.load(map_pickle)

lr_pickle.close()
map_pickle.close()


con2 = st.container(border=True)
reliabilty_info = pd.DataFrame(
    {'Evaluation Metrics' : ['Accuracy', 'Precision', 'F1 Score', 'R-squared'],
     'Description' : ['The percentage that correctly predicts an output out of all predictions.',
                     'Quality of correct predictions.',
                     'The percentage that measures accuracy by the total times it predicts correctly throughout the whole dataset.',
                     'The proportion of the variance in the dependent variable that is predictable from the independent variables.'],
     'Percentage/Score' : [accuracy, precision, f1score, r2]}
)   
con2.subheader("Linear Regression's Metric Evaluation Results")
con2.table(reliabilty_info)

st.divider()

with st.container():
    st.subheader('Prediction Features')
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
            distance = st.number_input('Horizontal Distance (km) From Surface of Epicentre To Nearest Station', min_value=0.0000)    
            gap = st.number_input('Largest Gap Between Two Azimuthally Stations (degrees)', min_value=0.0, max_value=360.0)
        with col2[1]:
            depth = st.number_input('Depth of Epicentre (km)', min_value=0.000)
            stations = st.number_input('Number of Seismic Stations', min_value=0)
            lat = st.number_input('Latitude', min_value=-90.0000, max_value=90.0000)
            long = st.number_input('Longitude', min_value=-180.0000, max_value=180.0000)
        submit = st.form_submit_button('Predict Alert Level', use_container_width=True)

st.write('Alert Level :')

if submit:
    features = [[magnitude, mmi, tsunami, distance, gap, depth, stations, lat, long]]
    
    prediction = lr_model.predict(features)
  
    prediction_alert = None
    if prediction.item() in mapping:  
        prediction_alert = mapping[prediction.item()]
    else:
       
        if prediction.item() < 0.5:
            prediction_alert = 'green'
        else:
            prediction_alert = 'unknown'
    
    if prediction_alert == 'unknown':
        st.success('UNKNOWN')
    elif prediction_alert == 'green':
        st.success('GREEN')
    elif prediction_alert == 'yellow':
        st.success('YELLOW')
    elif prediction_alert == 'orange':
        st.success('ORANGE')
    else:
        st.success('RED')
