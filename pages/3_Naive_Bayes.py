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
col = con1.columns(2)

with col[0]:
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

with col[1]:
    st.image('features_importance.png', caption='Importance of Each Feature For Prediction') # How to centralize

st.markdown("""---""")

# Pickle for prediction and mapping values
nb_pickle = open('nbm.pickle', 'rb')
map_pickle = open('outputnb.pickle', 'rb')

nbm = pickle.load(nb_pickle)
mapping = pickle.load(map_pickle)

nb_pickle.close()
map_pickle.close()


# Pre-model and modelling process
df = pd.read_csv('cleaned_earthquake.csv')
features = df[['magnitude', 'mmi', 'tsunami', 'dmin', 'gap', 'depth', 'nst', 'latitude', 'longitude']]
output = df['alert']
x_train, x_test, y_train, y_test = train_test_split(features,
                                                    output,
                                                    test_size= 0.2,
                                                    random_state=42)
nbmodel = GaussianNB()
nbmodel.fit(x_train, y_train)
y_pred = nbmodel.predict(x_test)

# To show reliability for users
con2 = st.container(border=True)
# accuracy = accuracy_score(y_pred, y_test)
# f1score = round(f1_score(y_test, y_pred, average='weighted'), 2)
# precision = round(precision_score(y_test, y_pred, average='weighted'), 2)
reliabilty_info = pd.DataFrame(
    {'Evaluation Metrics' : ['Accuracy', 'Precision', 'F1score'],
     'Description' : ['The percentage that correctly predicts an output out of all predictions.',
                     'Quality of correct predictions.',
                     'The percentage that measures accuracy by the total times it predicts correctly throughout the whole dataset.'],
     'Percentage' : [accuracy, precision, f1score]}
)
st.table(reliabilty_info)

# User inputs
with st.form('inputs'):
    # 'magnitude', 'mmi', 'tsunami', 'dmin', 'gap', 'depth', 'latitude', 'longitude'
    magnitude = st.slider('Magnitude of Earthquake', 0.0, 10.0, 5.0, step=0.1)
    mmi = st.number_input('Estimated Instrument Intensity during Earthquake', min_value=0)    
    tsunami = st.number_input('Occurence of Tsunami', min_value=0, max_value=1)
    distance = st.number_input('Horizontal Distance (km) From Surface of Epicentre To Nearest Station', min_value=0)    
    gap = st.number_input('Largest Gap Between Two Azimuthally Stations (degrees)', min_value=0, max_value=360)
    depth = st.number_input('Depth of Epicentre (km)', min_value=0)
    stations = st.number_input('Number of Seismic Stations', min_value=0)
    lat = st.number_input('Latitude', min_value=-90.00, max_value=90.00)
    long = st.number_input('Longitude', min_value=-180.00, max_value=180.00)
    submit = st.form_submit_button()

if submit :
    prediction = nbm.predict([[magnitude, mmi, tsunami, distance, gap, depth, stations, lat, long]])
    prediction_alert = mapping[prediction][0] # Takes the highest possibility of the prediction
    st.write(prediction_alert)