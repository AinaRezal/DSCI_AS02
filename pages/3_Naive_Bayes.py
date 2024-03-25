import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
import pickle
from sklearn.naive_bayes import GaussianNB

st.set_page_config(layout="wide",
                   initial_sidebar_state="collapsed",
                   page_title='Naive Bayes'
                   )

st.markdown("<h1 style='text-align: center; color: black;'>Naive Bayes</h1>", unsafe_allow_html=True)

st.write('Write. This will be where the description of what will be predicted.')
st.text('Text. This will be where the description of what will be predicted.')
st.markdown("""---""")

# Pickle for prediction and mapping values
nb_pickle = open('nbm.pickle', 'rb')
map_pickle = open('outputnb.pickle', 'rb')

nbm = pickle.load(nb_pickle)
mapping = pickle.load(map_pickle)

nb_pickle.close()
map_pickle.close()

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
accuracy = accuracy_score(y_pred, y_test)
f1score = f1_score(y_test, y_pred, average='weighted')
st.write('Accuracy:', accuracy)
st.write('f1 score:', f1score)


with st.form('inputs'):
    # 'magnitude', 'mmi', 'tsunami', 'dmin', 'gap', 'depth', 'latitude', 'longitude'
    magnitude = st.slider('Magnitude', 0.0, 10.0, 5.0, step=0.1)
    mmi = st.number_input('Estimated Instrument Intensity', min_value=0)    
    tsunami = st.number_input('Occurence of Tsunami', min_value=0, max_value=1)
    distance = st.number_input('Horizontal Distance (km)', min_value=0)    
    gap = st.number_input('Gap of stations (degrees)', min_value=0, max_value=360)
    depth = st.number_input('Depth in kilometers', min_value=0)
    stations = st.number_input('Number of Seismic Stations', min_value=0)
    lang = st.number_input('Latitude', min_value=-90.00, max_value=90.00)
    long = st.number_input('Longitude', min_value=-180.00, max_value=180.00)
    submit = st.form_submit_button()

if submit :
    prediction = nbm.predict([[magnitude, mmi, tsunami, distance, gap, depth, stations, lang, long]])
    prediction_alert = mapping[prediction] # Takes the highest possibility of the prediction
    print(prediction)

    st.write(prediction_alert)
    # something is wrong with mapping
    # st.write(prediction_alert)