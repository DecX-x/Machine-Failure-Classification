import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf

# Load the model
model = tf.keras.models.load_model('model.h5')

# Function to predict the failure
def predict_failure(data):
    data = np.array(data).reshape(1, -1)
    prediction = model.predict(data)
    return prediction

# Title
st.title('Predicting Machine Failure')


# aq voc
# Sidebar
st.write('\nInput Parameters')

data = [0, 0, 0]


# Slider for the input parameters
data_Aq = st.slider('Air Quality', 0, 10, 5)

data_voc = st.slider('Volatile organic compounds level', 0, 10, 5)

data_temp = st.slider("Temperature", 0, 30, 15)

#convert to numpy array
data = [data_Aq, data_voc, data_temp]

# plot the data
st.write('\n\nData Distribution')
data1 = pd.DataFrame(data, index=['Air Quality', 'VOC', 'Temperature'], columns=['Value'])
st.bar_chart(data1)


# Predict apakah mesin akan rusak atau tidak
if st.button('Predict'):
    prediction = predict_failure(data)
    if prediction[0][0] > 0.5:
        st.write('The machine will fail')
    else:
        st.write('The machine will not fail')
    # Show the prediction with percentage
    st.write(f'Failure Percantage: {prediction[0][0] * 100:.2f}%')


















