from mimetypes import init
from keras.models import model_from_json
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input as mobilenet_v2_preprocess_input
from tensorflow.keras.preprocessing import image
import tensorflow as tf
import streamlit as st
import numpy as np
import cv2
import streamlit.components.v1 as components
import time

# page configuration
st.set_page_config(
    page_title="minor project",
    page_icon="",
    layout='centered',
    initial_sidebar_state="collapsed",
)
st.title("Eye Disease Prediction Using ML")


# progress bar
st.markdown(
    """
    <style>
        .stProgress > div > div > div > div {
            background-color: #CE4980;
        }
    </style>""",
    unsafe_allow_html=True,
)
progress = st.progress(0)
for i in range(100):
    time.sleep(0.01)
    progress.progress(i+1)

# video_file
video_file = open('eye.mp4', 'rb')
video_bytes = video_file.read()
st.video(video_bytes, start_time=1)
st.header("Test Your Eyes Today !!")


# model = tf.keras.models.load_model("saved_model/mdl_wts.hdf5")
json_file = open("saved_model/model-101.json", 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
# load weights into new model
model.load_weights("saved_model/model.h5")

# load file
uploaded_file = st.file_uploader(
    "Upload Your Eye Image Here ", type="jpg")

map_dict = {0: 'Bulging',
            1: 'Cataract',
            2: 'Crossed',
            3: 'Glaucoma ',
            4: 'Uveitis'
            }

if uploaded_file is not None:
    # Convert the file to an opencv image.
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    opencv_image = cv2.imdecode(file_bytes, 1)
    opencv_image = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(opencv_image, (224, 224))

    # Now do something with the image! For example, let's display it:

    col1, col2 = st.beta_columns(2)
    col1.subheader("Original Image ")
    col1.image(resized, use_column_width=True)

    col2.subheader("Grayscale Image")
    grayscale_image = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2GRAY)
    resized_gray = cv2.resize(grayscale_image, (224, 224))
    col2.image(resized_gray, use_column_width=True)

    img_reshape = resized[np.newaxis, ...]
# code
    code_copy = ('''import cv2
import numpy as np
import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input as mobilenet_v2_preprocess_input

st.header("Deep Learning Project")
model = tf.keras.models.load_model("saved_model/mdl_wts.hdf5")
# load file
uploaded_file = st.file_uploader("  ", type="jpg")

map_dict = {0: 'Bulging',
            1: 'Cataract',
            2: 'Crossed',
            3: 'Glaucoma ',
            4: 'Normal ',
            5: 'Uveitis'
            }


if uploaded_file is not None:
    # Convert the file to an opencv image.
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    opencv_image = cv2.imdecode(file_bytes, 1)
    opencv_image = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(opencv_image, (224, 224))

    # Now do something with the image! For example, let's display it:
    st.image(resized, caption=None, width=None, use_column_width=None,
             clamp=False, channels='RGB', output_format='auto')
    img_reshape = resized[np.newaxis, ...]

    Genrate_pred = st.button("Generate Prediction")
    if Genrate_pred:
        prediction = model.predict(img_reshape).argmax()
        print(prediction)

        st.title("Predicted Label for the image is {}".format(
            map_dict[prediction]))
        print(map_dict[prediction])

        ''')
    Genrate_pred = st.button("Generate Prediction")
    if Genrate_pred:
        prediction = model.predict(img_reshape).argmax()
        print(prediction)

        st.title("Predicted Label for the image is {}".format(
            map_dict[prediction]))
        print(map_dict[prediction])


# content in sidebar.


def draw_all(
    key,
    plot=False,
):
    st.write(
        """
        # Project Description
       
        Minor Project On ML 
        
        """
    )
    st.write("#")
    st.header("Contributed by")
    st.write(
        """
        Pavankalyan D S  - 1RV18EC108\n
        Rakesh Reddy P -  1RV18EC109\n
         Prajwal  B  Raj -  1RV18EC111 \n
        Pratheek J Bhat -  1RV18EC105\n
        
        """
    )
    st.write("#")
    st.header("Guided by")
    st.write(
        """
        Rajani Katiyar 
        """
    )


with st.sidebar:
    draw_all("sidebar")
