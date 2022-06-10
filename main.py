import streamlit as st
import os
from PIL import Image
import numpy as np
import pickle
import tensorflow
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.applications.resnet50 import ResNet50,preprocess_input

from numpy.linalg import norm
from annoy import AnnoyIndex
from tensorflow.keras.applications.vgg16 import preprocess_input
feature_list = np.array(pickle.load(open('C:/Users/sagar/Downloads/fashion-recommender-system-main/fashion-recommender-system-main/embeddings.pkl','rb')))
#filenames = pickle.load(open('filenames.pkl','rb'))
feature_listhm = np.array(pickle.load(open('C:/Users/sagar/Downloads/fashion-recommender-system-main/fashion-recommender-system-main/embeddingshm.pkl','rb')))
filenameshm = pickle.load(open('filenameshm.pkl','rb'))

model = ResNet50(weights='imagenet',include_top=False,input_shape=(224,224,3))
model.trainable = False

model = tensorflow.keras.Sequential([
    model,
    GlobalMaxPooling2D()
])
st.title('Fashion Recommender System')

def save_uploaded_file(uploaded_file):
    try:
        with open(os.path.join('uploads',uploaded_file.name),'wb') as f:
            f.write(uploaded_file.getbuffer())
        return 1
    except:
        return 0

def feature_extraction(img_path,model):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    expanded_img_array = np.expand_dims(img_array, axis=0)
    preprocessed_img = preprocess_input(expanded_img_array)
    result = model.predict(preprocessed_img).flatten()
    normalized_result = result / norm(result)

    return normalized_result

def recommend(features):
    u = AnnoyIndex(f, 'angular')
    u.load('H&M.ann')
    indices = u.get_nns_by_vector(features, 6)
    return indices
f = len(feature_list[0])
uploaded_file = st.file_uploader("Choose an image")
if uploaded_file is not None:
    if save_uploaded_file(uploaded_file):
        # display the file
        display_image = Image.open(uploaded_file)
        st.image(display_image)
        # feature extract
        features = feature_extraction(os.path.join("uploads",uploaded_file.name),model)
        #st.text(features)
        # recommendention
        indices= recommend(features)
        # show
        col1,col2,col3,col4,col5 = st.columns(5)
        c1,c2,c3,c4,c5=st.columns(5)

        with col1:
            st.image(filenameshm[indices[0]])
        with col2:
            st.image(filenameshm[indices[1]])
        with col3:
            st.image(filenameshm[indices[2]])
        with col4:
            st.image(filenameshm[indices[3]])
        with col5:
            st.image(filenameshm[indices[4]])
    else:
        st.header("Some error occured in file upload")

