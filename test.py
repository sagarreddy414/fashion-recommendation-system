import pickle
import tensorflow
import numpy as np
from numpy.linalg import norm
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPooling2D

from sklearn.neighbors import NearestNeighbors
import cv2
from tensorflow.keras.applications.vgg16 import VGG16

feature_list = np.array(pickle.load(open('embeddingshm.pkl','rb')))
filenames = pickle.load(open('filenameshm.pkl','rb'))
model=VGG16(weights='imagenet',include_top=False)
model = tensorflow.keras.Sequential([
    model,
    GlobalMaxPooling2D()
])

img = image.load_img('sample/s1.jpg',target_size=(224,224))
img_array = image.img_to_array(img)
expanded_img_array = np.expand_dims(img_array, axis=0)
preprocessed_img = preprocess_input(expanded_img_array)
result = model.predict(preprocessed_img).flatten()
normalized_result = result / norm(result)

neighbors = NearestNeighbors(n_neighbors=6,algorithm='brute',metric='euclidean')
neighbors.fit(feature_list)

distances,indices = neighbors.kneighbors([normalized_result])

print(indices)

#for file in indices[0][1:6]:
    #temp_img = cv2.imread(filenames[file])
    ##cv2.imshow('output',cv2.resize(temp_img,(512,512)))
    #cv2.waitKey(0)

