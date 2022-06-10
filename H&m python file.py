import tensorflow
import os
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.applications.vgg16 import preprocess_input
import numpy as np
from tqdm import tqdm
from numpy.linalg import norm
import pickle
model=VGG16(weights='imagenet',include_top=False)
model = tensorflow.keras.Sequential([
    model,
    GlobalMaxPooling2D()
])
def extract_features(img_path,model):
    img = image.load_img(img_path,target_size=(224,224))
    img_array = image.img_to_array(img)
    expanded_img_array = np.expand_dims(img_array, axis=0)
    preprocessed_img = preprocess_input(expanded_img_array)
    result = model.predict(preprocessed_img).flatten()
    normalized_result = result / norm(result)

    return normalized_result
filenames=[]
for file in os.listdir('hmi'):
    filenames.append(os.path.join('hmi',file))

feature_list=[]
for file in tqdm(filenames):
    feature_list.append(extract_features(file,model))
pickle.dump(feature_list,open('embeddingshm.pkl','wb'))
pickle.dump(filenames,open('filenameshm.pkl','wb'))
print(len(feature_list[0]))

