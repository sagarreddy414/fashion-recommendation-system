from annoy import AnnoyIndex
import pickle
import tensorflow
import numpy as np
from numpy.linalg import norm
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.applications.resnet50 import ResNet50,preprocess_input
from tqdm import tqdm
from sklearn.neighbors import NearestNeighbors
import cv2
from tensorflow.keras.applications.vgg16 import VGG16
feature_list = np.array(pickle.load(open('embeddings.pkl','rb')))
filenames = pickle.load(open('filenames.pkl','rb'))
feature_listhm=np.array(pickle.load(open('embeddingshmresnet50.pkl','rb')))
filenames_hm=pickle.load(open('filenameshm.pkl','rb'))
model = ResNet50(weights='imagenet',include_top=False,input_shape=(224,224,3))
model.trainable = False

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

f = len(feature_list[0])

u = AnnoyIndex(f, 'angular')
t = AnnoyIndex(f, 'angular')
for i in tqdm(range(len(feature_list))):
    t.add_item(i,feature_list[i])
t.build(100)
t.save('myntra.ann')
z=AnnoyIndex(f,'angular')
for i in tqdm(range(len(feature_listhm))):
    z.add_item(i,feature_listhm[i])
z.build(100)
z.save('H&M.ann')
u.load('H&M.ann') # super fast, will just mmap the file

indices=u.get_nns_by_vector(normalized_result , 6)
# will find the 1000 nearest neighbors
print(indices)
for file in indices:
    if file < len(filenames):
        temp_img = cv2.imread(filenames_hm[file])
        cv2.imshow('output',cv2.resize(temp_img,(512,512)))
        cv2.waitKey(0)