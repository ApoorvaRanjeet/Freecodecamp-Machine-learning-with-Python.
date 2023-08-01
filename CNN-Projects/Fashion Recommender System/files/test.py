import pickle
import numpy as np
import PIL
from tensorflow.keras.preprocessing import image
import tensorflow as tf
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.applications.resnet50 import ResNet50,preprocess_input
from numpy.linalg import norm
import sklearn
from sklearn.neighbors import NearestNeighbors
import cv2 as cv

feature_list = np.array(pickle.load(open('embeddings.pkl','rb'))) # rb means read binary mode



filenames = pickle.load(open('filenames.pkl','rb'))

print(feature_list.shape)

model = ResNet50(weights='imagenet',include_top=False,input_shape=(224,224,3))

# usually in transfer learning this is the standard size of the image

model.trainable=False

# in this case we are using the trainable function which is set to false means we dont want to train the model becaise it has already been trained on imagenet dataset
# freeze all layers except for last one which will be used to classify our images

model = tf.keras.Sequential(
    [
        model,
        GlobalMaxPooling2D()
    ]
)
print(model.summary())

img=image.load_img('test\shirt.jpg',target_size=(224,224)) # loading the image
img_array=image.img_to_array(img)                  # converting the image into an array
expanded_img= np.expand_dims(img_array,axis=0)     # adding the extra dimension to the image
preprocessed_img=preprocess_input(expanded_img)    # preprocessing the expanded image
result = model.predict(preprocessed_img).flatten()        # prediction got flattened means is converted to 1D array
normalized_result = result/norm(result)            # normalizing the result

# now we are finding the distance between the normalized_result of the test image and the feature_list
# using the nearest neigbor algorithm

neighbors = NearestNeighbors(n_neighbors=6,algorithm='brute',metric='euclidean')
neighbors.fit(feature_list)

# we have to find the 5 nearest neigbors of normalized result from feature_list
distances , indices = neighbors.kneighbors([normalized_result])

print(indices)

for file in indices[0][1:6]:
    temp_img = cv.imread(filenames[file])
    cv.imshow('output_image',cv.resize(temp_img,(512,512)))
    cv.waitKey(0)