import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.applications.resnet50 import ResNet50,preprocess_input
import numpy as np
import PIL
from tqdm import tqdm
from numpy.linalg import norm
import os
import pickle

model = ResNet50(weights='imagenet',include_top=False,input_shape=(224,224,3))
# usually in transfer learning this is the standard size of the image
model.trainable=False
# in this case we are using the trainable function which is set to false means we dont want to train the model because it has already been trained on imagenet dataset
# freeze all layers except for last one which will be used to classify our images

model = tf.keras.Sequential(
    [
        model,
        GlobalMaxPooling2D()
    ]
)


def extract_features(img_path,model):
    img=image.load_img(img_path,target_size=(224,224)) # loading the image
    img_array=image.img_to_array(img)                  # converting the image into an array
    expanded_img= np.expand_dims(img_array,axis=0)     # adding the extra dimension to the image
    preprocessed_img=preprocess_input(expanded_img)    # preprocessing the expanded image
    result = model.predict(preprocessed_img).flatten()       # prediction got flattened means is converted to 1D array
    normalized_result = result/norm(result)            # normalizing the result

    return normalized_result

# now we are going to build the python list for the image folder where each image has its own file name

filenames=[]

for file in os.listdir('archive\myntradataset\images'):
    filenames.append(os.path.join('archive\myntradataset\images',file))

# so now we are calling extract_features function for each image file 
feature_list=[]
for file in tqdm(filenames):
    feature_list.append(extract_features(file,model))
# feature_list = np.array(feature_list.reshape(-1, feature_list.shape[-1]))
# print(np.array(feature_list).shape)
pickle.dump(feature_list,open('embeddings.pkl','wb'))
pickle.dump(filenames,open('filenames.pkl','wb'))

# the above line of code is to export the filenames and feature_list files 