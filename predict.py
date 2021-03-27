import argparse

parser = argparse.ArgumentParser(description='TF AI APP')

parser.add_argument('--top_k', default=5,type=int)
parser.add_argument('--category_names', default='./label_map.json')
parser.add_argument('--path_to_image', default='./test_images/hard-leaved_pocket_orchid.jpg')
parser.add_argument('--model',default='zeizer_model.h5')

arguments=parser.parse_args()

topk=arguments.top_k
catnames=arguments.category_names
image_path=arguments.path_to_image
model=arguments.model





import warnings
warnings.filterwarnings('ignore')
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_hub as hub

import matplotlib.pyplot as plt
import numpy as np
#import seaborn as sns
import time as time
import json

import PIL 
from PIL import Image

#import seaborn as sns

#print('Using:')
#print('\t\u2022 TensorFlow version:', tf.__version__)
#print('\t\u2022 tf.keras version:', tf.keras.__version__)
#print('\t\u2022 Running on GPU' if tf.test.is_gpu_available() else '\t\u2022 GPU device not found. Running on CPU')


modelloaded = tf.keras.models.load_model(model,custom_objects={'KerasLayer':hub.KerasLayer}, compile=False)


with open(catnames, 'r') as f:
    class_names = json.load(f)

def process_image(image):
 #   im = Image.open(image)
#    im_tf=tf.image.resize(image,[224,224,3])
#    image=tf.image.convert_image_dtype(image, dtype=tf.float16, saturate=False)
    image=tf.convert_to_tensor(image,dtype=tf.float16)
    im_tf=tf.image.resize(image,[224,224])
    np_image=im_tf.numpy()
    np_image=np.array(np_image)/255
#    np_image=(np_image -np.array([0.485,0.456,0.406]))/np.array([0.229,0.224,0.225])
#    np_image=np_image.transpose((2, 0, 1))
    return np_image


def predict(image_path, model, topk=5):
    im = Image.open(image_path)
    test_image = np.asarray(im)
#    test_image = tf.cast(test_image, tf.float32)
    
    inputs=process_image(test_image)
#    inputs=tf.convert_to_tensor(inputs, dtype=tf.float32)
    results=model.predict(np.expand_dims(inputs,axis=0))
    results=results[0].tolist()
    val,ind=tf.math.top_k(results,k=topk,sorted=True)
#    sorted_vals=np.argsort(results[0])[::-1][:len(results)]
#    pred_class=class_names[str(sorted_vals[:k])]
    top_k_val=val.numpy().tolist()
    top_k_ind=ind.numpy().tolist()
    flowers=[class_names[str(i+1)] for i in top_k_ind]
#    top_k_flowers=np.argsort(top[0])[::-1][:len(results)]
    return top_k_val,flowers#,sorted_vals#[13]

def plot_testing(model,image_path):
# Setting up the plot
    plt.figure(figsize = (10,10))
    print('chega aqui')
    ax1,ax2 = plt.subplot(2,1)
    # Setting up the title
    # taking the third element from the splitting of the path
#    flower_num = image_path.split('/')[2]
    flower_num = image_path.split('/')[-1]
    # using the json from the beginning!
#    title_ = class_names[flower_num]
    title_ = 'Flower Classification'
    im = Image.open(image_path)
    test_image = np.asarray(im)
#processed_test_image = process_image(test_image)
    processed_test_image = process_image(test_image)
#    image = process_image(image_path)
#    imshow(processed_test_image, ax, title = title_);
    ax.imshow(processed_test_image)
    probs, inds = predict(image_path, model)
    plt.subplot(2,1,2)
#    flowers=[class_names[str(inds[i])] for i in range(len(inds))]
#    plt.barh(x=probs, y=flowers);
    plt.barh(x=probs, y=inds);
    plt.show()
    
#plot_testing(model,image_path)


### Matplotlib does not work here to do beautiful graphs! :(
top_k_val,flowers=predict(image_path, modelloaded, topk)
print('the top probabilities and the top indices are:')
print(top_k_val,flowers)