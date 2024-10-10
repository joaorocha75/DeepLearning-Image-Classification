import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import load_model
import streamlit as st
import numpy as np

# Carregar o modelo salvo
model = load_model('/Users/joaorocha/Desktop/Image_Classification_Model/Image_classify.keras')

# Lista de categorias
data_cat = [
 'apple', 'banana', 'beetroot', 'bell pepper', 'cabbage', 
 'capsicum', 'carrot', 'cauliflower', 'chilli pepper', 'corn', 
 'cucumber', 'eggplant', 'garlic', 'ginger', 'grapes', 'jalepeno', 
 'kiwi', 'lemon', 'lettuce', 'mango', 'onion', 'orange', 'paprika', 
 'pear', 'peas', 'sweetcorn', 'sweetpotato', 'tomato', 'turnip', 
 'watermelon'
]

# Definir dimensões da imagem
img_height = 180
img_width = 180

# Caminho para a imagem
image_path = '/Users/joaorocha/Desktop/Image_Classification_Model/Apple.jpg'

# Carregar e processar a imagem
image_load = tf.keras.utils.load_img(image_path, target_size=(img_height, img_width))
img_arr = tf.keras.utils.img_to_array(image_load)  # Corrigido para usar 'image_load'
img_bat = tf.expand_dims(img_arr, 0)  # Adicionar batch dimension

# Fazer predição
predict = model.predict(img_bat)

# Aplicar softmax nas predições para converter em probabilidades
score = tf.nn.softmax(predict[0])

# Mostrar a imagem no Streamlit
st.image(image_path)

# Mostrar a predição e a acurácia
st.write('Veg/fruit in image is {} with accuracy of {:0.2f}%'.format(data_cat[np.argmax(score)], np.max(score)*100))
