import pandas as pd
import numpy as np
import tensorflow as tf
from sentence_transformers import SentenceTransformer
import ast
from utils import *


df = pd.read_csv('./dataset/table_interactions.csv')
db = pd.read_csv('./dataset/table_courses_info.csv')
db['embedding'] = db['embedding'].apply(lambda x: np.array(ast.literal_eval(x)))

model = tf.keras.models.load_model('./model/MF_model_v1.keras.')
encoder = SentenceTransformer('all-MiniLM-L6-v2')

input_user = input('input apa: ')
# lanjutt..