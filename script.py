import numpy as np
import tensorflow as tf
import pandas as pd
from collections import defaultdict
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv('./ML-Development/dataset/dataset_user_item_rating_v2.csv')
db = pd.read_csv('./ML-Development/dataset/db_courses_embedding.csv')
dd = defaultdict(LabelEncoder)
cols_cat = ['user_id', 'course_id']
for c in cols_cat:
    dd[c].fit(df[c].unique())
    df[c+'_encoded'] = dd[c].transform(df[c])

def vector_search(encoder, skillset, k=10):
    vector_skillset = encoder.encode(skillset)
    vector_course = np.array(db['embedding'].tolist())

    tensor_skillset = tf.convert_to_tensor(vector_skillset, dtype=tf.float64)
    tensor_course = tf.convert_to_tensor(vector_course, dtype=tf.float64)

    dot_product = tf.matmul(tensor_course, tensor_skillset, transpose_b=True)
    cosine_similarity = dot_product / (tf.norm(tensor_course, axis = 1, keepdims=True) * tf.transpose(tf.norm(tensor_skillset, axis=1, keepdims=True)))
    top_k_idx = tf.argsort(cosine_similarity, axis=0, direction='DESCENDING')[:k].numpy().flatten()

    rec_id = db.iloc[top_k_idx]['course_id'].tolist()
    rec_name = db.iloc[top_k_idx]['Course Name'].tolist()

    return rec_id, rec_name

def recommender(user_id, skillset, encoder, model, dd, n=50, k=10):
    course_ids, course_names = vector_search(encoder, skillset, k=n)
    user = dd['user_id'].transform([user_id])[0] # Harus diubah ke pengambilan data dari DB User
    course = dd['course_id'].transform(course_ids)

    pred = model.predict(user, course, k=k)
    pred_id = dd['course_id'].inverse_transform(pred)

    rec = db[db['course_id'].isin(pred_id)][['course_id', 'Course Name', 'Course URL']]

    return rec.to_dict()