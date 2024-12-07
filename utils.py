import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import layers
from tensorflow.keras.saving import register_keras_serializable
import numpy as np
import pandas as pd
import tensorflow_hub as hub

df = pd.read_csv('./dataset/table_interactions.csv')
db = pd.read_csv('./dataset/table_courses_info.csv')

@register_keras_serializable(package='Custom')
class MF(tf.keras.Model):
    def __init__(self, num_users, num_items, emb_dim, init=True, bias=True, sigmoid=True, name="MF", **kwargs):
        super(MF, self).__init__(name=name, **kwargs)
        self.num_users = num_users
        self.num_items = num_items
        self.emb_dim = emb_dim
        self.init = init
        self.bias = bias
        self.sigmoid = sigmoid

        # Embedding layers
        self.user_emb = layers.Embedding(num_users, emb_dim)
        self.item_emb = layers.Embedding(num_items, emb_dim)

        if init:
            self.user_emb.embeddings_initializer = tf.keras.initializers.RandomUniform(0., 0.05)
            self.item_emb.embeddings_initializer = tf.keras.initializers.RandomUniform(0., 0.05)

        if bias:
            self.user_bias = self.add_weight(name="user_bias", shape=(num_users,), initializer="zeros", trainable=True)
            self.item_bias = self.add_weight(name="item_bias", shape=(num_items,), initializer="zeros", trainable=True)
            self.offset = self.add_weight(name="offset", shape=(), initializer="zeros", trainable=True)

    def call(self, inputs):
        user, item = inputs

        # Look up embeddings
        user_emb = self.user_emb(user)
        item_emb = self.item_emb(item)

        # Compute dot product
        element_product = tf.reduce_sum(user_emb * item_emb, axis=1)

        if self.bias:
            # Add biases
            user_b = tf.gather(self.user_bias, user)
            item_b = tf.gather(self.item_bias, item)
            element_product += user_b + item_b + self.offset

        if self.sigmoid:
            return self.sigmoid_range(element_product, low=0, high=5.5)

        return element_product

    def predict(self, user_id, course_id, k=10):
        tensor_user = tf.convert_to_tensor([user_id] * len(course_id), dtype=tf.int32)
        tensor_course = tf.convert_to_tensor(course_id, dtype=tf.int32)

        pred = self.call((tensor_user, tensor_course))
        rank = tf.argsort(pred, direction='DESCENDING')[:k].numpy().flatten()
        rec_id = tf.gather(tensor_course, rank)

        return rec_id.numpy().tolist()

    @staticmethod
    def sigmoid_range(x, low=0, high=5.5):
        return tf.sigmoid(x) * (high - low) + low

    def get_config(self):
        config = super(MF, self).get_config()
        config.update({
            "num_users": self.num_users,
            "num_items": self.num_items,
            "emb_dim": self.emb_dim,
            "init": self.init,
            "bias": self.bias,
            "sigmoid": self.sigmoid,
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

model = tf.keras.models.load_model('./saved_models/v1/MF_model.keras')
encoder = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")
desc_embedding = encoder(db['Description'])

def vector_search(skillset, k=10, threshold=None):
    tensor_skillset = encoder(skillset)
    tensor_course = desc_embedding
    tensor_skillset = tf.nn.l2_normalize(tensor_skillset, axis=-1)
    tensor_course = tf.nn.l2_normalize(tensor_course, axis=-1)

    cos_sim = tf.squeeze(tf.matmul(tensor_course, tensor_skillset, transpose_b=True))

    if threshold is not None:
        indices = tf.where(cos_sim >= threshold).numpy().flatten()
        top_idx = indices[tf.argsort(tf.gather(cos_sim, indices), direction='DESCENDING').numpy()]
    else:
        top_idx = tf.argsort(cos_sim, axis=0, direction='DESCENDING')[:k].numpy().flatten()

    rec_id = db.iloc[top_idx]['ID'].tolist()
    rec_name = db.iloc[top_idx]['Title'].tolist()

    return rec_id, rec_name


def recommender(user_id, skillset, n=50, k=5):
    if user_id > model.user_emb.input_dim - 1:
        user = 0
    else:
        user = user_id

    course_ids, course_names = vector_search(skillset, k=n)
    course = course_ids

    pred = model.predict(user, course, k=k)
    rec = db[db['ID'].isin(pred)]

    return rec['ID'].tolist()