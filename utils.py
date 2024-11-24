import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import layers
from tensorflow.keras.saving import register_keras_serializable
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
import ast

df = pd.read_csv('./dataset/table_interactions.csv')
db = pd.read_csv('./dataset/table_courses_info.csv')
db['embedding'] = db['embedding'].apply(lambda x: np.array(ast.literal_eval(x)))

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

model = tf.keras.models.load_model('./saved_models/v1/MF_model_v1.keras')
encoder = SentenceTransformer('all-MiniLM-L6-v2')

def new_user_update(new_user_id, preferences, encoder=encoder, threshold=0.5):
    interactions = df.pivot_table(
        index='user_id',
        columns='course_id',
        values='rating',
        fill_value=0
    )
    course_ids, _ = vector_search(encoder, preferences, threshold=threshold)

    n_items = len(df['course_encode'].unique())

    rating_new_user = np.zeros(n_items)
    indices = interactions.columns.get_indexer(course_ids)
    rating_new_user[indices] = 4

    interactions.loc[new_user_id] = rating_new_user
    new_df = interactions.reset_index().melt(
        id_vars='user_id',
        var_name='course_id',
        value_name='rating'
    )
    new_df = new_df[new_df['rating'] != 0].reset_index(drop=True)

    inv_user_map = df.groupby('user_id')['user_encode'].first().reset_index().set_index('user_encode').to_dict()[
        'user_id']
    user_map = {v: k for k, v in inv_user_map.items()}
    user_map[new_user_id] = max(user_map.values()) + 1

    inv_course_map = \
    df.groupby('course_id')['course_encode'].first().reset_index().set_index('course_encode').to_dict()['course_id']
    course_map = {v: k for k, v in inv_course_map.items()}
    course_map[new_user_id] = max(course_map.values()) + 1

    new_df['user_encode'] = new_df['user_id'].map(user_map)
    new_df['course_encode'] = new_df['course_id'].map(course_map)

    return new_df


def vector_search(encoder, skillset, k=10, threshold=None):
    vector_skillset = encoder.encode(skillset)
    vector_course = np.array(db['embedding'].tolist())

    tensor_skillset = tf.convert_to_tensor(vector_skillset, dtype=tf.float64)
    tensor_course = tf.convert_to_tensor(vector_course, dtype=tf.float64)
    tensor_skillset = tf.nn.l2_normalize(tensor_skillset, axis=-1)
    tensor_course = tf.nn.l2_normalize(tensor_course, axis=-1)

    cos_sim = tf.squeeze(tf.matmul(tensor_course, tensor_skillset, transpose_b=True))

    if threshold is not None:
        indices = tf.where(cos_sim >= threshold).numpy().flatten()
        top_idx = indices[tf.argsort(tf.gather(cos_sim, indices), direction='DESCENDING').numpy()]
    else:
        top_idx = tf.argsort(cos_sim, axis=0, direction='DESCENDING')[:k].numpy().flatten()

    rec_id = db.iloc[top_idx]['course_id'].tolist()
    rec_name = db.iloc[top_idx]['Course Name'].tolist()

    return rec_id, rec_name


def recommender(input_user, input_skillset, encoder=encoder, model=model, n=50, k=10):
    user_encode = df[df['user_id'] == input_user]['user_encode'].values[0]
    if user_encode > model.user_emb.input_dim - 1:
        interactions = df.pivot_table(
            index='user_id',
            columns='course_id',
            values='rating',
            fill_value=0
        )
        new_interaction = interactions.loc[input_user]
        exist_interaction = interactions.drop(input_user)

        similarity = np.matmul(exist_interaction.values, new_interaction.values)
        position = tf.argsort(similarity, direction='DESCENDING').numpy()[0]
        user_sim = exist_interaction.index[position]
        user = df[df['user_id'] == user_sim]['user_encode'].unique().item()
    else:
        user = df[df['user_id'] == input_user]['user_encode'].unique().item()

    course_ids, course_names = vector_search(encoder, input_skillset, k=n)
    course = df[df['course_id'].isin(course_ids)]['course_encode'].unique().tolist()

    pred = model.predict(user, course, k=k)
    pred_id = df[df['course_encode'].isin(pred)]['course_id']

    rec = db[db['course_id'].isin(pred_id)][['course_id', 'Course Name', 'Course URL']]

    response = {
        idx: {
            'course_id': row['course_id'],
            'course_name': row['Course Name'],
            'course_url': row['Course URL'],
        }
        for idx, row in rec.iterrows()
    }
    return response