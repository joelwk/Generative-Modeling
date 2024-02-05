import tensorflow as tf
from tensorflow.keras.layers import Layer, Dense, Dropout, LayerNormalization, Embedding, Input
from tensorflow.keras.models import Model
from generative_text.general_tnn.utils.fnProcessing import read_config, pad_punctuation, normalize_text, remove_whitespace

config_params = read_config(section='params', config_path='./generative_text/configpaired.ini')
params = {key: config_params[key] for key in config_params}

max_len = int(params['max_len'])
vocab_size = int(params['vocab_size'])
num_heads = int(params['n_heads'])
num_layers = int(params['n_layers'])
embed_dim = int(params['embed_dim'])
feed_forward = int(embed_dim * 2)
key_dim = int(embed_dim//2)
dropout_rate = float(params['dropout'])
warmup_steps = int(params['warmup_steps'])
activation = params['activation']
epsilon = float(params['epsilon'])
epochs = int(params['epochs'])
batch_size = int(params['batch_size'])

class MultiHeadAttention(Layer):
    def __init__(self, num_heads, embed_dim, **kwargs):
        super().__init__(**kwargs)
        self.num_heads = num_heads
        self.embed_dim = embed_dim
        assert embed_dim % self.num_heads == 0, "embedded dim must be divisible by num_heads"
        self.depth = embed_dim // self.num_heads
        self.Wq = Dense(embed_dim)
        self.Wk = Dense(embed_dim)
        self.Wv = Dense(embed_dim)
        self.dense = Dense(embed_dim)

    def split_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, v, k, q):
        batch_size = tf.shape(q)[0]
        q = self.Wq(q)
        k = self.Wk(k)
        v = self.Wv(v)
        q = self.split_heads(q, batch_size)
        k = self.split_heads(k, batch_size)
        v = self.split_heads(v, batch_size)
        matmul_qk = tf.matmul(q, k, transpose_b=True)
        dk = tf.cast(tf.shape(k)[-1], tf.float32)
        scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)
        attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)
        output = tf.matmul(attention_weights, v)
        output = tf.transpose(output, perm=[0, 2, 1, 3])
        concat_attention = tf.reshape(output, (batch_size, -1, self.embed_dim))
        output = self.dense(concat_attention)
        return output, attention_weights

class TransformerBlock(Layer):
    def __init__(self, embed_dim, num_heads, dff, rate=0.1, **kwargs):
        super().__init__(**kwargs)
        self.mha = MultiHeadAttention(num_heads, embed_dim)
        self.ffn = tf.keras.Sequential([
            Dense(dff, activation='relu'),
            Dense(embed_dim)
        ])
        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)
        self.dropout1 = Dropout(rate)
        self.dropout2 = Dropout(rate)

    def call(self, x, training):
        attn_output, _ = self.mha(x, x, x)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(x + attn_output)

        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(out1 + ffn_output)
        return out2

class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, embed_dim, warmup_steps=5000):
        super().__init__()
        self.embed_dim = tf.cast(embed_dim, tf.float32)
        self.warmup_steps = warmup_steps

    def __call__(self, step):
        step = tf.cast(step, tf.float32)
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)
        return tf.math.rsqrt(self.embed_dim) * tf.math.minimum(arg1, arg2)

    def get_config(self):
        return {
            "embed_dim": self.embed_dim.numpy(),
            "warmup_steps": self.warmup_steps,
        }

def build_transformer_model(vocab_size, embed_dim, num_heads, ff, num_layers, rate=0.1):
    inputs = Input(shape=(None,))
    embedding_layer = Embedding(input_dim=vocab_size, output_dim=embed_dim)(inputs)
    x = embedding_layer
    for _ in range(num_layers):
        x = TransformerBlock(embed_dim, num_heads, ff, rate)(x)
    outputs = Dense(vocab_size, activation='softmax')(x)
    model = Model(inputs=inputs, outputs=outputs)
    return model