import tensorflow as tf

import logging

from sklearn.metrics import classification_report
from spacy.lang.en import English

import en_core_web_lg

from twitter_dataset import TwitterCorpus

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

embedding_model: English = en_core_web_lg.load()  # spacy.load('K:\\en_core_web_lg-2.3.1')

corpus = TwitterCorpus()

embedding_vectors = embedding_model.vocab.vectors.data


def get_word_index(word: str):
    return embedding_model.vocab.vectors.key2row.get(embedding_model.vocab.strings[word])


(x_train, x_test, y_train, y_test) = corpus.get_train_test_datasets(get_word_index)

sentiment_model = tf.keras.models.Sequential([
    tf.keras.layers.Embedding(input_dim=embedding_vectors.shape[0], output_dim=embedding_vectors.shape[1],
                              weights=[embedding_vectors], trainable=False),
    tf.keras.layers.LSTM(units=25, activation=tf.keras.activations.tanh),
    tf.keras.layers.Dense(units=2, activation=tf.keras.activations.softmax)
])

sentiment_model.compile(loss=tf.keras.losses.BinaryCrossentropy(),
                        optimizer=tf.keras.optimizers.Adam(),
                        metrics=[tf.keras.metrics.BinaryAccuracy()])

sentiment_model.fit(x_train, y_train, verbose=2, epochs=75, validation_split=0.15)

y_test_results = sentiment_model.predict(x_test)
single_y_test = [(1 if y >= 0.5 else 0) for (_, y) in y_test]
single_y_result = [(1 if y >= 0.5 else 0) for (_, y) in y_test_results]
print(classification_report(single_y_test, single_y_result))
