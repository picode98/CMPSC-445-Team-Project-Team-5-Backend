from typing import Dict, List, Iterable, Set, Tuple, Union
from nltk.corpus import wordnet, WordNetCorpusReader

import tensorflow as tf

import os
from functools import reduce

from tensorflow.python.keras.models import Sequential

from spacy.lang.en import English
import en_core_web_lg

from twitter_dataset import TwitterCorpus

sentiment_model_key = 'SENTIMENT_MODEL'
if sentiment_model_key in os.environ:
    sentiment_model: Sequential = tf.keras.models.load_model(os.environ[sentiment_model_key], compile=True)
else:
    raise Exception(f'Failed to load sentiment model: {sentiment_model_key} environment variable not specified.')

embedding_model: English = en_core_web_lg.load(disable=['tagger', 'parser', 'ner'])


def get_word_index(word: str):
    return embedding_model.vocab.vectors.key2row.get(embedding_model.vocab.strings[word])


corpus = TwitterCorpus()
words_to_docs = corpus.word_to_docs_map()


def union_all(sets: Iterable[Set]):
    return reduce(lambda existing, new_item: existing.union(new_item), sets, set())


def intersection_all(sets: Iterable[Set]):
    try:
        iterator = iter(sets)
        first_set = next(iterator)
    except StopIteration:
        return set()

    return reduce(lambda existing, new_item: existing.intersection(new_item), iterator, first_set)


def search_docs(keywords: List[str]):
    lower_keywords = (this_keyword.lower() for this_keyword in keywords)

    return intersection_all(words_to_docs[this_keyword] for this_keyword in lower_keywords if (this_keyword in words_to_docs))


wordnet.ensure_loaded()
wordnet: WordNetCorpusReader


def predict_avg_sentiment(topic: str, lexical_type: str = None) -> Tuple[Union[float, None], int]:
    """
    Predict the aggregate sentiment polarity for the given topic from the Twitter corpus.

    Example: predict_avg_sentiment('cat', 'noun.animal')

    :param topic: The topic on which to aggregate sentiment
    :param lexical_type: The WordNet lexical type/lexname file of topic for finding direct replacements/aliases
            for topic (e.g. "car" -> "automobile").
    :return: A tuple (sentiment polarity [0 - 1], number of documents sampled). If no documents were sampled,
    the sentiment polarity value will be None.
    """

    if lexical_type is None:
        all_filtered_results = [topic.split()]
    else:
        result = wordnet.synsets('_'.join(topic.split()))
        filtered_result = [synset.lemma_names() for synset in result if synset.lexname() == lexical_type]
        all_filtered_results = [this_alias.split('_') for this_alias_list in filtered_result for this_alias in
                                this_alias_list]

    result_doc_ids = union_all(search_docs(these_keywords) for these_keywords in all_filtered_results)

    if len(result_doc_ids) == 0:
        return None, 0

    predictions = sentiment_model.predict(corpus.docs_to_numpy_array(result_doc_ids, get_word_index))
    prediction_avg = sum(predictions[:, 0]) / predictions.shape[0]
    # actual_avg = sum(i >= 800000 for i in result_doc_ids) / len(result_doc_ids)
    # print(f'Predicted: {prediction_avg}; actual: {actual_avg}')
    return prediction_avg, len(result_doc_ids)


print('Sentiment model loaded.')
