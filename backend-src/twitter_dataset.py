from typing import List, Dict, Set, Iterable, Callable

import numpy as np
# from nltk.corpus import twitter_samples
from sklearn.model_selection import train_test_split
import csv


class TwitterCorpus(object):

    def __init__(self):
        # twitter_samples.ensure_loaded()

        self.positive_tweets: List[str] = []
        self.negative_tweets: List[str] = []
        with open('data/mldata1/mldata1.csv', 'r', encoding='utf8') as corpus_file:
            parser = csv.reader(corpus_file)
            next(parser)

            while True:
                parsed_line = next(parser)
                if parsed_line[1] != '0':
                    break
                self.negative_tweets.append(parsed_line[2])

            try:
                while True:
                    self.positive_tweets.append(parsed_line[2])
                    parsed_line = next(parser)
            except StopIteration:
                pass

        # self.positive_tweets: List[str] = twitter_samples.strings('negative_tweets.json')
        # self.negative_tweets: List[str] = twitter_samples.strings('positive_tweets.json')
        self.processed_docs = [self.preprocess_doc(doc) for doc in self.negative_tweets + self.positive_tweets]

    def preprocess_doc(self, doc: str):
        alpha_str = [(this_char if (this_char.isalpha() or this_char == '\'') else ' ') for this_char in doc]
        return ''.join(alpha_str).split()

    def __2d_numpy_array_from_jagged_lists(self, lists, fill_value):
        longest_sublist = max(len(this_list) for this_list in lists)
        return np.array([[fill_value] * (longest_sublist - len(xi)) + xi for xi in lists])

    def word_to_docs_map(self):
        result_dict: Dict[str, Set[int]] = {}

        for (index, this_doc) in enumerate(self.processed_docs):
            for this_word in this_doc:
                lower_word = this_word.lower()
                if lower_word in result_dict:
                    result_dict[lower_word].add(index)
                else:
                    result_dict[lower_word] = {index}

        return result_dict

    def docs_to_numpy_array(self, doc_ids: Iterable[int], word_index_function: Callable) -> np.ndarray:
        doc_word_indices = []
        for this_doc in (self.processed_docs[i] for i in doc_ids):
            word_indices = []
            for this_word in this_doc:
                word_index = word_index_function(this_word)
                if word_index is None:
                    word_index = word_index_function(this_word.replace('\'', ''))

                if not (word_index is None):
                    word_indices.append(word_index)

            doc_word_indices.append(word_indices)

        np_array = self.__2d_numpy_array_from_jagged_lists(doc_word_indices, 0)
        return np_array[:, -30:]

    def get_train_test_datasets(self, word_index_function: Callable):
        x_values = self.docs_to_numpy_array(range(len(self.processed_docs)), word_index_function)

        y_values = [(0, 1) for _ in range(len(self.negative_tweets))] + [(1, 0) for _ in range(len(self.positive_tweets))]
        y_values = np.array(y_values)
        return train_test_split(x_values, y_values, shuffle=True, test_size=0.97, random_state=1)