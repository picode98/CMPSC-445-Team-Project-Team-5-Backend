import inference

import json

with open('sample_topics.txt', 'r') as sample_topics_file:
    sample_topics = sample_topics_file.readlines()

inference_results = {this_topic: inference.predict_avg_sentiment(this_topic) for this_topic in sample_topics}

result_dict = {'entries': [{'name': key, 'sentiment': value[0], 'amountProcessed': value[1]}
                           for (key, value) in inference_results.items()]}

with open('batch_results.json', 'w') as batch_results_file:
    json.dump(result_dict, batch_results_file)
