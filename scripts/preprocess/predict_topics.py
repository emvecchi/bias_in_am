import bz2, json
import projpath
import pickle
import numpy as np
from bertopic import BERTopic

from common.normalize import normalizeTextForParse

def correct_alternative_cosine(ds):
    result = np.empty_like(ds)
    for i in range(ds.shape[0]):
        result[i] = 1.0 - np.power(2.0, ds[i])
    return result

import pynndescent
pynn_dist_fns_fda = pynndescent.distances.fast_distance_alternatives
pynn_dist_fns_fda["cosine"]["correction"] = correct_alternative_cosine
pynn_dist_fns_fda["dot"]["correction"] = correct_alternative_cosine

def readFile(filename):
    result = [ ]

    with open(filename, 'r') as data:
        for line in data:
            cmv = json.loads(line)
            result.append(cmv)
    
    return result

def get_post_text(cmv):
    result = [ normalizeTextForParse(cmv["title"]) ]

    for line in cmv["selftext"].split("\n"):
        clean_text = normalizeTextForParse(line)
        
        if clean_text:
            result.append(clean_text)

    return "\n".join(result)

if __name__ == '__main__':
    import os, sys

    args = sys.argv[1:]
    dataFile = args[0]
    outInfo = args[1]
    outData = args[2]

    fullData = readFile(dataFile)
    data = [ get_post_text(cmv) for cmv in fullData ]

    print("Data read", len(data))

    topic_model = BERTopic()
    topics, probs = topic_model.fit_transform(data)

    print("Model trained", len(topics))
    print(topic_model.get_topic_info())

    with open(outInfo, "w") as out_topics:
        for index, row in topic_model.get_topic_info().iterrows():
            out_topics.write(str(row['Topic']) + ": " + row['Name'] + "\n")

    out = open(outData, "w")
    for (cmv, topic) in zip(fullData, topics):
        out.write(cmv["id"] + "\t" + str(topic) + "\n")