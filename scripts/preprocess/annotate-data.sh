#!/bin/bash

set -ue

D=$(readlink -f $(dirname ${BASH_SOURCE[0]}))

## Project paths

source $D/../get_global_vars.sh

## File to start from
data_file=$INT_DATA/cmv.jsonlist

## Filtering authors' gender
explicit_gender=$INT_DATA/explicit_gender.txt
cat $PROJPATH/manual_annotations/* > $explicit_gender

authors_gender=$INT_DATA/authors_gender.txt

if [ ! -e $authors_gender ]; then
    python $CMD/preprocess/find_authors_gender.py $data_file $explicit_gender $authors_gender
fi

## Topics
topics_info=$INT_DATA/topics.info
topics_file=$INT_DATA/topics.txt
if [ ! -e $topics_file ]; then
    CUDA_VISIBLE_DEVICES=1 python $CMD/preprocess/predict_topics.py $data_file $topics_info $topics_file
fi