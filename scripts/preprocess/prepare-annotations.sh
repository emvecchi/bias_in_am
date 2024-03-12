#!/bin/bash

set -ue

D=$(readlink -f $(dirname ${BASH_SOURCE[0]}))

## Project paths

source $D/../get_global_vars.sh

## File to start from
data_file=$DATA/cmv.jsonlist

if [ ! -e $data_file ]; then
    train_file=$ORG_DATA/all/train_period_data.jsonlist.bz2
    heldout_file=$ORG_DATA/all/heldout_period_data.jsonlist.bz2
    python $CMD/common/normalize_cmv.py $train_file $heldout_file $data_file 
fi

## Filtering authors' gender
explicit_gender=$PROJPATH/annotations/explicit_gender.txt
cat $PROJPATH/manual_annotations/*explicit_gender.txt > $explicit_gender

authors_gender=$PROJPATH/annotations/authors_gender.txt

if [ ! -e $authors_gender ]; then
    python $CMD/preprocess/explicit_to_implici.py $data_file $explicit_gender $authors_gender
fi

## Topics
topics_info=$PROJPATH/annotations/topics.info
topics_file=$PROJPATH/annotations/topics.txt
if [ ! -e $topics_file ]; then
    CUDA_VISIBLE_DEVICES=1 python $CMD/preprocess/predict_topics.py $data_file $topics_info $topics_file
fi

## Merge annotations
ann_file=$DATA/cmv.annotations.topics.jsonlist
if [ ! -e $ann_file ]; then
    python $CMD/preprocess/merge_annotations.py $data_file $explicit_gender $authors_gender $topics_file $ann_file
    rm $data_file
fi