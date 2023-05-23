#!/bin/bash

set -ue

D=$(readlink -f $(dirname ${BASH_SOURCE[0]}))

source $D/../get_global_vars.sh

VER=$1

DATA_DIR=$ORG_DATA/cmv/hidey_etal_2017/change-my-view-modes/$VER/

OUT_DIR=$INT_DATA/hidey_elal_2017/$VER
if [[ ! -e $OUT_DIR ]]; then
    mkdir -p $OUT_DIR
fi

# filtering thread ids
threadIds=$OUT_DIR/$VER.thread.ids
cat $DATA_DIR/*/*xml | grep "ID=" | sort | uniq > $threadIds

# filtering reply ids
replyIds=$OUT_DIR/$VER.reply.ids
cat $DATA_DIR/*/*xml | grep "<reply id=" | sort | uniq > $replyIds

# filtering tal
TAL_ETAL=$ORG_DATA/cmv/tan_etal_2016/all/train_period_data.jsonlist.bz2
jsonFile=$OUT_DIR/$VER.jsonlist
python $CMD/preprocess/filter_json_items.py $TAL_ETAL $threadIds $replyIds $jsonFile

jsonThreads=$OUT_DIR/$VER.thread.jsonlist
jsonReplies=$OUT_DIR/$VER.reply.jsonlist
python $CMD/preprocess/filter_json_fields.py $jsonFile $jsonThreads $jsonReplies