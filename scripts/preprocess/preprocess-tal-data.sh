#!/bin/bash

set -ue

D=$(readlink -f $(dirname ${BASH_SOURCE[0]}))

source $D/../get_global_vars.sh


## filtering gender for train files

TRAIN="train"
TRAIN_FILE=$ORG_DATA/cmv/tan_etal_2016/all/${TRAIN}_period_data.jsonlist.bz2

TRAIN_DIR=$INT_DATA/tal_elal_2016/${TRAIN}/annotations/

if [[ ! -e $TRAIN_DIR ]]; then
    mkdir -p $TRAIN_DIR
fi

trainThreadGenders=$TRAIN_DIR/explicit_gender.threads.txt
trainReplyGenders=$TRAIN_DIR/explicit_gender.replies.txt
trainAuthorGenders=$TRAIN_DIR/author_gender.authors.txt

#python $CMD/explicit/explicit_gender.py $TRAIN_FILE $trainThreadGenders $trainReplyGenders $trainAuthorGenders

## filtering gender for heldout files

HELDOUT="heldout"
HELDOUT_FILE=$ORG_DATA/cmv/tan_etal_2016/all/${HELDOUT}_period_data.jsonlist.bz2

HELDOUT_DIR=$INT_DATA/tal_elal_2016/${HELDOUT}/annotations/

if [[ ! -e $HELDOUT ]]; then
    mkdir -p $HELDOUT
fi

heldoutThreadGenders=$HELDOUT_DIR/explicit_gender.threads.txt
heldoutReplyGenders=$HELDOUT_DIR/explicit_gender.replies.txt
heldoutAuthorGenders=$HELDOUT_DIR/author_gender.authors.txt

#python $CMD/explicit/explicit_gender.py $HELDOUT_FILE $heldoutThreadGenders $heldoutReplyGenders $heldoutAuthorGenders

ANN_DIR=$INT_DATA/tal_elal_2016/annotations/

if [[ ! -e $ANN_DIR ]]; then
    mkdir -p $ANN_DIR
fi

threadGenders=$ANN_DIR/explicit_gender.threads.txt
replyGenders=$ANN_DIR/explicit_gender.replies.txt
authorGenders=$ANN_DIR/author_gender.authors.txt

cat $trainThreadGenders $heldoutThreadGenders > $threadGenders
cat $trainReplyGenders $heldoutReplyGenders > $replyGenders
cat $trainAuthorGenders $heldoutAuthorGenders > $authorGenders

OUT_TRAIN=$INT_DATA/tal_elal_2016/${TRAIN}_period_data.annotations.jsonlist
python $CMD/preprocess/annotate_additional_fields.py $TRAIN_FILE $ANN_DIR $OUT_TRAIN

OUT_HELDOUT=$INT_DATA/tal_elal_2016/${HELDOUT}_period_data.annotations.jsonlist
python $CMD/preprocess/annotate_additional_fields.py $HELDOUT_FILE $ANN_DIR $OUT_HELDOUT