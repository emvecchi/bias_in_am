#!/bin/bash

set -ue

D=$(readlink -f $(dirname ${BASH_SOURCE[0]}))

source $D/../get_global_vars.sh

DATA_FILE=$DATA/cmv.jsonlist

OUT_DIR=$PROJPATH/to_annotate/
mkdir -p $OUT_DIR/threads
mkdir -p $OUT_DIR/responses

python $CMD/explicit/highlight_genders.py $DATA_FILE threads $OUT_DIR/threads
python $CMD/explicit/highlight_genders.py $DATA_FILE responses $OUT_DIR/responses