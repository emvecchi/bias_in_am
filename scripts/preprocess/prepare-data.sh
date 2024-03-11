#!/bin/bash

set -ue

D=$(readlink -f $(dirname ${BASH_SOURCE[0]}))

## Project paths

source $D/../get_global_vars.sh

mkdir -p $ORG_DATA
mkdir -p $INT_DATA

## Download data

link="https://chenhaot.com/data/cmv/cmv.tar.bz2"
wget -nc -P $ORG_DATA $link

## Unpack data

if [ ! -d "$ORG_DATA/all" ]; then
    tar -xjf $ORG_DATA/cmv.tar.bz2 -C $ORG_DATA
fi

## Prepare files

train_file=$ORG_DATA/all/train_period_data.jsonlist.bz2
heldout_file=$ORG_DATA/all/heldout_period_data.jsonlist.bz2
data_file=$INT_DATA/cmv.jsonlist

if [ ! -e $data_file ]; then
    python $CMD/common/normalize_cmv.py $train_file $heldout_file $data_file 
fi