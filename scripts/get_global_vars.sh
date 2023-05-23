#!/bin/bash

DIR=$(readlink -f $(dirname ${BASH_SOURCE[0]}))

## Project

export PROJPATH=$DIR/..
export ORG_DATA=$PROJPATH/original-data
export INT_DATA=$PROJPATH/intermediate-data

export CMD=$PROJPATH/scripts/