#!/usr/bin/env python

import sys,os

thisFile = os.path.realpath(__file__)
thisPath = os.path.dirname(thisFile)
parentPath = os.path.dirname(thisPath)
sys.path.append(parentPath)