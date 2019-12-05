#!/bin/bash
DIR="/local/gong/testdata"

for entry in "$DIR"/*
do
    echo $entry
    FileName=`basename $entry`
    FileNameWithoutExt=${FileName%.*}
    echo $FileNameWithoutExt
    python gluon-test.py $entry
done

