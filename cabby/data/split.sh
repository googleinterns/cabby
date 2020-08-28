#!/bin/bash

read directory_name

# Filter entities that do not have text.
jq -c 'select(.text != "")'  ${directory_name}wikigeo_pittsburgh_raw.json > ${directory_name}wikigeo_pittsburgh.json
jq -c 'select(.text != "")'  ${directory_name}wikigeo_manhattan_raw.json > ${directory_name}wikigeo_manhattan.json


# Split the file into a train(%80), dev(10%) and test(%10) sets. 
jq -c -s 'sort_by(.qid)[:length*8/10]|.[]'  ${directory_name}wikigeo_pittsburgh.json > ${directory_name}wikigeo_pittsburgh_train.json 
jq -c -s 'sort_by(.qid)[length*8/10+1:length*9/10]|.[]'  ${directory_name}wikigeo_pittsburgh.json > ${directory_name}wikigeo_pittsburgh_dev.json 
jq -c -s 'sort_by(.qid)[length*9/10:]|.[]'  ${directory_name}wikigeo_pittsburgh.json > ${directory_name}wikigeo_pittsburgh_test.json 
jq -c -s 'sort_by(.qid)[:length*8/10]|.[]'  ${directory_name}wikigeo_manhattan.json > ${directory_name}wikigeo_manhattan_train.json 
jq -c -s 'sort_by(.qid)[length*8/10+1:length*9/10]|.[]'  ${directory_name}wikigeo_manhattan.json > ${directory_name}wikigeo_manhattan_dev.json 
jq -c -s 'sort_by(.qid)[length*9/10:]|.[]'  ${directory_name}wikigeo_manhattan.json > ${directory_name}wikigeo_manhattan_test.json 

