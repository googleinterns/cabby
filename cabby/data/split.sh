# Filter entities that do not have text.
jq -c 'select(.text != "")'  /mnt/hackney/data/cabby/wikigeo/wikigeo_pittsburgh_raw.json > /mnt/hackney/data/cabby/wikigeo/wikigeo_pittsburgh.json
jq -c 'select(.text != "")'  /mnt/hackney/data/cabby/wikigeo/wikigeo_manhattan_raw.json > /mnt/hackney/data/cabby/wikigeo/wikigeo_manhattan.json


# Split the file into a train(%80), dev(10%) and test(%10) sets. 
jq -c -s 'sort_by(.qid)[:length*8/10]|.[]'  /mnt/hackney/data/cabby/wikigeo/wikigeo_pittsburgh.json > /mnt/hackney/data/cabby/wikigeo/wikigeo_pittsburgh_train.json 
jq -c -s 'sort_by(.qid)[length*8/10+1:length*9/10]|.[]'  /mnt/hackney/data/cabby/wikigeo/wikigeo_pittsburgh.json > /mnt/hackney/data/cabby/wikigeo/wikigeo_pittsburgh_dev.json 
jq -c -s 'sort_by(.qid)[length*9/10:]|.[]'  /mnt/hackney/data/cabby/wikigeo/wikigeo_pittsburgh.json > /mnt/hackney/data/cabby/wikigeo/wikigeo_pittsburgh_test.json 
jq -c -s 'sort_by(.qid)[:length*8/10]|.[]'  /mnt/hackney/data/cabby/wikigeo/wikigeo_manhattan.json > /mnt/hackney/data/cabby/wikigeo/wikigeo_manhattan_train.json 
jq -c -s 'sort_by(.qid)[length*8/10+1:length*9/10]|.[]'  /mnt/hackney/data/cabby/wikigeo/wikigeo_manhattan.json > /mnt/hackney/data/cabby/wikigeo/wikigeo_manhattan_dev.json 
jq -c -s 'sort_by(.qid)[length*9/10:]|.[]'  /mnt/hackney/data/cabby/wikigeo/wikigeo_manhattan.json > /mnt/hackney/data/cabby/wikigeo/wikigeo_manhattan_test.json 

