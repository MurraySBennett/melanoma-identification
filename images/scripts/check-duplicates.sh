#!/usr/bin/env bash

fdupes ~/win_home/melanoma-identification/images/ISIC-database > ~/win_home/melanoma-identification/images/duplicates/duplicate.txt

# run fsdupes and return unique files
fdupes -f ~/win_home/melanoma-identification/images/ISIC-database > ~/win_home/melanoma-identification/images/duplicates/rm-list.txt

