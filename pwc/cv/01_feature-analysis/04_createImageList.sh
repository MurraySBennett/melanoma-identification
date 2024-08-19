#!/bin/bash

if [ $# -ne 3 ]; then
    echo "Usage: $0 in_file out_file var_name"
    exit 1
fi

# set name of input file
# IN_FILE="sampled_shape_ids.txt"
IN_FILE="$1"
OUT_FILE="$2"
VAR_NAME="$3"

# set name of output file
# OUT_FILE="imageList.txt"

# create output file and start the imageList array
echo "window.${VAR_NAME} = [" > "$OUT_FILE"

# read in file line by line and add image ID to imageList
awk 'NR>1' $IN_FILE | while IFS= read -r IMAGE_ID
do
    IMAGE_NAME="${IMAGE_ID}.JPG"
    echo "\"${IMAGE_NAME}\", " >> "$OUT_FILE"
done < "$IN_FILE"

# remove trailing comma from array 
sed -i '$ s/,//' "$OUT_FILE"
# close the array and export as constant
echo "];" >> "$OUT_FILE"
echo "" >> "$OUT_FILE"
# echo "export { imageList };" >> "$OUT_FILE"

