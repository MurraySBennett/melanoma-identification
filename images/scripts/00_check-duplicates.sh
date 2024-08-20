#!/usr/bin/env bash

unset -v img_location
unset -v save_destination

save_name="duplicate.txt"
rm_name="rm-list.txt"
while getopts i:s: flag; do
    case $flag in
        i) img_location=$OPTARG ;;
        s) save_destination=$OPTARG ;;
        *)
            echo 'Error in cmd line parsing' >&2
            exit 1
    esac
done
shift "$(( OPTIND - 1 ))"

if [ -z "$img_location" ]; then
    echo 'Missing -i flag: must include path to image directory' >&2
    exit 1
fi
if [ -z "$save_destination" ]; then
    echo 'Missing -s flag: must include directory path to save the output txt.' >&2
    exit 1
fi

echo "Checking for duplicate images in: $img_location"
fdupes $img_location > $save_destination/$save_name
echo "Saving list of identified duplicates to $save_destination/$rm_name"

fdupes -f $img_location > $save_destination/$rm_name
echo "Saving list of to-be-removed image IDs to $save_destination/$save_name"

# fdupes ~/win_home/melanoma-identification/images/ISIC-database > ~/win_home/melanoma-identification/images/duplicates/duplicate.txt
# run fsdupes and return unique files
# fdupes -f ~/win_home/melanoma-identification/images/ISIC-database > ~/win_home/melanoma-identification/images/duplicates/rm-list.txt