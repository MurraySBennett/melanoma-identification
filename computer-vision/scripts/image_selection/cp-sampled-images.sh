#!/bin/bash

# check if target directory is provided
if [ -z "$1" ] || [ -z "$2" ] || [ -z "$3" ]; then
    echo "Usage: $0 [target_directory] [image_file_path] [image_list]"
    exit 1
fi

if [ "$(ls -A "$1")" ]; then
    echo "Target directory is not empty. Do you want to overwrite the existing files? (y/n)"
    read answer
    if [ "$answer" = "y" ]; then
        echo "Clearing directory contents..."
        rm -rf "$1"/*
    else
        exit 0
    fi
fi

tail -n +2 "$3" | sed "s|^|$2/|;s|$|.JPG|" | xargs cp -t "$1"

