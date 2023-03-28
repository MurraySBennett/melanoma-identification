#!/usr/bin/env bash

file_path=$1
n_lines=`sed "/^\s*$/d" $file_path | wc -l`
div_pairs=2
n_remove=$(( n_lines / div_pairs ))

echo "Total items: $n_lines"
echo "Half of that: $n_remove"

