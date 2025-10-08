close all; clear; clc
%% 
file_path = ['..' filesep 'data' filesep 'cleaned' filesep 'btl-asymmetry.csv'];
bor = readtable(file_path, TextType="string");

bor = removevars(bor, ["sender", "timestamp", ])
