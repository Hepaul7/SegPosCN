#!/bin/bash
echo 'loading data'
python3 Data_Processing/data_processing.py
python3 train.py
