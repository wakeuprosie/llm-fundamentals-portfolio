#!/usr/local/google/home/risapark/.venv/bin/python
import os 
import sys  
from tokenizer import BasicTokenizer

tokenizer = BasicTokenizer()

#find the path of the text file
script_dir = os.path.dirname(os.path.abspath(__file__))
sample_path = os.path.join(script_dir, "sample.txt")

#open text file
with open(sample_path, "r", encoding="utf-8") as f:
    text = f.read()

#initialize tokenizer and run
tokenizer.train(text, 300)
print("vocab: ", tokenizer.vocab)
print("merges: ", tokenizer.merges)
