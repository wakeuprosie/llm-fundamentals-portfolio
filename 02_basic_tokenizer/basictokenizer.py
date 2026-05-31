#!/usr/bin/env python
class BasicTokenizer:
    #initialize token dictionary using UTF-8 and an empty dictionary to store all merges for easier decoding later
    def __init__(self):
        self.vocab = {i: bytes([i]) for i in range(256)}
        self.merges = {}

    #train the tokenizer on the input text file to create your token vocab
    def train(self, text, vocab_size):        
        #encode text into UTF-8 byte sequence
        tokens = list(text.encode("UTF-8"))

        #calculate how many merges we need to meet the desired vocab size
        num_merges = vocab_size - 256

        #the core BPE function
        for i in range(num_merges):
            #count all adjacent pairs seen in the text file and store in a dictionary called paircounts
            paircounts = {}
            for pair in zip(tokens, tokens[1:]):
                paircounts[pair] = paircounts.get(pair, 0) + 1

            if not paircounts:
                break

            #get the most frequent pairs and merge them into new tokens
            pairofinterest = max(paircounts, key=paircounts.get)
            newtokenid = 256 + i

            #add the new token id to the vocab and merges dictionaries
            self.vocab[newtokenid] = self.vocab[pairofinterest[0]] + self.vocab[pairofinterest[1]]
            self.merges[pairofinterest] = newtokenid

            #replace each pair in the text with the new token id
            pointer = 0
            while pointer < len(tokens) - 1:
                if tokens[pointer] == pairofinterest[0] and tokens[pointer+1] == pairofinterest[1]:
                    tokens[pointer] = newtokenid
                    tokens.pop(pointer+1)
                pointer += 1

            #print the merges done to confirm it's working
            print("merged ", pairofinterest, " as ", newtokenid)

    #this encodes text to tokens
    def encode(self, text): 
        #encode text first to UTF-8 byte sequences
        tokens = list(text.encode("UTF-8"))

        #go pair by pair in the merges dictionary, and replace all occurrences of the pair with the new token id
        for item in self.merges:
            i = 0
            while i < len(tokens) - 1:
                if (tokens[i], tokens[i+1]) == item:
                    tokens[i] = self.merges.get(item)
                    tokens.pop(i+1)
                    i += 1
                else:
                    i += 1
        return tokens

    #this decodes tokens back to text
    def decode(self, ids):
        #find the token in the vocab dictionary, replace with its associated UTF-8 byte sequence, then encode that back to text
        textinbytes = []
        for item in ids:
            byte = self.vocab.get(item)
            textinbytes.append(byte)
        return b''.join(textinbytes).decode('UTF-8')

if __name__ == "__main__":
    # Quick test/example usage
    tokenizer = BasicTokenizer()
    print("BasicTokenizer initialized.")
