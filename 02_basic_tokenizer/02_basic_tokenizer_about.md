# 02_tokenization_lab/

Tokens are the “atoms” of LLMs and frequently a source of the quality issues seen in LLM behavior. Tokens are essentially representations of chunks of text data. LLMs process information on the level of tokens instead of raw text. It’s like the LLM’s ‘alphabet’.

## Why tokens? 
The primary reason we use tokens is for computational efficiency - it’s very inefficient to process text data on the level of characters. Using tokens reduces the computational power required by 4x as compared to character-level processing. What you aim to do through tokens is to maximize semantic context you can fit into a model while minimizing the computational "size" of the info - context is critical in LLMs. One token can represent a whole word, a punctuation, part of a word, a number, a symbol, etc. Tokens directly impact cost, latency, context window usage, prompt design, and evaluation design.

## Where do tokens come from? 
A tokenizer is a model that encodes and decodes raw text into tokens using a pre-established vocabulary of tokens to byte sequences. You establish this vocabulary by training the tokenizer with a large text data set. This data set is typically *different* from the set you use to train the LLM model. 

## How does the tokenizer work:
In a nutshell, you 'chunk' up the text data according to common patterns. Then you give each chunk a unique ID and this becomes your token. You continue doing this until you get to your desired token vocab size. The token vocab represents all possible tokens it knows.

You can use this website Andrej Karpathy references in his lecture visualize how Chat GPT-4 tokenizes text in real-time: https://tiktokenizer.vercel.app/?model=gpt-4.

Having a larger token vocab does not necessarily mean better. Chat GPT-5 and Gemini Pro 3.1 have vocab sizes in the range of 200K-250K, while older versions of these models had vocabularies in the 100K range. The jump in size was primarily to improve quality of performance with non-Latin languages and coding use cases - which generally require more tokens per context.

## A basic tokenizer implementation
See basictokenizer.py for a working implementation of basic tokenizer in python.
