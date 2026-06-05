# Transformer Architecture
The transformer step is where the query understanding and output generation happens in the LLM model. The key components of a transformer are attention, the feed-forward network (FFN), also known as the multi-layer perceptron (MLP), and positional encoding. Learning about transformers was where I learned to appreciate the sheer amount of calculations happening in LLMs. This file outlines the basics of transformer architecture.

<img src="decoder_transformer_visual_diagram.svg" alt="Decoder Transformer Visual Diagram" width="800">

## Positional Encoding
Technically this is a pre-cursor step to the transformer. Embeddings capture semantics, but not order. So you add in positional encoding to give the transformer info about where each token is in the sequence. 

Positional information is stored in a weighted matrix of size [max_sequence_length, embedding_dim]. This is often called P. P is initialized randomly and then updated through backpropagation.  For each token in the input query, you look up its corresponding P vector based on its position in the sequence and add it to the token's embedding vector. Concretely, row 0 in the P matrix = a vector representing being at position 0 in the sequence. For context, Claude's Sonnet 4.6 supports a 500K context window in paid chat plans on claude.ai while Claude Code supports 1M.

With the token's positional info encoded in its vector, you move onto to the attention layer. 

## Attention Layer
The attention layer's purpose is to incorporate context of surrounding tokens into each token. An analogy used is comparing the meaning of the word 'model' in the phrase 'an LLM model' vs 'a fashion model'. In different contexts, 'model' means quite different things. Attention layer incorporates this kind of relational context.

B,T,C are the key parameters of the attention layer. These parameters determine how the model processes your input query. 
B represents batches, how many batches of token sequences you'll process in parallel.
T represents token sequence size - how long is each token sequence you'll analyze.
C represents channels - each channel represents some info about the token.
Essentially they determine the size of the tensor input that flows to the attention layer. You break up the input query into token sequences. So B and T values are dynamic based on your input query at runtime. T is the hard constraint - which has a max_sequence_length that you set at the positional encoding step prior.

Second key concept in the attention layer is that each layer contains multiple ‘heads’. Each 'head' contains weights that look for certain features of surrounding token relationships. 
* Early attention layers tend to address low level patterns like syntax, word order, and punctuation.
* Mid layers are more about semantic relationships: subjects to objects, conference (“he” means “Michael”).
* Late layers are more abstract and about factual associations or reasoning.
* An analogy for the different layers:
    * First pass: identify the words and grammar
    * Second pass: understand who is doing what
    * Third pass: understand the deeper meaning and implications

Step-by-step of what happens in each token sequence: I'm going to follow the sizes used in GPT-2 to help you track how the vectors flow through each step. 

* Get the embedding vector for T from the static embedding table. Let's assume 1-D array size 768. The embedding vector at this point has no context baked in.
* Apply normalization to the vector - you do this because the embedding vector can vary wildly in size, and you want to reset the scale to keep the signal strength stable through each layer of calculations.
* Dot product this embedding vector with Wq and Wk matrixes to get Q and K vectors. These vectors determine what information is important for each token. Q represents what information this token should pay attention to and K represents what information this token contains. Each vector is size 1-D. Let's assume 1-D array size 64 - this is a design choice.
* Dot product the Q and K vectors to calculate V vector. V represents what information this token passes along to other tokens.
* The attention layer has multiple heads. We just described going though 1 head. You would go through all heads in the same manner and end up with multiple V vector outputs. 
* Concatenate the outputs into one final 1-D vector. Remember each head vector 1-D size 64. In GPT-2, there's 12 heads, so when you concatenate them you end up with a 1-D array of size 768. (You're back to the original embedding vector shape!)
* Apply normalization again to this vector.
* Now you have a new embedding vector with context baked in, this can now flow through the MLP/FFN layer.

Briefly on the size of Q and K, we mentioned this is a design choice. You typically decide C, (for B,T,C), then you pick the number of heads, and then with C and D you get the Q and K size. In our example for GPT-2, C = 768, heads = 12, so Q/K = 768 channels /12 heads = 64.

## Multi-Layer Perceptron (MLP)
The multi-layer perception (MLP) layer takes in the contextualized embedding vector and adds in the relevant knowledge/info. 

Step-by-step:
* Matrix multiply the embedding vector with yet another, learned weighted matrix. In GPT-2, the dimensions are (768, 3072) - so the vector is getting larger at this step. Each row of this weighted matrix contains questions about the token - like ‘are you a noun’? 'are you related to a European city?'.
* The output is an array of numbers ranging from negative to positive.
* Add bias. This is a simple addition and it dictates how much you want to adjust the embedding vector of your token according to those numbers, it's like adjusting the intensity of the output from the matrix multiply step.
* Run through Relu activation function. Tactically, this removes all negative numbers. In concept, what you're doing is creating a threshold for applying the calculated changes - basically only the large changes get added here.
* Then size the output back down to embedding - size 768.
* Add this output to the embedding vector. 

Your output is an emebdding vector that represents a token with context and info baked in. This is after 1 pass of attn and MLP.

You repeat this process over how many ever layers your transformer has. These layers are where the name ‘deep’ learning comes from. 

Similarly to each attention layer, each FFN layer address different informational aspects of the tokens. Early mlp layers may shape more surface level attributes of the tokens, middle layers contain most of the factual and sematic knowledge, and late layers refine the representation of tokens towards the final next token prediction.

We discussed typically there are multiple rounds of passing through attention and MLP/FFN layers. after you finish all the rounds of attention and FFN, you can move to inference - aka generating the next token. 

## Inference - Next Token Generation
You do this using the output of the final MLP pass. Then you take the very LAST embedding vector of your sequence, and use just this to do inference.

You transform the embedding vector into an array of scores by dot product with yet another weighted matrix - this represents the scores for each possible next token. 
Then you run this through softmax to turn this into distributed probabilities.
Then you generate the token with the highest probability.

Now you have your token sequence + one new token. 

## Key Transformer Takeaways
All of this happens in parallel, for the whole input query. Every batch, every sequence, and every token is being processed all at once, that’s why the GPUs are so important - they enable this parallel processing.

Modern transformers like GPT-5 contain 120 layers, compared to older versions like GPT-2 which had only 12.
Lately the trend is moving towards wider models, not just deeper - so bigger channels per token (the C value in B,T,C).

More layers = more depth and context, potentially better inference up to a point
But also more layers = slower inference
And also more layers = harder to train because more layers to back propagate through
Memory and compute costs scale linearly with the number of layers!

