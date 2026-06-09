# Transformer Architecture
The transformer step in an LLM is where the input query understanding and response generation actually happens. The key components of a transformer are 1. attention and 2. the feed-forward network(FFN) (also commonly called multi-layer perceptron (MLP)). Learning about transformers is where I really started to appreciate the sheer amount of calculations happening every time we query an LLM. Outlined below are the basics of transformer architecture for a decoder-only transformer, which is the basis for most LLMs today like GPT-series and Claude.

<img src="decoder_transformer_visual_diagram.svg" alt="Decoder Transformer Visual Diagram" width="800">

## Positional Encoding 
In the transformer framework you have an **embedding dictionary** which is like a dictionary of what each token represents. Conceptually, the embedding vectors represent each token's position in a multi-dimensional space, which helps represent its meaning.

These embeddings capture semantics, but not token position, so you need to add in **positional encoding**. The position of a token in a sequence matters for its meaning. For example, in "I can't wait to see the dog", the word 'dog' at the end of the sentence is the main subject of the sentence. If we swapped the position of 'I' and 'dog', the entire meaning of the sentence would change completely.

Positional information is stored in a weight matrix called P or **positional encoding matrix**. P is initialized randomly and updated through backpropagation. For each token in the input query, you look up its corresponding P vector and add it to the token's embedding vector. For context, Claude's Sonnet 4.6 supports a 500K context window in paid chat plans on claude.ai while Claude Code supports 1M.

With the token's positional info encoded in its vector, you move on to the attention layer. 

## Attention Layer
The attention layer's purpose is to incorporate context of surrounding tokens into each token. A common analogy used is comparing the meaning of the word 'model' in the phrase 'an LLM model' vs 'a fashion model'. In different contexts, the word 'model' means different things. The attention step incorporates this type of relational context into each token.

**B,T,C** are the key parameters of the attention layer. These parameters determine how the model processes your input query. Essentially, the query enters the attention layer as a tensor of shape (B,T,C).
* B represents batches - how many token sequences you'll process in parallel.
* T represents token sequence size - how long is each sequence you'll analyze.
* C represents channels - each channel represents some info about the token.
Here, B and T values are dynamic based on your input query. However, T is the hard constraint with a max_sequence_length, which you set at the positional encoding step prior.

The second key point is that each attention layer contains multiple **‘heads’**. Each head contains weights that look for certain features of surrounding token relationships. 
* Early attention layers tend to address low level patterns like syntax, word order, and punctuation.
* Mid layers are more about semantic relationships: subjects to objects, conference (“he” means “Michael”).
* Late layers are more abstract and about factual associations or reasoning.

Lastly, there are **multiple attention layers**. In each layer you build more contextual understanding than the one before it. In the example of GPT-2, there are 12 attention layers. So you repeat this process of attention and normalization 12 times.

**A step-by-step of attention:**
I'm going to discuss the tensor shapes and sizes in the context of GPT-2 to help you track what's happening in each step.

* First, you have your B,T,C input shape established. Then for every token in each T, you look up its **embedding vector**, E, from the static embedding table. Remember, at this point the vector has no context baked in.
    * In the example of GPT-2, the input tensor shape is [B, T, 768] and each embedding vector is size 1 x 768.
* Apply **normalization** to E. You do this because the embedding vector can vary wildly in size as you go through the transformer layers, so you want to reset the scale to keep the signal strength stable through each layer.
* Matrix multiply E for each token in the T sequence with **Wq**, **Wk**, and **Wv** matrixes to get each token's **Q**, **K**, and **V** vectors. All three matrixes are learned through backpropagation and help determine what information is important for each token. Q represents what information each token should pay attention to, K represents what information each token contains, and V contains what info to pass along. Each vector is then partitioned into h separate heads and downsized (to dhead size).
    * In GPT-2, Wq, Wk, and Wv are size 768 x 768. Each Q, K, and V vector is size 1 x 768. After partition, each head vector is downsized to 1 x 64.
* In each head, you dot product Q and K to calculate the **attention pattern**. Then you mask out the upper triangle of the resulting matrix - this ensures you only attend to preceding tokens. The output is a matrix of floating point scores that represent how closely related each token pair in the sequence is.
    * In GPT-2, when you dot product Q and K the output is one floating point value. For the full sequence, you will have a T x T shape of attention scores in the attention pattern.
* Each score in the attention map is scaled down by dividing by the square root of the dimension size, and then you apply **softmax** to get your final **scores matrix** - this is still a T x T shape. The softmax ensures the scores of each row sum to 1, and are in the range 0 to 1. Remember, each row represents a single token and the attention it should pay to all tokens that came before it.
* Then, matrix multiply the scores matrix with the **V vectors** of each token you're attending to, this gives you a vector of weighted changes to apply to the tokens doing the attending. You do this for all heads, then **concatenate** the outputs into one final vector. 
        * In GPT-2, each head vector is size 64. Since there's 12 heads, when you concatenate them you end up with a 1-D array of size 768. (You're back to the original embedding vector shape!)
* Pass the output vector through one final layer where you multiply by **Wo**, the **Output Projection Layer**. This final matrix multiply serves to "mix" the data processed by each head together.
* Finally, this output vector is then added to the original embedding vector. 
* Now you have a new embedding vector with context baked in, this can now flow through the MLP/FFN layer.

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

