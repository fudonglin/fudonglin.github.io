# Similarity Is All You Need: A Step-by-Step Guide to Understanding Transformers

## Preface

The first time I read the Transformer paper [1] was in the spring of 2021, during the first year of my Ph.D. study. 
At the time, it was difficult for me to fully understand this paper. 
As I progressed through my Ph.D., I revisited this paper several times, and each reading revealed new insights into its core ideas. 
Today, Artificial Intelligence (AI)—particularly Large Language Models like ChatGPT—is deeply integrated into nearly every aspect of our lives. 
If you're interested in AI and want to build a solid foundation, I highly recommend starting with this paper.



## Machine Translation 

Let's begin our journey with a machine translation task (translating from Chinese to English):

| Source Language (Chinese) | Target Language (English) |
| ------------------------- |---------------------------|
| 纽约是一座城市            | New York is a city        |



In this blog, we explore machine translation using an Autoregressive (AR) model. Let $\mathbf{x}$ and $\textbf{y}$ denote the source and target language sequences, respectively. 
Given a target sentence of length $T$, we use $y\_{t}$ to represent the word at position $t$, and $\mathbf{y}\_{<t}$ to denote all preceding words. 
The AR model generates the translation one word at a time, predicting $y\_{t}$ based on the source input $\mathbf{x}$ and the full semantic context provided by $\mathbf{y}\_{<t}$:

$$
\max_{\mathbf{\theta}} ~\log p_{\mathbf{\theta}}(y | \mathbf{x}) = \sum_{t=1}^{T} \log p_{\mathbf{\theta}} (y_{t} | y_{\lt t}, \mathbf{x}),
$$




where $\mathbf{\theta}$ denotes the parameters to be learned. 
For example, the AR model predicts the word `"city"` based on the context of  `"New York is a"` and `"纽约是一座城市"`, i.e., $p\_{\mathbf{\theta}} (\textrm{"city"} ~|~ \textrm{"New York is a"}, ~\text{"纽约是一座城市"})$.



## Recurrent Neural Networks (RNNs) for Machine Translation 

For machine translation, the encoder-decoder architecture is a widely adopted framework for building the Autoregressive (AR) model. 
Before the emergence of Transformers, both the encoder and decoder were commonly implemented using Recurrent Neural Networks (RNNs). 

Here, the encoder RNN processes a variable-length input sequence from the source language and encodes it into a fixed-size hidden state, which serves as a summary of the input. 
This final hidden state is then passed to the decoder at every decoding step to condition the output on the source sequence. 
The decoder RNN generates the target-language output token by token, relying on both the encoded input and the previously generated tokens. 

During training, this process is guided by the ground truth target tokens (i.e., teacher forcing), while during inference, it generates tokens sequentially, conditioning each prediction on its own prior outputs.


![rnn_encoder_decoder](https://github.com/fudonglin/fudonglin.github.io/blob/main/_posts/2025-07-04-transformer/rnn_encoder_decoder.png?raw=true)

Figure 1: RNN-based Sequence-to-Sequence Learning for Machine Translation.


Figure 1 illustrates how the encoder-decoder architecture is used for machine translation. Special tokens such as `<bos>` (beginning of sequence) and `<eos>` (end of sequence) are used to mark the start and end of the target sequence, respectively. 

However, RNN-based encoder-decoder architectures suffer from two major limitations:

- First, the sequential nature of RNNs prevents parallelization during training and inference, making them computationally inefficient.

- Second, RNNs struggle to capture long-range dependencies in sequences. For example, in the sentence `"New York is a city"`, the prediction of the word `"city"` relies on understanding the earlier phrase `"New York"`. Since RNNs process tokens step by step, they often fail to retain information from distant parts of the sequence, leading to degraded performance on longer inputs.

  

> If you're interested in learning more about the RNN-based encoder-decoder architecture for machine translation, please read the reference [2]: [Sequence-to-Sequence Learning for Machine Translation](https://www.d2l.ai/chapter_recurrent-modern/seq2seq.html#encoderdecoder-for-sequence-to-sequence-learning).



## Transformers


The Transformer architecture addresses two key limitations of the RNN-based encoder-decoder model: (i) limited parallelization; and (ii) inability to capture long-range dependencies. 

![Transformers](https://github.com/fudonglin/fudonglin.github.io/blob/main/_posts/2025-07-04-transformer/transformers.png?raw=true)

Figure 2: Illustration of the Transformer architecture for machine translation.


Like RNNs, Transformers also employ an encoder-decoder structure for machine translation. 
Figure 2 shows the model architecture of Transformers. 
Specifically, the encoder uses Multi-Head Self-Attention on the source language to learn contextualized embeddings. The decoder performs two types of Multi-Head Attention mechanisms:

(i) **Masked Multi-Head Self-Attention** on the target language to preserve the autoregressive property, and

(ii) **Multi-Head Cross-Attention** between the target and source languages to incorporate information from the source input.



Now, let's look at an example to illustrate the elegant design within the Transformer architecture. 
Suppose we want to predict the second English word, i.e., `"York"`. 
According to the Autoregressive (AR) model, the Transformer can only attend to the source sentence, i.e., `"纽约是一座城市<bos>"`, and the preceding tokens in the target sentence, i.e., `"<bos> New"`. 
Therefore, the AR model predicts the word `"York"` by estimating the following conditional probability:

$$
p_{\mathbf{\theta}} (\textrm{"York"}| \textrm{"<bos> New <mask> <mask> <mask> <mask>"}, \textrm{"纽约是一座城市 <bos>"}).
$$

Here, the purpose of Masked Multi-Head Attention in the decoder is to prevent the model from accessing future tokens in the target sequence. This raises an important question: 

**Do we need to predict each word sequentially? For example, first "New", then "York", followed by "is", and so on?** 

The answer is  **NO** during training, but **YES** during inference.



During training, since the ground-truth target tokens are available, Transformers leverage Masked Multi-Head Attention to enable the autoregressive property. 
The entire target sequence is fed into the decoder at once, but attention masks are used to ensure that each position can only attend to previous tokens, i.e., mimicking autoregressive behavior. 
This allows the model to predict all tokens in parallel while preserving the correct dependency structure. 
The following example illustrates how Masked Multi-Head Attention enables the simultaneous prediction of tokens at different positions:

$$
\begin{gathered}
  \text{"<bos> <mask> <mask> <mask> <mask> <mask>"} \rightarrow \text{"New"} \\
  \text{"<bos> New <mask> <mask> <mask> <mask>"} \rightarrow \text{"York"} \\
  \cdots \\
  \text{"<bos> New York is a <mask>"} \rightarrow \text{"city"} \\
  \text{"<bos> New York is a city"} \rightarrow \text{"<eos>"}.
\end{gathered}
$$


During inference, since the ground-truth tokens are not available, the Transformer must generate the target sequence one token at a time, predicting each word sequentially. 
This sequential decoding is the reason why models like ChatGPT—built on decoder-only Transformer architectures—can be relatively slow during inference.



### Attention

We have seen how Masked Multi-Head Attention effectively enables the autoregressive property, thereby significantly enhancing parallelization—a key limitation of RNNs. 
Now, let’s take a closer look at Multi-Head Attention itself and explore how it helps overcome the long-range dependency issue. 
This challenge arises in RNNs because the model relies solely on the last hidden state to encode all preceding words.

#### Tokenization

Before discussing Multi-Head Attention, we first need to understand tokenization. 

**Why do we need a Tokenizer in the language model?** The reason is that computers do not understand human language—they only process binary signals. 
Therefore, we must convert human language into numerical representations that models can understand. 
In simple terms, a tokenizer functions like a dictionary: each token (usually a word or subword) is mapped to a unique index and a learnable embedding:

| Token  | Index |              Embedding               |
| :----: | :---: | :----------------------------------: |
| <bos>  |   0   | [-0.9200, -0.1600, -0.3100, -0.1000] |
|  New   |   1   | [ 0.4800, -1.2000, -0.4400,  2.2100] |
|  York  |   2   | [ 0.5600, -0.3500, -1.0900, -0.4300] |
|   is   |   3   | [ 1.5700,  0.5600,  0.4800, -1.0000] |
|   a    |   4   | [-0.8300,  0.0400,  0.4900, -1.0700] |
| <mask> |   5   | [ 1.2800, -1.3300,  0.6100, -1.2800] |
|  city  |   6   | [ 0.6500, -0.6000, -0.8700,  0.5900] |
| <eos>  |   7   | [ 1.0600,  1.2400, -0.7100,  0.8100] |



Figure 3 illustrates how the tokenizer operates within the Transformer architecture. Given an input sequence, the tokenizer first converts each token into a unique index and retrieves the corresponding embedding. These embeddings are then fed into the Transformer model to predict the next tokens. The output of the Transformer—namely, the predicted token embeddings—is mapped back to their respective indices and decoded into human-readable words.


![tokenization](https://github.com/fudonglin/fudonglin.github.io/blob/main/_posts/2025-07-04-transformer/tokenization.png?raw=true)

Figure 3: The End-to-End Workflow of Tokenizer.	



Personally, I believe the tokenizer plays a critical role in the recent success of Large Language Models (LLMs), as it effectively bridges the gap between human language and computer-readable input. 
For example, while vision models tend to generate unrealistic or distorted images, language models are much less likely to produce non-existent words—thanks in large part to robust tokenization. 

In the field of computer vision, we are still awaiting our own "ChatGPT moment". 
Perhaps the breakthrough lies in developing vision tokenizers capable of bridging the gap between visual data and machine understanding. It’s important to note, however, that language tokenization operates over a discrete and finite vocabulary (e.g., approximately 30,000 commonly used English tokens), whereas vision tokenization must grapple with a continuous and effectively infinite input space if we aim to model the visual world in its entirety. 
With that being said, building vision tokenizers is much more challenging than their language counterparts.



### Single-Head Attention

![image-20250614230247926](https://github.com/fudonglin/fudonglin.github.io/blob/main/_posts/2025-07-04-transformer/attention.png?raw=true)

Figure 4: The Single-Head and Multi-Head Attention methods.




Next, let’s uncover the Attention—the core idea in the Transformer—step by step. We begin by introducing the equation of the Attention mechanism:

$$
\text{Attention}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{softmax}\left(\frac{\mathbf{QK}^{T}}{\sqrt{d_k}}\right)\mathbf{V}.
$$

Here, $\mathbf{Q} = \mathbf{M}\_{q} \cdot \mathbf{X} $, $ \mathbf{K} = \mathbf{M}\_{k} \cdot \mathbf{X} $, and $ \mathbf{V} = \mathbf{M}\_{v} \cdot \mathbf{X} $ are the **query**, **key**, and **value** matrices, respectively. 
These are obtained by applying learned linear projections—$ \mathbf{M}\_{q} $, $ \mathbf{M}\_{k} $, and $\mathbf{M}\_{v} $—to the input token embeddings $\mathbf{X}$.



In essence, **attention computes a weighted sum over all tokens in the input sequence, assigning higher weights to those that are more relevant (or similar) to the query**. We can rewrite the attention equation more intuitively as:

$$
\text{Attention}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \mathbf{W} \mathbf{V}, \quad \text{where} ~\mathbf{W} = \text{softmax}\left(\frac{\mathbf{QK}^{T}}{\sqrt{d_k}}\right).
$$

Here, $ \mathbf{QK}^T $ computes the similarity scores between queries and keys, effectively capturing the relevance between tokens.



Returning to our machine translation example, suppose the model is trying to predict the word `"city"` in the following conditional probability:

$$
p_{\mathbf{\theta}}(\text{"city"} \mid \text{"<bos> New York is a <mask>"}).
$$

For simplicity, we exclude the conditioning on the source language and restrict our discussion to Masked Self-Attention. 
Then, we have:

$$
\begin{gathered}
\text{"<bos> New York is a <mask>"} \\
\downarrow \\
\mathbf{X} = [x_0, x_1, x_2, x_3, x_4, x_5] \\
\mathbf{M}_{v} \cdot \mathbf{X} ~\downarrow \\
\mathbf{V} = [v_0, v_1, v_2, v_3, v_4, v_5].
\end{gathered}
$$


​									

The output embedding at position 5 is computed as:

$$
\begin{gathered}
y_5 = [w_0^5, w_1^5, \dots, w_5^5] 
\begin{bmatrix}
v_0 \\
v_1 \\
\vdots \\
v_5
\end{bmatrix}.
\end{gathered}
$$

Since `"city"` is semantically related to `"New York"`, we expect the model to assign higher attention weights to $ w\_2^5 $ and $ w\_3^5 $, which correspond to the tokens `"New"` and `"York"`.

The cross-attention operates in a similar way. For example, consider the case where the model aims to predict the word `"city"` under the following condition:

$$
p_{\mathbf{\theta}}(\text{"city"} \mid \text{"<bos> New York is a <mask>"}, \text{"纽约是一座城市<eos>"}).
$$

Ideally, the model should assign higher attention weights to the tokens `"New"` and `"York"`, as well as the corresponding source-language tokens `"城"` and `"市"`.



#### The Scaling Term in Attention

Now, we have a solid understanding of how Attention works. 
But one important question remains: 

**What is the role of the scaling term** $ \sqrt{d\_{k}} $ **in the Attention mechanism?**

The purpose of dividing by $ \sqrt{d\_{k}} $ is to produce a smoother distribution of attention weights. 
Without this scaling term, the dot products in $ QK^T $ can become large in magnitude when the dimensionality $ d\_{k} $ is high, leading the softmax function to produce very sharp distributions that overly emphasize a few tokens. 

As illustrated below, omitting $ \sqrt{d\_{k}} $ can cause the model to assign nearly all attention to a small number of tokens, while including it results in a more balanced attention distribution:

$$
\begin{gathered}
\mathbf{w} = [10^{-6}, 10^{-3}, 0.99,  10^{-6}, \dots, 10^{-6}] \quad \text{(without } \sqrt{d_k} \text{)}, \\
\mathbf{w} = [0.01, 0.1, 0.85, \dots, 0.01] \quad \text{(with } \sqrt{d_k} \text{)}.
\end{gathered}
$$

> Note: The scaling term $\sqrt{d_k}$ refers to the dimensionality of the keys and should not be confused with the model's hidden dimension, denoted as $d_{\text{model}}$.



### Multi-Head Attention

In the paper, the authors found that optimizing Transformers across multiple smaller hidden spaces can lead to better generalization and efficiency compared to relying on a single large unified space. 
Driven by this observation, the authors propose the Multi-Head Attention, expressed as below:

$$
\begin{gathered}
\text{MultiHead}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{Concat}(\text{head}_1, \dots, \text{head}_h) \mathbf{W}^O \\
\text{where} \quad \text{head}_i = \text{Attention}(\mathbf{Q} \mathbf{W}_i^Q, \mathbf{K} \mathbf{W}_i^K, \mathbf{V} \mathbf{W}_i^V) \quad
\text{and} \quad \text{Attention}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{softmax}\left( \frac{\mathbf{Q} \mathbf{K}^T}{\sqrt{d_k}} \right) \mathbf{V},
\end{gathered}
$$

where $i$ denote the $i$-th attention head, and $\mathbf{W}^{O}$ is a learnable matrix.

Let's see an example for better understanding the Multi-Head Attention. 
Assume the hidden dimension $d\_{\textrm{model}}$ is 512, and the number of attention heads is 8. 
Then, Multi-Head Attention splits the hidden dimension into 8 subspaces, each with a dimensionality of $d\_{k} = 64$. Within each subspace, a separate Single-Head Attention operation is performed. 
The outputs from all heads are then concatenated and projected using the weight matrix $\mathbf{W}^{O}$ to produce the final output embedding.

Despite effectiveness, the authors did not offer a clear explanation for why Multi-Head Attention outperforms Single-Head Attention when both share the same hidden dimension size.
Based on my previous experiments with various Vision Transformers (ViTs) [3], this phenomenon may be partly explained. 
I found that activating only the top 75% most responsive parameters resulted in just a 0.1% to 0.4% drop in performance on the ImageNet benchmark across multiple ViT variants. 
This suggests that the remaining 25% of parameters have not been utilized and contribute minimally to the final performance. 
One possible reason why Multi-Head Attention is more effective is that **dividing a large hidden dimension into smaller subspaces can reduce the number of inactive or underutilized parameters**, thereby enhancing the effectiveness and generalization of Transformers.



### Positional Embedding

Unlike RNNs or CNNs, the **Transformer architecture is order-agnostic**:

- Its self-attention mechanism processes all tokens simultaneously, with no inherent sense of sequence order.
- While this enables excellent parallelization, it also means the model lacks a built-in understanding of token positions.

This limitation arises because attention is essentially a weighted sum over all tokens in the input sequence. As a result, the model requires **explicit positional information** to capture the structure of the input—for example:

- "New York is a city"  $\neq$  "City is a New York";   
- Or to recognize whether one word comes before or after another.

Currently, there are two main types of positional embeddings: **fixed (non-learnable) and learnable embeddings**. 
In the paper, the authors used fixed sinusoidal positional embeddings. However, learnable positional embeddings have become increasingly popular in recent models. For example, the ViT architecture adopts learnable embeddings to improve model flexibility and performance.



The authors represent the positional information using  a combination of sine and cosine functions at varying frequencies:

$$
\begin{gathered}
PE(pos, 2i) = \sin(\frac{pos}{10000^{2i/d_{\textrm{model}}}}), \\
PE(pos, 2i+1) = \cos(\frac{pos}{10000^{2i/d_{\textrm{model}}}}).
\end{gathered}
$$

Here, $pos$ denotes the position of a word (or token) in the sequence, $d\_{\textrm{model}}$ is the dimensionality of the model's embeddings (e.g., $d_{\textrm{model}} = 512$ used in the paper), and $i \in \{0, 1, 2, \dots, d_{\textrm{model}} - 1 \}$ indexes the embedding dimensions. 
The goal of positional embedding is to transform a one-dimensional position index into a high-dimensional embedding representation, enabling it to be directly added to the token embedding. 
Note that the positional representation should satisfy the following criteria:

- **Uniqueness**: Each index (i.e., word position in a sentence) should be mapped to a unique embedding.
- **Consistency**: The relative distance between any two positions should remain consistent, regardless of the total sequence length.
- **Generalization**: The embedding scheme should generalize to longer sequences beyond those seen during training, without requiring retraining or architectural changes. Additionally, the embedding values should remain bounded to ensure numerical stability.
- **Determinism**: The positional encoding must be deterministic, producing the same output for the same input every time.

> If you're interested in how the sine-cosine positional encoding satisfies the above criteria, please refer to references [4] and [5] for detailed explanations. 



When I first encountered the positional embedding, I was confused by the roles of `pos` and `i` in the equation. 
Let’s walk through an example to clarify. 
Given the sentence `"New York is a city"`, suppose we want to compute the positional embedding for the word `"city"`. 
Since `"city"` is the fourth word in the sequence (with zero-based indexing), its position index is 3. We set $\text{pos} = 3$ and $d_{\text{model}} = 512$. 

According to the sinusoidal positional encoding scheme, the values at even-numbered dimensions are computed using the sine function, while the values at odd-numbered dimensions use the cosine function. 
This maps the position index 3 to a unique 512-dimensional positional embedding.



If you're still unsure how positional embeddings work, try running the following code:

```python
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)  # (max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)  # (max_len, 1)

        # Compute the division term, i.e., \frac{10000}{-2i * d_{model}}
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))  # (d_model/2,)

        # Fill even-numbered dimension with sine values
        pe[:, 0::2] = torch.sin(position * div_term)
        # Fill odd-numbered dimension with cosine values
        pe[:, 1::2] = torch.cos(position * div_term)

        self.pe = pe.unsqueeze(0)  # (1, max_len, d_model)

    def forward(self, x):
        pos_embed = self.pe[:, :x.size(1), :].to(x.device)
        x = x + pos_embed
        return x
```



## Takeaways

- The Transformer was originally proposed as a replacement for RNNs in sequence-to-sequence machine translation.
- At its core, the Attention mechanism computes a weighted sum over the entire input sequence, assigning higher weights to more relevant tokens. This allows the model to effectively capture long-range dependencies, a limitation inherent in RNNs.
- Masked Multi-Head Attention effectively models the autoregressive (AR) property while allowing parallelization during training.
- Subsequent Transformer models have adopted either encoder-only architectures, such as BERT, or decoder-only architectures, like GPT. Encoder-only models benefit from bidirectional context, meaning they can attend to both preceding and succeeding tokens, which is ideal for understanding tasks. However, they assume that each masked token is predicted independently of the others. In contrast, decoder-only models generate text autoregressively, conditioning each token only on the preceding ones—this enables coherent text generation but limits the model to unidirectional context.



## Reference

[1] Attention Is All You Need, NeurIPS 2017

[2] [Sequence-to-Sequence Learning for Machine Translation](https://www.d2l.ai/chapter_recurrent-modern/seq2seq.html#encoderdecoder-for-sequence-to-sequence-learning)

[3] An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale, ICLR 2021

[4] [Transformer Architecture: The Positional Encoding](https://kazemnejad.com/blog/transformer_architecture_positional_encoding)

[5] [Understanding Positional Encoding in Transformers](https://erdem.pl/2021/05/understanding-positional-encoding-in-transformers)

[6] BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding

[7] Language Models are Unsupervised Multitask Learners
