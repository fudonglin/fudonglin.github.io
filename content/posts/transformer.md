+++
date = '2025-03-29T00:34:34-07:00'
draft = false
title = 'Similarity Is All You Need: A Step-by-Step Guide to Understanding Transformers'

+++

## Preface

The first time I read the Transformer paper [1] was in the spring of 2021, during the first year of my Ph.D. studies. At the time, I found the paper difficult to fully understand. However, as I progressed through my Ph.D., I revisited it several times, and each reading revealed new insights into its core ideas. Today, Artificial Intelligence (AI)—particularly Large Language Models like ChatGPT—is deeply integrated into nearly every aspect of our lives. If you're interested in AI and want to build a solid foundation, I highly recommend starting with this paper.



## Machine Translation 

Let's begin our journey with a machine translation task (translating from Chinese to English):

| Source Language (Chinese) | Target Lanuage (English) |
| ------------------------- | ------------------------ |
| 纽约是一座城市            | New York is a city       |



In this blog, we explore machine translation using an Autoregressive (AR) model. Let $\mathbf{x}$ and $\mathbf{y}$ denote the source and target language sequences, respectively. Given a target sentence of length $T$, we use $\mathbf{y}_{t}$ to represent the word at position $t$, and $\mathbf{y}_{<t}$ to denote all preceding words. The AR model generates the translation one word at a time, predicting $\mathbf{y}_t$ based on the source input $\mathbf{x}$ and the full semantic context provided by $\mathbf{y}_{<t}$:
$$
\max_{\boldsymbol{\theta}} ~\log p_{\mathbf{\theta}}(\mathbf{y} | \mathbf{x}) = \sum_{t=1}^{T} \log p_{\mathbf{\theta}}(\mathbf{y}_{t} | \mathbf{y}_{<t}, \mathbf{x}),
$$
where $\boldsymbol{\theta}$ denotes the parameters to be learned. For example, the AR model predicts the word `city` based on the context of  `New York is a` and `纽约是一座城市`, i.e., $p_{\boldsymbol{\theta}} (\text{``city"}| \text{``New York is a"}, \text{``纽约是一座城市''})$.



## Recurrent Neural Networks (RNNs) for Machine Translation 

For machine translation, the encoder-decoder architecture is a widely adopted framework for building the Autoregressive (AR) model. Before the emergence of Transformers, both the encoder and decoder were commonly implemented using Recurrent Neural Networks (RNNs). Here, the encoder RNN processes a variable-length input sequence from the source language and encodes it into a fixed-size hidden state, which serves as a summary of the input. This final hidden state is then passed to the decoder at every decoding step to condition the output on the source sequence. The decoder RNN generates the target-language output token by token, relying on both the encoded input and the previously generated tokens. During training, it is guided by the ground truth target tokens (teacher forcing), while during inference, it generates tokens sequentially, conditioning each prediction on its own prior outputs.

![rnn_encoder_decoder](https://github.com/fudonglin/fudonglin.github.io/blob/main/content/images/rnn_encoder_decoder.png)

​								Figure 1: RNN-based Sequence-to-Sequence Learning for Machine Translation.

Figure 1 illustrates how the encoder-decoder architecture is used for machine translation. Special tokens such as `<bos>` (beginning of sequence) and `<eos>` (end of sequence) are used to mark the start and end of the target sequence, respectively. 

However, RNN-based encoder-decoder architectures suffer from two major limitations:

- First, the sequential nature of RNNs prevents parallelization during training and inference, making them computationally inefficient.

- Second, RNNs struggle to capture long-range dependencies in sequences. For example, in the sentence `"New York is a city"`, the prediction of the word `"city"` relies on understanding the earlier phrase `"New York"`. Since RNNs process tokens step by step, they often fail to retain information from distant parts of the sequence, leading to degraded performance on longer inputs.

  

> If you're interested in learning more about the RNN-based encoder-decoder architecture for machine translation, I recommend reading the reference [2]: [Sequence-to-Sequence Learning for Machine Translation](https://www.d2l.ai/chapter_recurrent-modern/seq2seq.html#encoderdecoder-for-sequence-to-sequence-learning).



## Transformers

![Transformers](./../images/transformers.png)

​								Figure 2: Illustrations of the Transformer architecture for machine translation.



The Transformer architecture addresses two key limitations of the RNN-based encoder-decoder model: (i) limited parallelization, and (ii) difficulty in capturing long-range dependencies. Like RNNs, Transformers also employ an encoder-decoder structure for machine translation. Figure 2 show the Transformer architecture for machine translation. Specifically, the encoder uses Multi-Head Self-Attention on the source language to learn contextualized embeddings. The decoder performs two types of Multi-Head Attention mechanisms:

(i) **Masked Multi-Head Self-Attention** on the target language to preserve the autoregressive property, and

(ii ) **Cross-Attention** between the target and source languages to incorporate information from the source input.



Now, let's look at an example to illustrate the elegant design within the Transformer architecture. Suppose we want to predict the second English word, "York". According to the Autoregressive (AR) model, the Transformer can only attend to the source sentence — "纽约是一座城市<bos>" — and the preceding tokens in the target sentence — "<bos> New". Therefore, the model estimates the following conditional probability:
$$
p_{\boldsymbol{\theta}} (\text{``York"}| \text{``<bos> New <mask> <mask> <mask> <mask>"}, \text{``纽约是一座城市 <bos>''}).
$$
Here, the purpose of Masked Multi-Head Attention in the decoder is to prevent the model from accessing future tokens in the target sequence. This raises an important question: **do we need to predict each word sequentially, as in RNN-based machine translation—for example, first "New," then "York," followed by "is," and so on?** The answer is  **NO** during training, but **YES** during inference.



During training, since the ground-truth target tokens are available, Transformers leverage Masked Multi-Head Attention to enable parallelization. The entire target sequence is fed into the decoder at once, but attention masks are used to ensure that each position can only attend to previous tokens—mimicking autoregressive behavior. This allows the model to predict all tokens in parallel while preserving the correct dependency structure. The following example illustrates how Masked Multi-Head Attention enables the simultaneous prediction of tokens at different positions:
$$
\text{``<bos> <mask> <mask> <mask> <mask> <mask>"} \rightarrow \text{``New''} \\
\text{``<bos> New <mask> <mask> <mask> <mask>"} \rightarrow \text{``York''} \\
\cdots \\
\text{``<bos> New York is a <mask>"} \rightarrow \text{``city''} \\
\text{``<bos> New York is a city"} \rightarrow \text{``<eos>''}.
$$


During inference, since the ground-truth tokens are not available, the Transformer must generate the target sequence one token at a time, predicting each word sequentially. This sequential decoding is the reason why models like ChatGPT—built on decoder-only Transformer architectures—can be relatively slow during inference.



### Attention

We have seen how Masked Multi-Head Attention enables parallelization, effectively addressing one of the key limitations of RNNs. Now, let’s take a closer look at Multi-Head Attention itself and explore how it helps overcome the long-range dependency issue. This challenge arises in RNNs because the model relies solely on the final hidden state to encode the entire input sequence.

#### Tokenization

![tokenizer](./../images/tokenizer.png)

​													Figure 3: Example of Tokenizer.

Before discussing Multi-Head Attention, we first need to understand tokenization. **Why do we need a Tokenizer?** The reason is that computers do not understand human language—they only process binary signals. Therefore, we must convert human language into numerical representations that models can understand. In simple terms, a Tokenizer functions like a dictionary: each token (usually a word or subword) is mapped to a unique index and a learnable embedding. In practice, we often use pre-trained Tokenizers, such as those available from Hugging Face.





![tokenization](./../images/tokenization.png)

​												Figure 4: The End-to-End Workflow of Tokenizer.	



Here’s how the tokenizer operates within the Transformer architecture. Given an input sequence, the tokenizer first converts each token into a unique index and retrieves the corresponding embedding. These embeddings are then fed into the Transformer model to predict the next tokens. The output of the Transformer—namely, the predicted token embeddings—is mapped back to their respective indices and decoded into human-readable words.



Personally, I believe that the Tokenizer plays a critical role in the recent success of large language models (LLMs), as it effectively bridges the gap between human language and computer-readable input. For example, while vision models tend to generate unrealistic or distorted images, language models are much less likely to produce non-existent words—thanks in large part to robust tokenization. 

In the field of computer vision, we are still awaiting our own “ChatGPT moment.” Perhaps the breakthrough lies in developing visionary Tokenizers capable of bridging the gap between visual data and machine understanding. It’s important to note, however, that language tokenization operates over a discrete and finite vocabulary (e.g., approximately 30,000 commonly used English tokens), whereas vision tokenization must grapple with a continuous and effectively infinite input space if we aim to model the visual world in its entirety. With that being said, building vision tokenizers is significantly more challenging than their language counterparts.



### Single-Head Attention

![image-20250614230247926](./../images/attention.png)

​									Figure 5: The Single-Head and Multi-Head Attenion method.



Everyone talks about attention—but do we truly understand how it works? While many of us might, if you're new to machine learning, let’s uncover its inner workings step by step. We begin by introducing the core equation of the attention mechanism:
$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^{T}}{\sqrt{d_k}}\right)V.
$$

Here, $Q = M_q X $, $ K = M_k X $, and $ V = M_v X $ are the **query**, **key**, and **value** matrices, respectively. These are obtained by applying learned linear projections—$ M_q $, $ M_k $, and $M_v $—to the input token embeddings $X $.



In essence, **attention computes a weighted sum over all tokens in the input sequence, assigning higher weights to those that are more relevant (or similar) to the query**. We can rewrite the attention equation more intuitively as:

$$
\text{Attention}(Q, K, V) = W \cdot V, \quad \text{where} \quad W = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)
$$

Here, $ QK^T $ computes the similarity scores between queries and keys, effectively capturing the relevance between tokens.



Returning to our machine translation example, suppose the model is trying to predict the word *"city"* in the following conditional probability:
$$
p_{\boldsymbol{\theta}}(\text{``city''} \mid \text{``<bos> New York is a <mask>''}).
$$

For simplicity, we exclude the conditioning on the source language and restrict our discussion to Masked Self-Attention. Then, we have:
$$
\text{``<bos> New York is a <mask>''} \\
\downarrow \\
X = [x_0, x_1, x_2, x_3, x_4, x_5] \\
\downarrow \\
V = [v_0, v_1, v_2, v_3, v_4, v_5]
$$


​									

The output embedding at position 5 is computed as:
$$
y_5 = [w_0^5, w_1^5, \dots, w_5^5] 
\begin{bmatrix}
v_0 \\
v_1 \\
\vdots \\
v_5
\end{bmatrix}
$$

Since *"city"* is semantically related to *"New York"*, we expect the model to assign higher attention weights to $ w_2^5 $ and $ w_3^5 $, which correspond to the tokens *"New"* and *"York"*.

Cross-attention operates in a similar way. For example, consider the case where the model aims to predict the word *"city"* under the following condition:

$$
p_{\boldsymbol{\theta}}(\text{``city''} \mid \text{``<bos> New York is a <mask>''}, \text{``纽约是一座城市<eos>''}).
$$

Ideally, the model should assign higher attention weights to the tokens *"New"*, *"York"*, as well as the corresponding source-language tokens *"城"*, and *"市"*



#### The Scaling Term $\sqrt{d_{k}}$

If you’ve followed along this far—congratulations! You now have a solid understanding of how Attention works. But one important question remains: **what is the role of** $ \sqrt{d_k} $ **in the Attention mechanism?**

The purpose of dividing by $ \sqrt{d_k} $ is to produce a **smoother distribution of attention weights**. Without this scaling term, the dot products in $ QK^T $ can become large in magnitude when the dimensionality $ d_k $ is high, leading the softmax function to produce very sharp distributions that overly emphasize a few tokens. As illustrated below, omitting $ \sqrt{d_k} $ can cause the model to assign nearly all attention to a small number of tokens, while including it results in a more balanced attention distribution:

$$
w = [10^{-6}, 10^{-3}, 0.99,  10^{-6}, \dots, 10^{-6}] \quad \text{(without } \sqrt{d_k} \text{)}, \\\\
w = [0.01, 0.1, 0.85, \dots, 0.01] \quad \text{(with } \sqrt{d_k} \text{)}.
$$

> Note: The scaling term $\sqrt{d_k}$ refers to the dimensionality of the keys and should not be confused with the model's hidden dimension, denoted as $d_{\text{model}}$, which appears in the context of multi-head attention.



### Multi-Head Attention

The key idea of Multi-Head Attention is to allow the model to attend to information from different representation subspaces at different positions in parallel:
$$
\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \dots, \text{head}_h) W^O \\
\text{where} \quad \text{head}_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V) \quad
\text{and} \quad \text{Attention}(Q, K, V) = \text{softmax}\left( \frac{QK^T}{\sqrt{d_k}} \right) V
$$
In the original paper, the authors found that optimizing Transformers across multiple smaller hidden spaces can lead to better generalization and efficiency compared to relying on a single large unified space. However, they did not offer a clear explanation for why Multi-Head Attention outperforms Single-Head Attention when both share the same total hidden dimension.

Based on my previous experiments with various Vision Transformers (ViTs) [3], this phenomenon may be partly explained. I found that activating only the top 75% most responsive parameters resulted in just a 0.1% to 0.4% drop in performance on the ImageNet benchmark across multiple ViT variants. This suggests that the remaining 25% of parameters contribute minimally to the final output. One possible reason why Multi-Head Attention is more effective is that dividing a large hidden dimension into smaller subspaces can reduce the number of inactive or underutilized parameters, thereby enhancing the effective utilization of neurons.



### Positional Embedding

Unlike RNNs or CNNs, the **Transformer architecture is order-agnostic**:

- Its self-attention mechanism processes all tokens simultaneously, with no inherent sense of sequence order.
- While this enables excellent parallelization, it also means the model lacks a built-in understanding of token positions.

This limitation arises because attention is essentially a weighted sum over all tokens in the input sequence. As a result, the model requires **explicit positional information** to capture the structure of the input—for example:

- “New York is a city”  $\neq$  “City is a New York”.   
- Or to recognize whether one word comes before or after another.

Currently, there are two main types of positional embeddings: fixed (non-learnable) and learnable embeddings. In the original paper, the authors used fixed sinusoidal positional embeddings. However, learnable positional embeddings have become increasingly popular in recent models. For example, the ViT architecture adopts learnable embeddings to improve model flexibility and performance.



In the original Transformer paper, the authors represent the positional information using  a combination of sine and cosine functions at varying frequencies:
$$
PE(pos, 2i) = \sin(\frac{pos}{10000^{2i/d_{model}}}) \\
PE(pos, 2i+1) = \cos(\frac{pos}{10000^{2i/d_{model}}})
$$
Here, $pos$ denotes the position of a word (or token) in the sequence, $d_{model}$ is the dimensionality of the model's embeddings (e.g., $d_{model} = 512$ used in the original paper), and $i \in \{0, 1, 2, \dots, d_{model} - 1 \}$ indexes the embedding dimensions. The goal of positional embedding is to transform a one-dimensional position index into a high-dimensional vector representation, enabling it to be directly added to the token embedding. Note that, the positional representation should satisfy the following criteria:

- **Uniqueness**: Each index (i.e., word position in a sentence) should be mapped to a unique embedding.
- **Consistency**: The relative distance between any two positions should remain consistent, regardless of the total sequence length.
- **Generalization**: The embedding scheme should generalize to longer sequences beyond those seen during training, without requiring retraining or architectural changes. Additionally, the embedding values should remain bounded to ensure numerical stability.
- **Determinism**: The positional encoding must be deterministic, producing the same output for the same input every time.

If you're interested in how the sine-cosine positional encoding satisfies the above criteria, please refer to references [4] and [5] for detailed explanations. 



In this blog, we’ll focus on implementing positional embeddings. When I first encountered the concept, I was confused by the roles of `pos` and `i` in the formula. Let’s walk through an example to clarify. Consider the sentence *"New York is a city"*. Suppose we want to compute the positional embedding for the word *"city"*. Since *"city"* is the fourth word in the sequence (with zero-based indexing), its position index is 3. We set $\text{pos} = 3$ and $d_{\text{model}} = 512$. 

According to the sinusoidal positional encoding scheme, the values at even-numbered dimensions are computed using the sine function, while the values at odd-numbered dimensions use the cosine function. This maps the position index 3 to a unique 512-dimensional embedding vector.



If you're still unsure how positional embeddings work, try running the following code snippet:

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

[1] Vaswni et. al., Attention Is All You Need, NeurIPS 2017

[2] [Sequence-to-Sequence Learning for Machine Translation](https://www.d2l.ai/chapter_recurrent-modern/seq2seq.html#encoderdecoder-for-sequence-to-sequence-learning)

[3] An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale, ICLR 2021

[4] [Transformer Architecture: The Positional Encoding](https://kazemnejad.com/blog/transformer_architecture_positional_encoding)

[5] [Understanding Positional Encoding in Transformers](https://erdem.pl/2021/05/understanding-positional-encoding-in-transformers)

[6] BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding

[7] Language Models are Unsupervised Multitask Learners
