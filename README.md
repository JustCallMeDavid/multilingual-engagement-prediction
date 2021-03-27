# multilingual-engagement-prediction

This work was carried out in the context of the 2020 ACM Recommender Systems challenge during my master's studies.

## Motivation

Engagement prediction on Tweets has to be carried out in a real-time manner, which creates the need for efficient architectures. In particular text, due to recend advances in deep learning methodology for NLP, is expensive to process in a prediction pipeline. Therefore, we analyze the benefit of textual information, i.e., the Tweet content, in generating engagement probabilities using a variety of state-of-the-art as well as specially developed deep learning models.

## Instructions

Download the data files from <https://www.recsyschallenge.com/2020/> (they should be available even after the end of the challenge).  You may need to register and be approved for this.

## Models

### State-of-the-art Models

We employ BERT, RoBERTa, DistilBERT and LASER.

### Multibpemb Based Models

#### Multibpemb-Conv

Based on the work of [Kim, 2014], we develop a deep convolutional neural network model that operates on top of multilingual sub-word embeddings.

#### Multibpemb-Encoder

We augment the basic Multibpemb-Conv model by an attention mechanism for two layers, applied on top of the multilingual sub-word embeddings. We do this using a modivied version of the Transformer model [Vaswani et. al. 2017], in which the decoder has been removed entirely and the encoder is limited to only two layers (to preserve computation efficiency).

### Char-Based Models

#### Char-Conv (small & large)
We develop a character based convolutional neural network, adapted from [Zhang et. al., 2015]. It uses a separate embedding for each unicode character, employing stacked convolutions thereover to generate prediction scores.

## Results

On a 1pct sample (due to computationally expensive state-of-the-art models):
| Model Name                               | Reply  |          | Retweet |         | Rt_w_com |          | Like   |         |
|------------------------------------------|--------|----------|---------|---------|----------|----------|--------|---------|
|                                          | PR-AUC | RCE      | PR-AUC  | RCE     | PR-AUC   | RCE      | PR-AUC | RCE     |
| BERT                                     | 0.0735 |   7.0204 |  0.2713 |  9.1249 |   0.0118 |   1.8522 | 0.6301 | 10.4906 |
| ROBERTA                                  | 0.0767 |   7.2825 |  0.2719 |  9.0547 |   0.0121 |   1.8062 | 0.6295 | 10.3674 |
| DistilBERT                               | 0.0714 |   6.7646 |  0.2675 |  8.4121 |   0.0120 |   1.5910 | 0.6259 |  9.8327 |
| CHAR_CONV_LARGE                          | 0.0515 |   4.4876 |  0.2199 |  5.6382 |   0.0088 |   0.4265 | 0.5978 |  7.9373 |
| CHAR_CONV_SMALL                          | 0.0557 |   5.0489 |  0.2301 |  6.2268 |   0.0093 |   0.0554 | 0.6024 |  8.2440 |
| CHAR_CONV_BASE                           | 0.0512 |   4.5430 |  0.2324 |  6.2385 |   0.0096 |  -0.2096 | 0.5889 |  7.4211 |
| MULTIBPEMB_CONV_BASE (100k)              | 0.0495 |   4.0945 |  0.2245 |  5.8505 |   0.0089 |  -1.2833 | 0.5841 |  7.5113 |
| MULTIBPEMB_CONV_BASE (100k, learn. emb.) | 0.0535 |   4.1012 |  0.2227 |  5.4711 |   0.0087 |  -1.4115 | 0.5778 |  6.9265 |
| MULTIBPEMB_CONV_BASE (1000k)             | 0.0523 |   4.5059 |  0.2224 |  6.0110 |   0.0089 |  -0.5108 | 0.5821 |  7.4159 |
| MULTIBPEMB_CONV_TWOLAYER (100k & 1000k)  | 0.0446 |   1.4209 |  0.2064 |  4.2417 |   0.0081 |  -9.5799 | 0.5631 |  5.4103 |
| MULTIBPEMB_ENCODER (100k)                | 0.0602 |   5.6371 |  0.2416 |  6.9282 |   0.0096 |   0.4329 | 0.6009 |  8.4794 |
| MULTIBPEMB_ENCODER (100K, learn. emb.)   | 0.0562 |   4.1920 |  0.2207 |  5.3033 |   0.0094 |  -0.5323 | 0.5827 |  7.3758 |
| MULTIBPEMB_ENCODER (1000k)               | 0.0606 |   5.2849 |  0.2424 |  6.9639 |   0.0099 |   0.1154 | 0.6034 |  8.5806 |

On a 10pct sample:

| Model Name                               | Reply  |        | Retweet |        | Rt_w_com |        | Like   |        |
|------------------------------------------|--------|--------|---------|--------|----------|--------|--------|--------|
|                                          | PR-AUC | RCE    | PR-AUC  | RCE    | PR-AUC   | RCE    | PR-AUC | RCE    |
| CHAR_CONV_LARGE                          | 0.0572 | 5.2286 |  0.2247 | 6.1752 |   0.0098 | 0.8643 | 0.6006 | 8.5159 |
| CHAR_CONV_SMALL                          | 0.0651 | 5.9617 |  0.2424 | 7.2500 |   0.0102 | 1.1952 | 0.6099 | 8.9363 |
| CHAR_CONV_BASE                           | 0.0574 | 5.1806 |  0.2400 | 7.0350 |   0.0101 | 0.9354 | 0.5972 | 8.2221 |
| MULTIBPEMB_CONV_BASE (100k)              | 0.0581 | 5.3107 |  0.2342 | 6.8027 |   0.0104 | 1.0613 | 0.5946 | 8.2331 |
| MULTIBPEMB_CONV_BASE (100k, learn. emb.) | 0.0630 | 5.5465 |  0.2403 | 7.0326 |   0.0108 | 0.8853 | 0.5952 | 8.0602 |
| MULTIBPEMB_CONV_BASE (1000k)             | 0.0587 | 5.4066 |  0.2394 | 7.0404 |   0.0104 | 1.0909 | 0.5967 | 8.4147 |
| MULTIBPEMB_CONV_TWOLAYER (100k  & 1000k) | 0.0514 | 4.0816 |  0.2058 | 3.8022 |   0.0096 | 0.3232 | 0.5614 | 5.1150 |
| MULTIBPEMB_ENCODER (100k)                | 0.0695 | 6.6942 |  0.2512 | 8.1153 |   0.0114 | 1.6365 | 0.6120 | 9.5136 |
| MULTIBPEMB_ENCODER (100k, learn. emb.)   | 0.0615 | 5.6250 |  0.2343 | 6.8047 |   0.0103 | 0.9268 | 0.5928 | 8.0311 |
| MULTIBPEMB_ENCODER (1000k)               | 0.0721 | 6.8844 |  0.2567 | 8.3877 |   0.0124 | 1.8871 | 0.6162 | 9.7253 |

Selected models were scaled to a 30pct sample (yielding no additional improvement):
| Model Name                | Reply  |        | Retweet |        | Retweet with comment |        | Like   |        |
|---------------------------|--------|--------|---------|--------|----------------------|--------|--------|--------|
|                           | PR-AUC | RCE    | PR-AUC  | RCE    | PR-AUC               | RCE    | PR-AUC | RCE    |
| CHAR_CONV_SMALL           | 0.0637 | 5.4933 |  0.2438 | 7.0034 |               0.0098 | 0.9048 | 0.6092 | 9.0747 |
| CHAR_CONV_LARGE           | 0.0536 | 4.6447 |  0.2231 | 5.6485 |               0.0093 | 0.8651 | 0.5982 | 8.1487 |
| MULTIBPEMB_ENCODER (100k) | 0.0711 | 6.6097 |  0.2559 | 8.2909 |               0.0121 | 1.7164 | 0.6133 | 9.4816 |

Numbers in paranthesis state the input vocabulary size of the multlingual sub-word embeddings. Experiments were carried out with and without learnable embeddings.
Most Experiments were carried out on a single GTX1080TI GPU.
State-of-the-art models needed between 4 and 6 hours per training epoch, while the remaining architectures trained 10-100 times faster (despite random initialization).

## Links

RecsysChallenge 2020 <https://www.recsyschallenge.com/2020/>

14th ACM Conference on Recommender Systems <https://recsys.acm.org/recsys20/>

ACM RecSys <https://recsys.acm.org/>

## References

Mikel Artetxe and Holger Schwenk. Massively multilingual sentence embeddings for zero-shot cross-lingual transfer and beyond. Transactions of the Association for Computational Linguistics, 7:597–610, 2018.

Dzmitry Bahdanau, Kyunghyun Cho, and Yoshua Bengio. Neural machine translation by jointly learning to align and translate. International Conference on Learning Representations, 2014.

Luca Belli, Sofia Ira Ktena, Alykhan Tejani, Alexandre Lung-Yut-Fon, Frank Portman, Xiao Zhu, Yuanpu Xie, Akshay Gupta, Michael Bronstein, Amra Delic, Gabriele Sottocornola, Walter Anelli,  azareno Andrade, Jessie Smith, and Wenzhe Shi. Privacy-aware recommender systems challenge on twitter’s home timeline. arXiv, 2004.13715, 2020.

Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. BERT: pre-training of deep bidirectional transformers for language understanding. arXiv, 1810.04805, 2018.

Benjamin Heinzerling and Michael Strube. BPEmb: Tokenization-free Pre-trained Subword Embeddings in 275 Languages. Proceedings of the International Conference on Language Resources and Evaluation, 2018.

Yoon Kim. Convolutional neural networks for sentence classification. Proceedings of the Conference on Empirical Methods in Natural Language Processing, 2014.

Xiang Zhang, Junbo Zhao, and Yann LeCun. Character-level convolutional networks for text classification. Advances in Neural Information Processing Systems, 15:649–657, 2015.

Yinhan Liu, Myle Ott, Naman Goyal, Jingfei Du, Mandar Joshi, Danqi Chen, Omer Levy, Mike Lewis, Luke Zettlemoyer, and Veselin Stoyanov. Roberta: A robustly optimized BERT pretraininga pproach. arXiv, 1907.11692, 2019.

Victor Sanh, Lysandre Debut, Julien Chaumond, and Thomas Wolf. Distilbert, a distilled version of bert: smaller, faster, cheaper and lighter. arXiv, 1910.01108, 2019.

Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, and Illia Polosukhin. Attention is all you need.  Proceedings of the International Conference on Neural  formation Processing Systems, 2017.

## Acknowledgements

For the encoder model we thank <https://nlp.seas.harvard.edu/2018/04/03/attention.html> for their implementation of the transformer.

We thank <https://nlp.h-its.org/bpemb/> for their user-friendly wrapper library providing us with multilingual sub-word embeddings.

Thank you Huggingface <https://huggingface.co/> and FacebookAI <https://engineering.fb.com/2019/01/22/ai-research/laser-multilingual-sentence-embeddings/> for the easy setup of the state-of-the-art NLP models! 
