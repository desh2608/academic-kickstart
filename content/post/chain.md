+++
title = "On lattice free MMI and Chain models in Kaldi"
date = 2019-05-21T11:49:12-04:00
draft = false

# Tags and categories
# For example, use `tags = []` for no tags, or the form `tags = ["A Tag", "Another Tag"]` for one or more tags.
tags = ["kaldi","chain"]
categories = []

# Featured image
# Place your image in the `static/img/` folder and reference its filename below, e.g. `image = "example.jpg"`.
# Use `caption` to display an image caption.
#   Markdown linking is allowed, e.g. `caption = "[Image credit](http://example.org)"`.
# Set `preview` to `false` to disable the thumbnail in listings.
[header]
image = ""
caption = ""
preview = true

+++
Recently, I came across [this paper](https://arxiv.org/pdf/1811.03700.pdf) which compares several sequence discriminative training criteria based on the popular lattice-free MMI (LF-MMI) objective, and concludes that "boosted" LF-MMI outperforms others consistently. Since I couldn't find the code publicly available, I set out to implement it myself in [Kaldi](http://kaldi-asr.org/). The idea was that even if the claim turned out to be false, this would give me a hands-on experience with C++ level implementations in Kaldi. 

On first look, the implementation seems trivial if you already have a LF-MMI (also called the "chain" model in Kaldi) implementation available. However, there are several tricks used in Kaldi which are worth pointing out. In this article, I start with giving an overview of LF-MMI and its implementation in the chain models, and then talk about how I implemented boosted LF-MMI. The majority of the theory here is based on [this paper which introduced LF-MMI](https://www.danielpovey.com/files/2016_interspeech_mmi.pdf) and [this doc on chain model](http://kaldi-asr.org/doc/chain.html).

### MMI -- a background

Maximum mutual information, or MMI, is a sequence discriminative training criteria popular in ASR. "Sequence" means that the objective takes into account the utterance as a whole instead of "frame-level" objectives like cross-entropy. "Discriminative" loosely means using an objective function which supposedly optimizes some criteria associated with the task, and then minimizing that objective directly using gradient-based methods. Discriminative training for LVCSR was made popular in [Dan Povey's thesis](https://www.danielpovey.com/files/phd_2003.pdf). Formally, the MMI objective for ASR is written as

$$ F\_{MMI}(\lambda) = \sum\_{r=1}^R \log \frac{P\_{\lambda}(O\_r|M\_{w\_r})P(w\_r)}{\sum\_{\hat{w}}P\_{\lambda}(O\_r|M\_{\hat{w}})P(\hat{w})}, $$

where $M\_w$ is the HMM corresponding to the transcription $w$. As you can see, the objective function considers the log-probability of the whole utterance in the numerator, and normalizes it by dividing with the log-probability of all possible utterances in the denominator.

However, computing the sum in the denominator means summing over an exponentially large number of word sequences, which is not practically feasible. To remedy this, we approximate the sum with either of two methods:

1. **N-best list**: This is computed once and used for all utterances. However, this approximation is less used since it is too crude.

2. **Lattice structure**: This may be word/phone based. A path through the lattice represents a possible word/phone sequence. One limitation with using a lattice is that it requires initialization with a trained model, and usually cross-entropy trained systems are used for this purpose. The older [nnet](https://kaldi-asr.org/doc/dnn.html) setups in Kaldi used this approach. 

With the advent of end-to-end models, such a requirement of a trained system to initialize the lattice comes across as a major drawback of lattice-based MMI. How can we avoid using a lattice?

### Lattice-free MMI

First proposed in [this paper from Dan Povey](https://www.danielpovey.com/files/2016_interspeech_mmi.pdf), lattice-free MMI is "purely sequence trained" in the sense that no cross-entropy training is required to initialize, since it does not use a lattice. So how does it approximate the sum in the denominator? Simply put, it does not "approximate" it --- it computes this sum exactly.

The key idea is that if we represent the denominator as a graph and somehow manage to fit this graph in the GPU, then computation can be performed efficiently. In the manner that it is formalized, the denominator graph cannot be fit into the GPU. To fix this, two major modifications are applied:

1. A phone LM is used instead of a word LM. The number of possible phones is much smaller than the number of possible words, which makes the size of graph for phone LM significantly smaller.

2. DNN outputs are computed at one-third the standard frame rate, which means that we now have 3 times fewer outputs to compute for any utterance. This is achieved by setting the frame shift to 30 ms instead of the traditional 10 ms.

This reduced frame rate also means that now we cannot use the standard 3-state left-to-right HMM topology that is common in ASR, since we want to traverse the entire HMM in a single frame. Instead, we use an HMM which can emit symbols in the set `ab*`.

To train such a system according to the MMI objective, we need a way to efficiently compute the objective itself and its derivative. In Kaldi, the numerator and denominator are represented as FSTs (corresponding to the HMMs) and the overall objective function is simply the difference of these in log-space. As such, we need a way to efficiently represent these FSTs and perform forward-backward on them.

### The denominator and numerator FSTs

Let us start with the denominator FST since it is much more expensive. The process of creating the denominator FST is very similar to the [decoding graph creation](http://kaldi-asr.org/doc/graph_recipe_test.html). The key idea, as in traditional ASR using WFSTs (see [Mohri's well-known paper](https://cs.nyu.edu/~mohri/pub/hbka.pdf)), is to have separate FSTs for `H` (HMM state graph), `C` (context-dependency), `L` (the lexicon), and `G` (the language model), and use WFST composition algorithms to get the final graph, with the exception that since we are using phones instead of words, we don't need the `L` graph. So our final graph is actually an `HCP` instead of an `HCLG`, where `P` denotes the phone LM.

At this point, I would like to point out some Kaldi specifics. The phone LM `P` is created in stage `-6` by calling the function [`create_phone_lm()`](https://github.com/kaldi-asr/kaldi/blob/8b54ef83e20b682a0b1f91cdbaf6abd53ce3c32d/egs/wsj/s5/steps/libs/nnet3/train/chain_objf/acoustic_model.py#L25). The denominator FST is created in the stage `-5` within the `train.py` script, which internally makes a [call to the binary `chain-make-den-fst`](https://github.com/kaldi-asr/kaldi/blob/8b54ef83e20b682a0b1f91cdbaf6abd53ce3c32d/egs/wsj/s5/steps/libs/nnet3/train/chain_objf/acoustic_model.py#L53). The denominator graph is specificied in [`chain-den-graph.cc`](https://github.com/kaldi-asr/kaldi/blob/master/src/chain/chain-den-graph.cc). It uses the files `$dir/tree` (the tree) and `$dir/0.trans_mdl` (the transition model), which correspond to the `C` and `H` components, and the phone LM that was created in the previous stage.

The phone LM `P` is constructed so that the overall size of the graph is minimized. It is a 4-gram with no backoff lower than 3-gram so that triphones not seen in training cannot be generated. The number of states is limited by completely removing low-count 4-gram states.

Once we have the composed graph `HCP`, a different kind of minimization technique is used, which consists of performing the following operations thrice in a row.

1. *Push* the weights
2. *Minimize* the graph
3. *Reverse* the arcs and swap initial and final states.

Another trick used to reduce the size of the denominator FST for training on the GPU is to train on chunks of 1-1.5 seconds, instead of the entire utterance. However, to do this, we would also need to break up the transcript, and 1-second chunks may not coincide with word boundaries. How do we solve this?

Recall that the numerator FST is defined to be utterance-specific, and encodes alternative pronunciations of the transcript of the original utterance. This lattice is turned into an FST that constrains at what time the phones can appear, with an error window of 0.05s from their position in the lattice. This is then processed into an FST whose labels are pdf-ids (neural net outputs). We extract fixed size chunks from this FST for training chunks in the denominator FST.

Another issue associated with chunk-level FSTs is that the initial probabilities are now different. We approximate this by running the HMM for a few iterations and then averaging the probabilities to use as the initial probability of any state. This is a crude approximation but it seems to work. We then call this the **normalization FST**.

The numerator FST is much easier since it just contains the lattice for one utterance, broken into chunks of fixed length. The only point worth mentioning here (and this will be important when we talk about boosted LF-MMI later) is that the numerator FST is composed with the normalization FST. This is done for two reasons.

1. It ensures that the objective function value is always negative, which makes it easier to interpret.
2. It also ensures that the numerator FST does not contain sequences that are not allowed by the denominator (or normalization) FST. This happens since the sum of the overall path weights for such sequences will be dominated by the normalization FST part.

### Forward-backward computations

Again, since the numerator FST is much smaller, its forward and backward computations are performed on CPU (the process is outlined in [`chain-numerator.h`](https://github.com/kaldi-asr/kaldi/blob/master/src/chain/chain-numerator.h)), while those for the denominator FST (outlined in [`chain-denominator.h`](https://github.com/kaldi-asr/kaldi/blob/master/src/chain/chain-denominator.h)) are performed on the GPU.

The basic forward and backward algorithm are the same as well known in literature, and a pseudocode is also given in the extended comments in `chain-denominator.h`. However, this algorithm is susceptible to numeric overflow and underflow. To avoid this, we multiply the emission probability of the frame with a normalizing factor $\frac{1}{alpha(t)}$ where $alpha(t) = \sum\_{i} \alpha\_i (t)$. This is also called an "arbitrary scale" since in principle it can be allowed to be any value and doesn't affect the posterior. However, we do need to add a quantity $\sum\_{t=0}^{T-1} \log alpha(t)$ to the final log probability obtained to make it equal to the actual log probability. This "arbitrary scaling" is used in both the forward and backward computations.

The actual objective function computation is implemented in [`ComputeChainObjfAndDeriv()`](https://github.com/kaldi-asr/kaldi/blob/8b54ef83e20b682a0b1f91cdbaf6abd53ce3c32d/src/chain/chain-training.cc#L205) defined in `chain-training.cc`. There are two Kaldi-specific things I must point out here.

1. The forward-backward computation for the denominator FST in the GPU is not done in the log domain, since computing log several times makes things slower. However, this also means that the objective function values can occasionally become "bad". To fix this, the [`PenalizeOutOfRange()`](https://github.com/kaldi-asr/kaldi/blob/8b54ef83e20b682a0b1f91cdbaf6abd53ce3c32d/src/chain/chain-training.cc#L49) function is used to encourage the objective to be within the [-30,30] range.

2. The denominator computation is performed before the numerator, so as to reduce the maximum memory usage. I am not sure how this is, but it is important to remember this detail as we move to the implementation of boosted LF-MMI.

### Implementing boosted LF-MMI

First, what is boosted LF-MMI? It is the same as LF-MMI, except that now we optimize the following objective function.

$$ F\_{bMMI}(\lambda) = \sum\_{r=1}^R \log \frac{P\_{\lambda}(O\_r|M\_{W\_r})P(W\_r)}{\sum\_{\hat{w}}P\_{\lambda}(O\_r|M\_{\hat{w}})P(\hat{w})e^{-bA(M\_{w_r},M\_{\hat{w}})}}, $$

where $b$ is the boosting factor and $A(M\_{w\_r},M\_{\hat{w}})$ is the accuracy function which measures the number of matching labels between the reference and hypothesis sequences. My Kaldi implementation for LF-bMMI can be found in [this branch](https://github.com/desh2608/kaldi/tree/bmmi). You may note that most of the changes are cosmetic and only serve to pass the new argument $b$ from the training script to the actual implementation, which is in the function [`ComputeBoostedChainObjfAndDeriv()`](https://github.com/desh2608/kaldi/blob/2e46097b7e4fcd1a07a7e9c1df6f1aaa062fbc33/src/chain/chain-training.cc#L319).

In our implementation, the only change is that in the [computation for `num_logprob_weighted`](https://github.com/desh2608/kaldi/blob/2e46097b7e4fcd1a07a7e9c1df6f1aaa062fbc33/src/chain/chain-training.cc#L369), we subtract from `numerator.forward()` by a term `b * num_seq * frames_per_seq`. This might seem weird at first, since in the expression of the objective function, we actually subtract the denominator by this term. However, recall that the numerator FST is composed with the normalization FST, so that this modification will result in the same result as the objective function above.

On trying out LF-bMMI for mini-Librispeech, I found it to be slightly worse than regular LF-MMI (11.86 vs 11.74 WER), and consultation with [Vimal Manohar](http://vimalmanohar.github.io/) revealed that he had tried LF-bMMI and LF-SMBR along with [Hossein Hadian](https://hhadian.github.io/) last year to similar results.  