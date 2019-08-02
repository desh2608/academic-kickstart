---
# Documentation: https://sourcethemes.com/academic/docs/managing-content/

title: "Xconfigs in Kaldi: A Tutorial"
subtitle: ""
summary: ""
authors: []
tags: ["kaldi"]
categories: []
date: 2019-07-29T11:34:00-04:00
lastmod: 2019-07-29T11:34:00-04:00
featured: false
draft: true

# Featured image
# To use, add an image named `featured.jpg/png` to your page's folder.
# Focal points: Smart, Center, TopLeft, Top, TopRight, Left, Right, BottomLeft, Bottom, BottomRight.
image:
  caption: ""
  focal_point: ""
  preview_only: false

# Projects (optional).
#   Associate this post with one or more of your projects.
#   Simply enter your project's folder or file name without extension.
#   E.g. `projects = ["internal-project"]` references `content/project/deep-learning/index.md`.
#   Otherwise, set `projects = []`.
projects: []
---
Before we get into what xconfigs are, let me first tell you what they are not. Xconfigs are not a deep neural network framework in Kaldi. Instead, they are a high-level config generator for `dnn3`, which is the Kaldi deep neural network implementation. When they were [first introduced](https://github.com/kaldi-asr/kaldi/issues/1124) the idea was to "have a simpler, intermediate version of the nnet3 config files-- kind of like the latex/tex distinction".

If you look at any of the `configs/final.config` files in the `exp` directories when running a Kaldi recipe, it is quite elaborate and repetitive. To reduce this redundancy for the user while writing new network architectures, the xconfig provides an easier high-level library, which is then parsed to generate the actual config files that build the network. Here is the [official description](https://github.com/kaldi-asr/kaldi/blob/master/egs/wsj/s5/steps/libs/nnet3/xconfig/__init__.py) in the library:

```
This library has classes and methods to form neural network computation graphs,
in the nnet3 framework, using higher level abstractions called 'layers'
(e.g. sub-graphs like LSTMS ).

Note : We use the term 'layer' though the computation graph can have a highly
non-linear structure as, other terms such as nodes/components have already been
used in C++ codebase of nnet3.

This is basically a config parser module, where the configs have very concise
descriptions of a neural network.

This module has methods to convert the xconfigs into a configs interpretable by
nnet3 C++ library.

It generates three different configs:

 'init.config' : which is the config with the info necessary for computing
               the preconditioning matrix i.e., LDA transform
               e.g.
                 input-node name=input dim=40
                 input-node name=ivector dim=100
                 output-node name=output input=Append(Offset(input, -2), Offset(input, -1), input, Offset(input, 1), Offset(input, 2), ReplaceIndex(ivector, t, 0)) objective=linear
 'ref.config' : which is a version of the config file used to generate
                a model for getting left and right context (it does not read
                anything for the LDA-like transform and/or
                presoftmax-prior-scale components)
 'final.config' : which has the actual config used to initialize the model used
                 in training i.e, it has file paths for LDA transform and
                 other initialization files
```

In this post, I will first list all available xconfig layers (as of July 2019), discuss how to build some basic network architectures using xconfigs in Kaldi, and some miscellaneous problems that come up in their use. Most of the discussion is based on questions raised in the `kaldi-help` Google group, the scripts in `local/nnet3` directories, and the code itself.

### The xconfig layers

A list of all available layers can be found [here](https://github.com/kaldi-asr/kaldi/blob/7637de77e0a77bf280bef9bf484e4f37c4eb9475/egs/wsj/s5/steps/libs/nnet3/xconfig/parser.py#L19). Here, I reproduce this list from the Python Class perspective. This is because several layer definitions map to the same class, since they are essentially different combinations of the same implementation.

1. [XconfigInputLayer](https://github.com/kaldi-asr/kaldi/blob/7637de77e0a77bf280bef9bf484e4f37c4eb9475/egs/wsj/s5/steps/libs/nnet3/xconfig/basic_layers.py#L314): This layer is used to feed inputs to the network, such as MFCCs or i-vectors. Example usage:
```
input name=input dim=40
```
2. [XconfigOutputLayer](https://github.com/kaldi-asr/kaldi/blob/7637de77e0a77bf280bef9bf484e4f37c4eb9475/egs/wsj/s5/steps/libs/nnet3/xconfig/basic_layers.py#L446): These are the actual output layers, which means they include a softmax by default (can be turned off using the `--include-log-softmax` option). Example usage:
```
output-layer name=output dim=4257 input=Append(input@-1, input@0, input@1)
```

3. [XconfigTrivialOutputLayer](https://github.com/kaldi-asr/kaldi/blob/7637de77e0a77bf280bef9bf484e4f37c4eb9475/egs/wsj/s5/steps/libs/nnet3/xconfig/basic_layers.py#L364): They are not exactly "layers" like the OutputLayer, since they have no linear or affine component. They just directly map to an output-node in nnet3. Example usage:
```
output name=output input=Append(input@-1, input@0, input@1)
```

4. [XconfigBasicLayer](https://github.com/kaldi-asr/kaldi/blob/7637de77e0a77bf280bef9bf484e4f37c4eb9475/egs/wsj/s5/steps/libs/nnet3/xconfig/basic_layers.py#L672): This class contains various combinations for non-linearities, comprising the following components. Possible combinations are mentioned in the example usages.

	* `relu`: $ f(x) = \max(0,x). $
	* `sigmoid`: $f(x) = \frac{1}{1 + e^{-x}}.$
	* `tanh`: $f(x) = \frac{e^{2x}-1}{e^{2x}+1}.$
	* `renorm`: This is a normalization component that was [implemented](https://github.com/kaldi-asr/kaldi/blob/master/src/nnet3/nnet-normalize-component.h) in Kaldi before BatchNorm became popular. It is similar to Hinton's layer-norm, except not normalizing the mean, only the variance. It implements the function
	$$ y = \frac{x \sqrt{d} \gamma}{|x|},$$
	where $d$ is the input dimension, and $\gamma$ is the [target-rms](#param-defs).
	* `batchnorm`: This is the conventional [BatchNorm](https://arxiv.org/pdf/1502.03167.pdf).
	* `so`: This is a Scale and Offset component. If you want to have updatable BatchNorm parameters, it is required to have an `so` after `batchnorm`.
	* `dropout`: This is the conventional [Dropout](https://www.cs.toronto.edu/~hinton/absps/JMLRdropout.pdf) method to prevent overfitting.

	Example usages:
```
relu-renorm-layer name=layer1 dim=1024 input=Append(-3,0,3)
sigmoid-layer name=layer1 dim=1024 input=Append(-3,0,3)
tanh-batchnorm-layer name=tdnn1 dim=1280
```

5. [XconfigAffineLayer](https://github.com/kaldi-asr/kaldi/blob/7637de77e0a77bf280bef9bf484e4f37c4eb9475/egs/wsj/s5/steps/libs/nnet3/xconfig/basic_layers.py#L1028): This applies an affine transformation on the input using an $(m+1) \times n$ matrix, where $m$ is the input dimensionality and $n$ is the output dimensionality ($n=m$ by default). Example usage:
```
affine-layer name=affine input=Append(-2,-1,0,1,2)
```

6. [XconfigFixedAffineLayer](https://github.com/kaldi-asr/kaldi/blob/7637de77e0a77bf280bef9bf484e4f37c4eb9475/egs/wsj/s5/steps/libs/nnet3/xconfig/basic_layers.py#L937): It is an affine transformation which is supplied at network initialization time and is not trainable.
```
fixed-affine-layer name=lda input=Append(-1,0,1,ReplaceIndex(ivector, t, 0)) affine-transform-file=foo/bar/lda.mat
```

7. [XconfigIdctLayer](https://github.com/kaldi-asr/kaldi/blob/7637de77e0a77bf280bef9bf484e4f37c4eb9475/egs/wsj/s5/steps/libs/nnet3/xconfig/basic_layers.py#L1128): It performs an inverse discrete cosine transform, to convert MFCCs to log Mel filterbank features. It is primarily used in ConvNet recipes. Example usage:
```
idct-layer name=idct dim=40 cepstral-lifter=22 affine-transform-file=foo/bar/idct.mat
```

8. [XconfigLstmLayer](https://github.com/kaldi-asr/kaldi/blob/7637de77e0a77bf280bef9bf484e4f37c4eb9475/egs/wsj/s5/steps/libs/nnet3/xconfig/lstm.py#L45): This adds an LSTM sub-graph (without output projection) to the network. Example usage:
```
lstm-layer name=lstm1 input=[-1] delay=-3
```

9. [XconfigLstmpLayer](https://github.com/kaldi-asr/kaldi/blob/7637de77e0a77bf280bef9bf484e4f37c4eb9475/egs/wsj/s5/steps/libs/nnet3/xconfig/lstm.py#L295): This is similar to XconfigLstmLayer, but also adds [output projection](https://arxiv.org/pdf/1402.1128.pdf) to the network. The overall network can be represented by the following equations.

$$ {i\_{t}=\sigma\left(W\_{i x} x\_{t}+W\_{i r} r\_{t-1}+W\_{i c} c\_{t-1}+b\_{i}\right)} $$
$$ {f\_{t}=\sigma\left(W\_{f x} x\_{t}+W\_{r f} r\_{t-1}+W\_{c f} c\_{t-1}+b\_{f}\right)} $$
$$ {c\_{t}=f\_{t} \odot c\_{t-1}+i\_{t} \odot g\left(W\_{c x} x\_{t}+W\_{c r} r\_{t-1}+b\_{c}\right)} $$
$$ {o\_{t}=\sigma\left(W\_{o x} x\_{t}+W\_{o r} r\_{t-1}+W\_{o c} c\_{t}+b\_{o}\right)} $$
$$ {r\_{t}=W\_{r m} m\_{t}} $$
$$ {p\_{t}=W\_{p m} m\_{t}} $$
$$ {p\_{t}=W\_{y p} p\_{t}+b\_{y}} $$
$$ {y\_{t}=W\_{y r} r\_{t}+W\_{y p} p\_{t}+b\_{y}} $$

Here, $p$ and $r$ are the recurrent and non-recurrent projections. In Kaldi implementation, these are replaced by a single projection.

10. [XconfigFastLstmLayer](https://github.com/kaldi-asr/kaldi/blob/7637de77e0a77bf280bef9bf484e4f37c4eb9475/egs/wsj/s5/steps/libs/nnet3/xconfig/lstm.py#L601): It performs the same function as XconfigLstmLayer, but in a faster manner by computing all non-linearities in a [special-purpose Cuda component](https://github.com/kaldi-asr/kaldi/blob/7637de77e0a77bf280bef9bf484e4f37c4eb9475/src/cudamatrix/cu-math.h#L132). Example usage:
```
fast-lstm-layer name=lstm1 input=[-1] delay=-3
``` 

11. [XconfigFastLstmpLayer](https://github.com/kaldi-asr/kaldi/blob/7637de77e0a77bf280bef9bf484e4f37c4eb9475/egs/wsj/s5/steps/libs/nnet3/xconfig/lstm.py#L994): Similar to XconfigFastLstmLayer but with output projection.

12. [XconfigLstmbLayer](https://github.com/kaldi-asr/kaldi/blob/7637de77e0a77bf280bef9bf484e4f37c4eb9475/egs/wsj/s5/steps/libs/nnet3/xconfig/lstm.py#L798): This is an unpublished LSTM variant with the following improvements over conventional LSTM.

	* Let $W\_{all}$ be the matrix combining all 8 full matrices in a standard LSTM. Then, $W\_{all}$ is factored into $W_a$ and $W_b$, with $W_a$ constrained to have orthonormal rows to keep training stable. (Note: This might have been a precursor to the [TDNN-F model](https://www.danielpovey.com/files/2018_interspeech_tdnnf.pdf).)
	* $W_b$ is followed by a trainable ScaleAndOffset component. (See [this paper](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/11/SelfLR.pdf) for a similar idea.)
	* The LSTM is followed by a BatchNorm component.

	Example usage:
```
lstmb-layer name=lstm1 input=[-1] delay=-3
```

13. [XconfigStatsLayer](https://github.com/kaldi-asr/kaldi/blob/7637de77e0a77bf280bef9bf484e4f37c4eb9475/egs/wsj/s5/steps/libs/nnet3/xconfig/stats_layer.py#L13): This is used in networks for training speaker embeddings (See [this paper](https://pdfs.semanticscholar.org/3697/28d7576683a25de8890e4bc02fae6132fccb.pdf) for details about the system). It pools the statistics (mean and std-dev) from the frame-level layers. Example usage:
```
stats-layer name=tdnn1-stats config=mean+stddev(-99:3:9:99) input=tdnn1
```

14. [XconfigConvLayer](https://github.com/kaldi-asr/kaldi/blob/7637de77e0a77bf280bef9bf484e4f37c4eb9475/egs/wsj/s5/steps/libs/nnet3/xconfig/convolution.py#L115): This adds a convolutional component to the network. Additionally, `batchnorm`, `relu`, `renorm`, etc. may also be applied after the `conv` operation. Example usages:
```
conv-batchnorm-layer name=conv2 height-in=40 height-out=40 \
      num-filters-out=64 height-offsets=-1,0,1 time-offsets=-1,0,1 \
      required-time-offsets=0
conv-renorm-layer name=conv3 height-in=40 height-out=20 \
      height-subsample-out=2 num-filters-out=128 height-offsets=-1,0,1 \
      time-offsets=-1,0,1 required-time-offsets=0
```

15. [XconfigResBlock](https://github.com/kaldi-asr/kaldi/blob/7637de77e0a77bf280bef9bf484e4f37c4eb9475/egs/wsj/s5/steps/libs/nnet3/xconfig/convolution.py#L416): This adds a residual block (similar to [ResNet](https://arxiv.org/pdf/1512.03385.pdf)). However, instead of adding the input to the output, it passes it through a convolutional layer followed by a BatchNorm component. The main path in this sub-graph is: `input -> relu1 -> batchnorm1 -> conv1 -> relu2 -> batchnorm2 -> conv2`. The `bypass-source` argument specifies what operations (`no-op`, `relu`, or `relu-batchnorm`) to apply to the input before adding it to the final output. Example usage:
```
res-block name=res1 num-filters=64 height=32 time-period=1
```


16. [XconfigRes2Block](https://github.com/kaldi-asr/kaldi/blob/7637de77e0a77bf280bef9bf484e4f37c4eb9475/egs/wsj/s5/steps/libs/nnet3/xconfig/convolution.py#L775): This is a more standard residual block, which does not support strided convolutions or any kind of downsampling of the input. Example usage:
```
res2-block name=res1 num-filters=64 height=32 time-period=1
```

17. [ChannelAverageLayer](https://github.com/kaldi-asr/kaldi/blob/7637de77e0a77bf280bef9bf484e4f37c4eb9475/egs/wsj/s5/steps/libs/nnet3/xconfig/convolution.py#L1149): This layer is used for averaging channels at the end of neural networks. [This script](https://github.com/kaldi-asr/kaldi/blob/master/egs/svhn/v1/local/nnet3/tuning/run_resnet_1d.sh#L106) provides an example usage for this layer.

18. [XconfigAttentionLayer](https://github.com/kaldi-asr/kaldi/blob/master/egs/wsj/s5/steps/libs/nnet3/xconfig/attention.py#L27): This adds a [self-attention](https://www.danielpovey.com/files/2018_icassp_attention.pdf) component in the network, with additional options for using nonlinearities. Example usage:
```
attention-renorm-layer num-heads=10 value-dim=50 key-dim=50 time-stride=3 num-left-inputs=5 num-right-inputs=2
```

19. [XconfigGruLayer](https://github.com/kaldi-asr/kaldi/blob/master/egs/wsj/s5/steps/libs/nnet3/xconfig/gru.py#L36): Similar to XconfigLstmLayer, but uses [GRU cells](https://en.wikipedia.org/wiki/Gated_recurrent_unit) instead of LSTM cells for recurrence. Example usage:
```
gru-layer name=gru1 input=[-1] delay=-3
```

20. [XconfigPgruLayer](https://github.com/kaldi-asr/kaldi/blob/master/egs/wsj/s5/steps/libs/nnet3/xconfig/gru.py#L197): It implements the Projected GRU proposed in [this paper](http://www.danielpovey.com/files/2018_interspeech_opgru.pdf), which is based on the LSTMP idea. The overall sub-graph can be summarized by the following equations.

$$ r\_{t} =\sigma\left(W\_{r x} x\_{t}+W\_{r s} s\_{t-1}+b\_{r}\right) $$
$$ z\_{t} =\sigma\left(W\_{z x} x\_{t}+W\_{z s} s\_{t-1}+b\_{z}\right) $$
$$ \tilde{h}\_{t} =\tanh \left(W\_{\tilde{h} x} x\_{t}+W\_{\tilde{h} s}\left(r\_{t} \odot s\_{t-1}\right)+b\_{\tilde{h}}\right) $$
$$ h\_{t} =\left(1-z\_{t}\right) \odot \tilde{h}\_{t}+z\_{t} \odot h\_{t-1} $$
$$ y\_{t} =W_{y h} h\_{t} $$
$$ s\_{t} =y\_{t}[0 : r-1] $$

21. [XconfigOpgruLayer](https://github.com/kaldi-asr/kaldi/blob/master/egs/wsj/s5/steps/libs/nnet3/xconfig/gru.py#L631): OPGRU stands for Output-gated Projected GRU. It is a generalization of the PGRU component

22. [XconfigNormPgruLayer]

23. [XconfigFastGruLayer]

24. [XconfigTdnnfLayer]

25. [XconfigPrefinalLayer]

26. [XconfigSpecAugmentLayer]

27. [XconfigRenormComponent]

28. [XconfigBatchnormComponent]

29. [XconfigNoOpComponent]

30. [XconfigLinearComponent]

31. [XconfigAffineCompoent]

32. [XconfigPerElementScaleComponent]

33. [XconfigDimRangeComponent]

34. [XconfigPerElementOffsetComponent]

35. [XconfigCombineFeatureMapsLayer]

<a name="param-defs"></a>
### Some Xconfig parameter definitions

In this section, we briefly summarize some parameters used in various xconfig layers.



### How to build some basic networks

#### 1. Simple feedforward network


