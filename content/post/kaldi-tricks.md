+++
title = "Some Kaldi Things"
date = 2019-03-27T12:05:41-04:00
draft = false

# Tags and categories
# For example, use `tags = []` for no tags, or the form `tags = ["A Tag", "Another Tag"]` for one or more tags.
tags = ["Kaldi"]
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
*This is a regularly updated post on some tips and tricks for working with [Kaldi](http://kaldi-asr.org/).* 

List of contents:

1. [How to stop training mid-way and decode using last trained stage](#stop-train)
2. [About `Sum()` and `Append()` in Kaldi xconfig](#sum-append)
3. [Checking training logs](#train-logs)
4. [Converting between FV and FM types](#convert)
5. [Number of epochs in Kaldi](#epochs)

***

<a name="stop-train"></a>
### How to stop training mid-way and decode using last trained stage


In the Kaldi chain model, suppose you are training for 4 epochs (which is close to 1000 iterations in the usual run of the TED-LIUM recipe). During training, suppose you decide to stop midway and check the decoding result. 

Now, the training can be stopped and resumed simply by supplying the arguments `--stage` and `--train-stage`, where the input to `stage` is the stage where the `train.py` is called, and `train-stage` is the stage from where you want to continue training.

But if you stop at, say, stage 239, and want to decode, you first have to prepare the model for testing. This is so that dropout and batchnorm aren't performed at test time. For this, first run

```
nnet3-am-copy --prepare-for-test=true <dir>/239.mdl <dir>/final.mdl
```

This creates a testing model called `final.mdl` which the `decode.sh` script uses for decoding. Instead of using the default name `final`, you can create any test copy name, say `239-final.mdl`. To use this mdl file for decoding, pass this as argument to the `--iter` argument in `decode.sh`.

***

<a name="sum-append"></a>
### About `Sum()` and `Append()` in Kaldi xconfig

If you have worked with Kaldi xconfig, it is pretty easy to define layer inputs and outputs, using something called [`Descriptors`](http://kaldi-asr.org/doc/dnn3_code_data_types.html). They act as a glue between components and can also perform easy operations like append, sum, scale, round, etc. So, for instance, you can have the following xconfig:

```
input name=ivector dim=100
input dim=40 name=input
relu-batchnorm-layer name=tdnn1 dim=1280 input=Append(-1,0,1,ReplaceIndex(ivector, t, 0))
linear-component name=tdnn2l dim=256 input=Append(-1,0)
relu-batchnorm-layer name=tdnn2 input=Append(0,1) dim=1280
linear-component name=tdnn3l dim=256
relu-batchnorm-layer name=tdnn3 dim=1280 input=Sum(tdnn3l,tdnn2l)
```

This network does not make too much sense and is only for purpose of representation. At some point, you may require to do something of the sort `Sum(Append(x,y),z)`, i.e., append two inputs and add it to a third input. This operation, however, isn't allowed in the xconfig. 

This is because `Sum()` takes 2 `<sum-descriptor>` types, while the output of `Append()` is a `<descriptor>` type which is a super class of `<sum-descriptor>`, and as such, there is an argument type mismatch. This can be easily solved:

```
no-op-component name=noop1 input=Append(x,y)
relu-batchnorm-layer name=tdnn3 dim=1280 input=Sum(noop1,tdnn2l)
```

Similarly, a `Scale()` outputs a `<fwd-descriptor>` while a `Sum()` expects a `<sum-descriptor>`, so to use `Scale()` inside `Sum()` we first have to pass it through a `no-op-component`.
***

<a name="train-logs"></a>
### Checking training logs

When you are training any chain model in Kaldi, it is important to know if the parameters are getting updated well and if the objective function is improving. All such information is stored in the `log` directories in Kaldi, but since there is so much information in there, it may be difficult to find what you are looking for.

Suppose your working directory is something like `exp/chain/tdnn_1a/`. Then, first go to the `log` directory by
```
cd exp/chain/tdnn_1a/log
```
Now, to check the objective functions for all the training iterations, do
```
ls -lt train* | grep -r 'average objective' .
```
This will print something like this, for all the iterations.
```
LOG (nnet3-chain-train[5.5.103~1-34cc4e]:PrintTotalStats():nnet-training.cc:348) Overall average objective function for 'output' is -0.100819 over 505600 frames.
LOG (nnet3-chain-train[5.5.103~1-34cc4e]:PrintTotalStats():nnet-training.cc:348) Overall average objective function for 'output-xent' is -1.17531 over 505600 frames.
```
Here, our actual objective is 'output'. The other objective is the cross-entropy regularization term. To avoid printing it, you can replace `'average objective'` with `"average objective function for 'output'"` in the previous command. Look at the values. If the model is learning well, the objective should be increasing (since it is the log-likelihood).

You may also want to see if your parameters are updating how you want them to be. For this, do
```
ls -lt progress* | grep -r 'Relative parameter differences' .
```
Usually, the relative parameter differences are close to the learning rate.

***

<a name="convert"></a>
### Converting between FM and FV types

Kaldi has two major types: Matrix and Vector. As such, features are often stored in one of these two file types. For instance, when you extract i-vectors, they are stored as a matrix of floats (FM) and if you extract x-vectors, they are stored as vectors of float (FV). Often it may be required to convert features stored as FV to FM and vice-versa.

Although there is no dedicated Kaldi binary to perform this conversion, we can leverage the fact that the underlying text format for both these types is the same and use this as an intermediate for the conversion. For example, to convert from FV to FM:
```
copy-vector --binary=false scp:exp/xvectors/xvector.scp ark,t:- | \
  copy-matrix ark,t:- ark,scp:exp/xvectors/xvector_mat.ark,exp/xvectors/xvector_mat.scp
```

Similarly, to convert from FM to FV:
```
copy-matrix --binary=false scp:exp/ivectors/ivector.scp ark,t:- | \
  copy-vector ark,t:- ark,scp:exp/ivectors/ivector_vec.ark,exp/ivectors/ivector_vec.scp
```
***

<a name="epochs"></a>
### Number of epochs in Kaldi

This is borrowed directly from [Dan's reply](https://groups.google.com/d/msg/kaldi-help/7OrqJI2Szvg/vk3P8qKWAwAJ) in a `kaldi-help` Google Group post.

> A few of the reasons we use relatively few epochs in Kaldi are as follows:

> * We actually count epochs *after* augmentation, and with a system that has frame-subsampling-factor of 3 we separately train on the data shifted by -1, 0 and 1 and count that all as one epoch.  So for 3-fold augmentation and frame-subsampling-factor=3, each "epoch" actually ends up seeing the data 9 times.

> * Kaldi uses natural gradient, which has better convergence properties than regular SGD and allows you to train with larger learning rates; this might allow you to reduce the num-epochs by at least a factor of 1.5 or 2 versus what you'd use with normal SGD.

> * We do model averaging at the end-- averaging over the last few iterations of training (an iteration is an interval of usually a couple minutes' training time).  This allows us to use relatively large learning rates at the end and not worry too much about the added noise; and it allows us to use relatively high learning rates at the end, which further decreases the training time.  This wouldn't work without the natural gradient; the natural gradient stops the model from moving too far in the more important directions within parameter space.

> * We start with aligments learned from a GMM system, so the nnet doesn't have to do all the work of figuring out the alignments-- i.e. it's not training from a completely uninformed start.

> So supposing we say we are using 5 epochs, we are really seeing the data more like 50 times, and if we didn't have those tricks (NG, model averaging) that might have to be more like 100 or 150 epochs, and without knowing the alignments, maybe 200 or 300 epochs.