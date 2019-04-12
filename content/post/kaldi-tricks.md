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
