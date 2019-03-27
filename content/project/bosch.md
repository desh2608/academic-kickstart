+++
title = "Monitoring production line performance to reduce failures"
date = 2017-03-31T17:01:52+05:30
draft = true

# Tags: can be used for filtering projects.
# Example: `tags = ["machine-learning", "deep-learning"]`
tags = ["machine learning"]

# Project summary to display on homepage.
summary = "Course project under [Prof. Rashmi Dutta Baruah](https://www.iitg.ac.in/r.duttabaruah/)"

# Optional image to display on homepage.
image_preview = ""

# Optional external URL for project (replaces project detail page).
external_link = ""

# Does the project detail page use math formatting?
math = false

# Does the project detail page use source code highlighting?
highlight = true

# Featured image
# Place your image in the `static/img/` folder and reference its filename below, e.g. `image = "example.jpg"`.
[header]
image = "bosch.png"
caption = "Outline of proposed method for fault detection"

+++
This project was first floated as a [Kaggle competition](https://www.kaggle.com/c/bosch-production-line-performance), with the dataset made available by Bosch. 

In this work, we pose the task of fault detection as a binary classification problem. The features include numerical, categorical, and timestamp features, and hence warranty a combination of several techniques for efficiently solving the problem.

First, a biased sampling method is used to reduce the effect of skewed data distribution. Thereafter, the categorical features are represented as 3 numerical features using sparse online classification algorithms: stochastic truncated gradient (STG), forward-backward splitting (FOBOS), and enhanced regularized dual averaging (ERDA). Once features are obtained, we try several classification methods like SVM and feed-forward networks to perform the fault detection. Finally, the overall objective is optimized using a Bayesian optimization technique.

* [Report](report/bosch.pdf)
* [Slides](ppt/bosch.pdf)