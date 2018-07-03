+++
title = "Similarity analysis on multidimensional fuzzy sets"
date = 2015-07-22T17:01:05+05:30
draft = false

# Tags: can be used for filtering projects.
# Example: `tags = ["machine-learning", "deep-learning"]`
tags = ["fuzzy"]

# Project summary to display on homepage.
summary = "Summer research project under [Prof. Frank Chung-Hoon Rhee](http://fuzzy.hanyang.ac.kr/members_prof.html)"

# Optional image to display on homepage.
image_preview = "fuzz-16-visual.jpg"

# Optional external URL for project (replaces project detail page).
external_link = ""

# Does the project detail page use math formatting?
math = false

# Does the project detail page use source code highlighting?
highlight = true

# Featured image
# Place your image in the `static/img/` folder and reference its filename below, e.g. `image = "example.jpg"`.
[header]
image = "tfs-16-analysis.png"
caption = "Flow diagram of the proposed method"

+++
The objective of the project was to propose guidelines for selecting fuzzy membership functions to represent several data sets. In general, the accuracy of representation increases with increasing complexity, which makes it a trade-off. My contributions are listed below.

* Analyzed various multidimensional fuzzy membership functions and compared similarity of data sets using Wilcoxons nonparametric tests.
* Established guidelines for selecting appropriate MFs based on data set and application requirements.
* Recently extended the proposed method for high-dimensional data using dimensionality reduction approaches like PCA, kernel PCA, probabilistic PCA, and t-SNE.

We proposed a new log-time algorithm which makes use of Wilcoxon's nonparametric tests to compare similarity between the original data and the synthetic data generated using the fuzzy MFs. The returned similarity score guides the choice of membership function.