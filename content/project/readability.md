+++
title = "Text readability analysis using language modeling"
date = 2017-04-30T17:01:44+05:30
draft = false

# Tags: can be used for filtering projects.
# Example: `tags = ["machine-learning", "deep-learning"]`
tags = ["natural language processing"]

# Project summary to display on homepage.
summary = "Course project under [Prof. Ashish Anand](http://www.iitg.ac.in/anand.ashish/)"

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
image = ""
caption = ""

+++
We conjecture that predictability of a text is a viable metric of its readability. By using modern language models as predictors, we believe this metric may provide an automated, fine-grained measure of readability. It also provides a natural mechanism to combine scores from different language models, and hence the ability to generalize to a diverse set of texts. Individual language models encode the specific linguistic background that a reader may have, hence providing customized scores for each type of reader. Our work provides authors with a valuable tool to

1. assess the readability of their content for readers with different linguistic backgrounds, and
2. identify pain-points at a word-level granularity in their text in order to improve it.

Our evaluations support our conjecture and show that the resulting scores work across a wide range of scenarios.

* [Report](report/readability.pdf)
* [Slides](ppt/readability.pdf)