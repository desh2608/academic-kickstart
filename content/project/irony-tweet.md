+++
title = "Irony detection in tweets"
date = 2018-03-20T17:00:16+05:30
draft = false

# Tags: can be used for filtering projects.
# Example: `tags = ["machine-learning", "deep-learning"]`
tags = ["deep learning","natural language processing"]

# Project summary to display on homepage.
summary = "[SemEval 2018 Task 3](https://github.com/Cyvhee/SemEval2018-Task3)"

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
The task was to recognize whether a tweet has irony or not - binary classification. In essence, we identified 2 aspects that were essential to identify irony in tweets:

* Semantic interaction between text and hashtags, modeled using [holographic embeddings](https://arxiv.org/pdf/1510.04935.pdf) (or circular cross-correlations).
* World knowledge about irony in text, obtained through transfer learning from [DeepMoji](https://deepmoji.mit.edu/).

We were able to obtain a validation accuracy of 69%, although the model performed poorly in the final test phase. The code for the project is available [here](https://github.com/desh2608/tweet-irony-detection).