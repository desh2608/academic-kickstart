+++
title = "Relation extraction for clinical text"
date = 2017-04-30T17:00:24+05:30
draft = false

# Tags: can be used for filtering projects.
# Example: `tags = ["machine-learning", "deep-learning"]`
tags = ["deep learning","natural language processing"]

# Project summary to display on homepage.
summary = "Bachelor Thesis Project under [Prof. Ashish Anand](http://www.iitg.ac.in/anand.ashish/)"

# Optional image to display on homepage.
image_preview = "conll-17-learning.png"

# Optional external URL for project (replaces project detail page).
external_link = ""

# Does the project detail page use math formatting?
math = false

# Does the project detail page use source code highlighting?
highlight = true

# Featured image
# Place your image in the `static/img/` folder and reference its filename below, e.g. `image = "example.jpg"`.
[header]
image = "conll-17-learning.png"
caption = "Architecture of the CRNN-Att network"

+++
The objective of the project was to devise a method for obtaining structured triplets from unstructured clinical records such as journal articles, patient health records etc. Simplifying this objective, I was tasked with creating a neural technique which can classify relations existing between entities in a given sentence, an NLP task known as relation classification.

The key insight is that convolutions can capture short-term phrases, while recurrence learns long-term dependencies. Combining both, we proposed the CRNN model which outperformed earlier single and double layer methods on two benchmark datasets: i2b2-2010 and DDI. Details about the method can be found in the publication.

This project was done as part of my undergraduate senior thesis.