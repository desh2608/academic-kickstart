+++
title = "Waldo: A system for optical character recognition"
date = 2018-07-03T16:56:46+05:30
draft = true

# Tags: can be used for filtering projects.
# Example: `tags = ["machine-learning", "deep-learning"]`
tags = ["natural language processing", "computer vision", "deep learning"]

# Project summary to display on homepage.
summary = "Contributor in project under [Prof. Daniel Povey](www.danielpovey.com)"

# Optional image to display on homepage.
image_preview = "waldo-ocr.png"

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
It is an ongoing project under Prof. Daniel Povey to develop an Optical Character Recognition system that is robust on focused as well as incidental text. My contributions are:

* Experimenting with the ICDAR 2015 Robust Reading Challenge dataset by modifying training script.
* A visualization and compression module for segmentation mask overlayed on images.

The system consists of a modified UNet first proposed in [this](https://arxiv.org/abs/1505.04597) paper.