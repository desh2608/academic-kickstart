+++
title = "Sptial Transformer Networks"
date = 2016-11-20T17:00:42+05:30
draft = true

# Tags: can be used for filtering projects.
# Example: `tags = ["machine-learning", "deep-learning"]`
tags = ["deep learning","computer vision"]

# Project summary to display on homepage.
summary = "Course project under [Prof. Arijit Sur](https://www.iitg.ac.in/arijit/)"

# Optional image to display on homepage.
image_preview = "stn.jpg"

# Optional external URL for project (replaces project detail page).
external_link = ""

# Does the project detail page use math formatting?
math = false

# Does the project detail page use source code highlighting?
highlight = true

# Featured image
# Place your image in the `static/img/` folder and reference its filename below, e.g. `image = "example.jpg"`.
[header]
image = "stn.jpg"
caption = "Architecture of the Spatial Transformer Network"

+++
[Jaderberg et al.](https://papers.nips.cc/paper/5854-spatial-transformer-networks.pdf) proposed the Spatial Transformer Network in NIPS 2015 in order to improve the classification of transformed images (i.e., images with affine transformation). In this project, we achieved 2 objectives:

1. Improved the STN architecture by applying a recurrence in the outermost layer, i.e., transformed images are again fed into the module for further processing.
2. Applied the network to egocentric image data to improve benchmark datasets like GTEA and Intel Egocentric Vision data.

* [Report](report/stn.pdf)
* [Slides](ppt/stn.pdf)