[![GitHub license](https://img.shields.io/github/license/hamelsmu/code_search.svg)](https://github.com/hamelsmu/code_search/blob/master/LICENSE)
[![Python 3.6](https://img.shields.io/badge/python-3.6-blue.svg)](https://www.python.org/downloads/release/python-360/)

## Semantic Code Search

Code For Medium Article: "[How To Create A Natural Language Semantic Search Engine for Arbitrary Objects With Deep Learning]()"

![Alt text](./gifs/live_search.gif)

---
## Resources

#### Docker Containers

You can use these container to reproduce the environment the authors used for this tutorial.  Incase it is helpful, I have provided a [requirements.txt](./requirements/requirements.txt) file, however, we highly recommend using the docker containers provided below as the dependencies can be complicated to build yourself.

 - [hamelsmu/ml-gpu](https://hub.docker.com/r/hamelsmu/ml-gpu/): Use this container for any *gpu* bound parts of the tutorial.  We recommend running the entire tutorial on an aws `p3.8xlarge` and using this image.

 - [hamelsmu/ml-cpu](https://hub.docker.com/r/hamelsmu/ml-cpu/): Use this container for any *cpu* bound parts of this tutorial.


 #### Notebooks

 The [notebooks](./notebooks) folder contains 5 Jupyter notebooks that correspond to Parts 1-5 of the tutorial.


#### Related Blog Posts

This tutorial assumes knowledge of the material presented in [a previous tutorial on sequence-to-sequence models](https://towardsdatascience.com/how-to-create-data-products-that-are-magical-using-sequence-to-sequence-models-703f86a231f8).

---
## PRs And Comments Are Welcome

We have made best attempts to make sure running this tutorial is as painless as possible.  If you think something can be improved, please submit a PR!   
