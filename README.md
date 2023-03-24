# KT
Knowledge Transduction for Cross-Domain Few-Shot Learning (under reviewer)

# Prepare data
Follow the instructions in [this repo](https://github.com/ml-jku/chef) 
to acquire and set up the data sets. 

# Pre-training
Run `python pretrain.py config/cdfsl_pretrain.json`.

# Testing
Run `python main.py config/cdfsl_text_xdom.json --dataset {isic,cropdisease,eurosat,chest}`.

Note that `hebb_lr=0.0005` under 50-shot.

# Note
This code is built upon the implementation from [CHEF](https://github.com/ml-jku/chef). We greatly thank for the authors for their contributions.

The dataset, model, and code are for non-commercial research purposes only.
