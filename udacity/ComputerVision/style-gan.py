import os
import pickle
import matplotlib.pyplot as plt
import torch

# Add styleGAN3 repo to the paths where python
# searches for modules
import sys
sys.path.append("/home/student/stylegan3")

with open('/home/student/.cache/dnnlib/downloads/20755e1ffb4380580e4954f8b0f9e630_stylegan3-r-afhqv2-512x512.pkl', 'rb') as f:
    stylegan_model = pickle.load(f)

stylegan_model.keys()