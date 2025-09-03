import kagglehub

import os
os.environ['KAGGLEHUB_CACHE'] = '/home/mpham/workspace/source/tfoundation/nosos/scripts/meds/avh_kaggle'

# Download latest version
path = kagglehub.dataset_download(
    "nehaprabhavalkar/av-healthcare-analytics-ii", 
    force_download=True
)

print("Path to dataset files:", path)