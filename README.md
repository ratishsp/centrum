
# Centrum 
This repo contains code for [Multi-Document Summarization with Centroid-Based Pretraining](https://arxiv.org/abs/2208.01006) (Ratish Puduppully and Mark Steedman.

## Citation
```
@misc{https://doi.org/10.48550/arxiv.2208.01006,
  doi = {10.48550/ARXIV.2208.01006},  
  url = {https://arxiv.org/abs/2208.01006},  
  author = {Puduppully, Ratish and Steedman, Mark},  
  title = {Multi-Document Summarization with Centroid-Based Pretraining},  
  publisher = {arXiv},  
  year = {2022}
}
```
## Requirements

Create conda environment:
```bash
conda create -n centrum_env python=3.7
conda activate centrum_env
conda install pytorch cudatoolkit=11 -c pytorch
```
All dependencies can be installed via:
```bash
pip install -r requirements.txt
```

## Code Details
The scripts to create the preprocessed NewSHead dataset are located in `utils` directory

## Model
The models are uploaded to HuggingFace. The pretrained checkpoint is available at [Centrum](https://huggingface.co/ratishsp/Centrum) and the checkpoint finetuned on MultiNews is available at [Centrum-multinews](https://huggingface.co/ratishsp/Centrum-multinews).

## Acknowledgements
Some code for preprocessing is based on the [Primera repo](https://github.com/allenai/PRIMER).

