# EECS 600 Final Project: Co-teaching Improvements

This repository contains a commandline program for training co-teaching models ((link to paper)[https://papers.nips.cc/paper/8072-co-teaching-robust-training-of-deep-neural-networks-with-extremely-noisy-labels])

# File Descriptions

- `data.py` contains method to obtain specific dataset.
- `model.py` contains different neural network models.
- `trainer.py` contains generic trainers for managing the training processes
- `task.py` the command-line program entry-point that accepts multiple arguments and invokes the training procedures defined in the aforementioned files

## Run on Google Cloud

```bash
gcloud ai-platform jobs submit training <job name> --package-path cot --module-name cot.task --region us-east1 --python-version 3.5 --runtime-version 1.14 --job-dir gs://co-training-project/keras --scale-tier BASIC_GPU
```