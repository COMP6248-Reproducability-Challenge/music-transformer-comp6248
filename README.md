# music-transformer-comp6248
** This re-implementation is meant for the COMP6248 reproducibility challenge. **

Re-implementation of music transformer paper published for ICLR 2019.  
Link to paper: https://openreview.net/forum?id=rJe4ShAcF7  

## Scope of re-implementation
We focus on re-implementing a subset of the experiments described in the published paper,
only using the JSB Chorales dataset for training three model variations listed below:
- Baseline Transformer (TF) (5L, 256hs, 256att, 1024ff, 8h)
- Baseline TF + concatenating positional sinusoids
- TF with efficient relative attention (5L, 512hs, 512att, 512ff, 256r, 8h)

## Results
Unfortunately, we were unable to successfully reproduce the results shown in the published paper. There are a number of reasons that contribute to this, but mainly due to insufficient
details provided for the JSB dataset and how the data was processed for the transformer model.  

Nonetheless, these are the results obtained after training for **300** epochs:

|          Model Variations          | Final Training Loss | Final Validation Loss |
|:----------------------------------:|:-------------------:|:---------------------:|
|             Baseline TF            |        1.731        |         3.398         |
| Baseline TF + concat pos sinusoids |        2.953        |       3.575           |
|     TF with relative attention     |        TO ADD       |      TO ADD           |


Please read the [report](report.pdf) for more detailed information regarding the re-implementation.

## Usage
To train the models, simply run `train.py` and add the arguments accordingly:  
`python train.py -src_data datasets/JSB-Chorales-dataset/Jsb16thSeparated.npz -epochs 300 -weights_name baselineTF_300epoch -device cuda:2 -checkpoint 10`

To generate music using a trained model, run `generate.py` and add the arguments accordingly:  
`python generate.py -src_data datasets/JSB-Chorales-dataset/Jsb16thSeparated.npz -load_weights weights -weights_name baselineTF_300epoch -device cuda:1 -k 3`  

To generate the plots for training/validation loss, please use `gen_training_plots.py`:  
`python gen_training_plots.py -t_loss_file <t_loss_npy_file> -v_loss_file <v_loss_npy_file>`  

`<t_loss_npy_file>` and `<t_loss_npy_file>` are generated during training and are saved in the `outputs` directory.

The generated sequences can be plotted and listened using this [IPython Notebook](gen_sequence_audio.ipynb).  
**Please note that this requires Magenta to be installed.**

## Datasets
1. JSB Chorales dataset  
  - https://github.com/czhuang/JSB-Chorales-dataset
2. MAESTRO dataset  **(Not used)**
  - https://magenta.tensorflow.org/datasets/maestro

## Environment Setup
1. PyTorch with CUDA-enabled GPUs.
  - Install CUDA 9.0 and CUDNN 7.4.1.5
  - `conda create -n torch python=3.6`
  - `conda activate torch`
  - `conda install pytorch torchvision cuda90 -c pytorch`


2. Magenta for plotting and playing the generated notes.  
Steps for installing on Ubuntu subsystem:
  + `conda create -n magenta python=3.6`
  + `conda activate magenta`
  + `sudo apt-get update`
  + `sudo apt-get install build-essential libasound2-dev libjack-dev libfluidsynth1 fluid-soundfont-gm`
  + `pip install --pre python-rtmidi`
  + `pip install jupyter magenta pyfluidsynth pretty_midi`
