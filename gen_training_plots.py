# Filename: gen_training_plots.py
# Date Created: 13-May-2019 10:44:46 pm
# Description: Script to generate the training/validation loss plot against epochs.
import numpy as np
import matplotlib.pyplot as plt
import argparse

def main():
    # Add parser to parse in the arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-t_loss_file', type=str, required=True)
    parser.add_argument('-v_loss_file', type=str, required=True)
    opt = parser.parse_args()

    print("Generating and saving plot...")

    # load in the training loss and validation loss
    t_loss = np.load(opt.t_loss_file)
    v_loss = np.load(opt.v_loss_file)

    # plot seperately
    plt.plot(np.arange(len(t_loss))+1,t_loss)
    plt.plot(np.arange(len(v_loss))+1,v_loss)
    plt.xlabel('No. of Epochs')
    plt.ylabel('NLL Loss')
    plt.grid()
    plt.legend(['Training','Validation'])
    plt.savefig('./plots/combined_plot.png',dpi=300,
                bbox_inches='tight', pad_inches=0)

if __name__ == "__main__":
    main()
