# Imports
import cv2 as cv
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt

# Load the distribution from the DIA
Datadict = sio.loadmat('results/output.mat')

# Load the distribution from the generation
generated_radii = np.load('results/generated_distribution.npy')

# Make the two histograms
bins = np.linspace(0,60,13,dtype=int)
def plot_histogram(array,bins,name):
    plt.figure()
    plt.hist(array,bins,density=True)
    plt.xlabel('Bubble radius [px]')
    plt.ylabel('Probability [-]')
    plt.grid(True)
    plt.ylim((0,0.05))
    # plt.show()
    plt.savefig(name,format="png")

plot_histogram(generated_radii,bins,'results/generated.png')
plot_histogram(Datadict['data1.tif'][:,2],bins,'results/calculated.png')

# Compare mean
mean_1 = generated_radii.mean()
mean_2 = Datadict['data1.tif'][:,2].mean()

print(f'The mean radius of the generated distribution is {mean_1}')
print(f'The mean radius of the detected distribution is {mean_2}')

N_detected = Datadict['data1.tif'][:,2].size
N_generated = generated_radii.size
N_missed = N_generated - N_detected
print(f'Generated: {N_generated} bubbles')
print(f'Detected: {N_detected} bubbles')
print(f'Missed: {N_missed} bubbles')

# plt.savefig(f'{output_folder}/pdf{n:05d}.png',format="png")