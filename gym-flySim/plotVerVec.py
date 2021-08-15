import numpy as np
import matplotlib.pyplot as plt


def plotVarVec():
    for i in range(24):
        mat = np.load(f'results{i+1}.npy')


if __name__=='__main__':
    plotVarVec()