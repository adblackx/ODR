import pandas as pd
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import torch
import glob
import os
import sys
from pathlib import Path


class Plot():
	def __init__(self,filepath):
		self.filepath = filepath
		self.data = pd.read_csv(filepath)


	def printLoss(self):
		fig, ax = plt.subplots(figsize=(10,5))
		x = self.data[['epoch']]
		y1 = self.data[['loss']]

		y2 = self.data[['val_loss']]
		ax.plot(x, y1, label="train loss")
		ax.plot(x, y2, label="validation loss")
		ax.legend()
		ax.set_title("Loss en fonction de l'epoque")
		plt.show()


if __name__ == '__main__':
	# exemple
    print("Hello, world!")
    filepath = sys.argv[1]
    #os.chdir("..")
    #d = os.getcwd()
    aff = Plot(filepath)
    aff.printLoss()
    #getattr(aff,"printLoss")(self)