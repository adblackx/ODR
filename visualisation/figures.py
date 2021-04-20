import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


from sklearn.metrics import plot_confusion_matrix
from sklearn.svm import SVC

class Plot():
	def __init__(self,filepath):
		self.filepath = filepath
		self.data = pd.read_csv(filepath)


	def printLoss(self,ax=None,getAx=False,label1="",label2="",color=""):
		if ax == None :
			fig, ax = plt.subplots(figsize=(10,5))

		x = self.data[['epoch']]
		y1 = self.data[['loss']]

		y2 = self.data[['val_loss']]

		if color=="":
			ax.plot(x, y1, label="train loss "+label1)
			ax.plot(x, y2, label="validation loss "+label2)
		else:
			ax.plot(x, y1, label="train loss "+label1, color=color)
			ax.plot(x, y2, label="validation loss "+label2, color=color,linestyle="--")
		ax.legend()
		ax.set_title("Loss en fonction de l'epoque")
		

		if getAx:
			return ax
		else:
			plt.show()

	def printAccuracy(self,ax=None,getAx=False,label1="",label2="",color=""):
		if ax == None:
			fig, ax = plt.subplots(figsize=(10,5))

		x = self.data[['epoch']]
		y1 = self.data[['accuracy']]
		y2 = self.data[['val_accuracy']]

		if color=="":
			ax.plot(x, y1, label="train accuracy "+label1)
			ax.plot(x, y2, label="validation accuracy "+label2)
		else:
			ax.plot(x, y1, label="train accuracy "+label1, color=color)
			ax.plot(x, y2, label="validation accuracy "+label2, color=color,linestyle="--")
		ax.legend()
		ax.set_title("Accuracy en fonction de l'epoque")
		

		if getAx:
			return ax
		else:
			plt.show()

	def plot_confusion_matrix(self,
								cm,
							  target_names,
							  title='Confusion matrix',
							  cmap=None,
							  normalize=True):
		"""
		given a sklearn confusion matrix (cm), make a nice plot

		Arguments
		---------
		cm:           confusion matrix from sklearn.metrics.confusion_matrix

		target_names: given classification classes such as [0, 1, 2]
					  the class names, for example: ['high', 'medium', 'low']

		title:        the text to display at the top of the matrix

		cmap:         the gradient of the values displayed from matplotlib.pyplot.cm
					  see http://matplotlib.org/examples/color/colormaps_reference.html
					  plt.get_cmap('jet') or plt.cm.Blues

		normalize:    If False, plot the raw numbers
					  If True, plot the proportions

		Usage
		-----
		plot_confusion_matrix(cm           = cm,                  # confusion matrix created by
																  # sklearn.metrics.confusion_matrix
							  normalize    = True,                # show proportions
							  target_names = y_labels_vals,       # list of names of the classes
							  title        = best_estimator_name) # title of graph

		Citiation
		---------
		http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html

		"""
		import matplotlib.pyplot as plt
		import numpy as np
		import itertools

		accuracy = np.trace(cm) / np.sum(cm).astype('float')
		misclass = 1 - accuracy

		if cmap is None:
			cmap = plt.get_cmap('Blues')

		plt.figure(figsize=(15, 15))
		plt.imshow(cm, interpolation='nearest', cmap=cmap)
		plt.title(title)
		plt.colorbar()

		if target_names is not None:
			tick_marks = np.arange(len(target_names))
			plt.xticks(tick_marks, target_names, rotation=45)
			plt.yticks(tick_marks, target_names)

		if normalize:
			cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]


		thresh = cm.max() / 1.5 if normalize else cm.max() / 2
		for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
			if normalize:
				plt.text(j, i, "{:0.4f}".format(cm[i, j]),
						 horizontalalignment="center",
						 color="white" if cm[i, j] > thresh else "black")
			else:
				plt.text(j, i, "{:,}".format(cm[i, j]),
						 horizontalalignment="center",
						 color="white" if cm[i, j] > thresh else "black")


		plt.tight_layout()
		plt.ylabel('True label')
		plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
		plt.show()
