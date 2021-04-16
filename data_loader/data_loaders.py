from torchvision import datasets, transforms
from base.base_data_loader import BaseDataLoader
from data_loader.dataset import Dataset
from torch.utils.data.dataloader import default_collate
import numpy as np
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader
import pandas as pd
from torch.utils.data.sampler import SubsetRandomSampler


class odr_data_loader(DataLoader):
	"""
	MNIST data loading demo using BaseDataLoader
	"""
	def __init__(self, data_dir, image_dir, batch_size, shuffle, equal_dist, validation_split, num_workers,extended, collate_fn=default_collate):
		"""trsfm = transforms.Compose([
			transforms.ToTensor(),
			transforms.Normalize((0.1307,), (0.3081,))
		])"""
		self.data_dir = data_dir
		self.validation_split = validation_split
		self.shuffle = shuffle
		self.equal_dist = equal_dist
		
		self.batch_size = batch_size
		self.image_dir = image_dir
		#transforms.Resize(256)
		#transforms.ToPILImage(mode=None)
		trsfm  = transforms.Compose([
			transforms.Resize(256),
			transforms.CenterCrop(244),
			transforms.ToTensor(),
			transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
		])

		self.dataset = Dataset(self.data_dir, self.image_dir, trsfm,extended)
		self.sampler, self.valid_sampler = self._split_sampler(self.validation_split)

		self.init_kwargs = {
			'dataset': self.dataset,
			'batch_size': batch_size,
			'shuffle': self.shuffle,
			'collate_fn': collate_fn,
			'num_workers': num_workers
		}

		super().__init__(sampler=self.sampler, **self.init_kwargs)

	def countNbElementByClasses(self, labels, msg = ""):
		labels = np.array(labels)
		print(msg)
		for i in np.unique(labels):
			print("nb classes" , i , " : ", np.count_nonzero(labels == i))
		
	def _split_sampler(self, split):
		print("LE BON SPLIT")
		if split == 0.0:
			return None, None
		#On extrait les labels et le filename de notre dataset

		print("SPLIT:", split)
		
		labels_list = self.dataset.labels
		size = len(labels_list)
		print("inital dataset size :", size)

		#On transforme les labels en chiffres
		'''labels_unique = np.unique(labels_list).tolist()
		labels = []
		for i in labels_list:
			labels.append(labels_unique.index(i))
		labels = np.array(labels)'''

		#on construit un tableau d'index qu'on mélange si demandé
		labels = labels_list
		idx_full = np.arange(size)
		if(self.shuffle):
			np.random.shuffle(idx_full)

		nb_img = np.zeros(len(np.unique(labels)))
		unique = np.unique(labels)
		if(self.equal_dist):
			nb = int(split * len(labels)/len(unique))
			nb_img += nb
			for i in range(len(unique)):
				nb_img[i] = int(min(nb_img[i], np.count_nonzero(labels == i)/3))
		else:
			for i in unique:
				nb_img[i] = int(split * np.count_nonzero(labels == i))
			
		valid_idx = []
			
		for i in idx_full:
			idx = labels[i]
			if nb_img[idx] > 0:
				nb_img[idx]-=1
				valid_idx.append(i)

		valid_idx = np.array(valid_idx)
		train_idx = np.delete(idx_full, valid_idx)

		#normalement on peut s'en passer ?
		train_sampler = SubsetRandomSampler(train_idx)
		valid_sampler = SubsetRandomSampler(valid_idx)
		#train_sampler, valid_sampler = train_idx, valid_idx
			
		print("train size:", len(train_idx),"-- validation size:", len(valid_idx), "-- total size dataset:", len(train_idx)+len(valid_idx))
		self.countNbElementByClasses(labels[train_idx], msg= "[TRAIN SET]")
		self.countNbElementByClasses(labels[valid_idx], msg= "[VALIDATION SET]")
		print(train_sampler)
		

		return train_sampler, valid_sampler
		

	def split_validation(self): 
	# on va appeler cette fonction pour récupérer le valid_sampler appele dans le constructeur de cette classe de la fonction précédente
		if self.valid_sampler is None:
			return None
		else:
			return DataLoader(sampler=self.valid_sampler, **self.init_kwargs)