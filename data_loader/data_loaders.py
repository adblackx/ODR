from torchvision import datasets, transforms
from data_loader.dataset import Dataset
from torch.utils.data.dataloader import default_collate
import numpy as np
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader
import pandas as pd
from torch.utils.data.sampler import SubsetRandomSampler


class odr_data_loader(DataLoader):
	"""
		ODR custom data loading: This data loader will load the data in a train and valid set of size validation_split
		extended : bool means that if DataSet getitem function will returns img_transformed, label, age or img_transformed, label
		dataAug : bool that means if DataSet getitem function will use data augmentation or not
	"""
	def __init__(self, data_dir, image_dir, batch_size, shuffle, equal_dist, validation_split, num_workers,extended, dataAug, collate_fn=default_collate):

		self.data_dir = data_dir
		self.validation_split = validation_split
		self.shuffle = shuffle
		self.equal_dist = equal_dist
		
		self.batch_size = batch_size
		self.image_dir = image_dir

		trsfm  = transforms.Compose([
			transforms.Resize(256),
			transforms.CenterCrop(244),
			transforms.ToTensor(),
			transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
		])

		self.dataset = Dataset(self.data_dir, self.image_dir, trsfm, extended, dataAug)
		self.sampler, self.valid_sampler = self._split_sampler(self.validation_split)

		self.init_kwargs = {
			'dataset': self.dataset,
			'batch_size': batch_size,
			'shuffle': self.shuffle,
			'collate_fn': collate_fn,
			'num_workers': num_workers
		}

		# we don't use the data augmentation on the ValidSet because the split sampler 
		# function provides an equal proportion of classes to validate
		self.init_kwargs2 = { 
			'dataset': Dataset(self.data_dir, self.image_dir, trsfm, extended, False),
			'batch_size': batch_size,
			'shuffle': self.shuffle,
			'collate_fn': collate_fn,
			'num_workers': num_workers
		}

		super().__init__(sampler=self.sampler, **self.init_kwargs)

	def countNbElementByClasses(self, labels, msg = ""):
		"""
		function to check that the desired train and valid set sizes have been produced
		"""
		labels = np.array(labels)
		print(msg)
		for i in np.unique(labels):
			print("nb classes" , i , " : ", np.count_nonzero(labels == i))
		
	def _split_sampler(self, split):
		"""
			We return two iterators on the train set and the valid set
			split: boolean that represents the proposal of the valid Set

		"""

		if split == 0.0:
			return None, None
		#We get the labels and the filename of our dataset
		
		labels_list = self.dataset.labels
		size = len(labels_list)
		print("inital dataset size :", size)


		#we build an array of indexes that we mix if requested
		labels = labels_list
		idx_full = np.arange(size)
		if(self.shuffle):
			np.random.shuffle(idx_full)

		nb_img = np.zeros(len(np.unique(labels)))
		unique = np.unique(labels)
		if(self.equal_dist):
			nb = int(split * len(labels)/len(unique))
			nb_img += nb
			cpt = 0
			for i in range(len(unique)):
				nb_img[i] = int(min(nb_img[i], np.count_nonzero(labels == i)/3))
				cpt += nb_img[i]
			print("Real split : {}%".format(cpt/size))
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

		train_sampler = SubsetRandomSampler(train_idx)
		valid_sampler = SubsetRandomSampler(valid_idx)
			
		print("train size:", len(train_idx),"-- validation size:", len(valid_idx), "-- total size dataset:", len(train_idx)+len(valid_idx))
		self.countNbElementByClasses(labels[train_idx], msg= "[TRAIN SET]")
		self.countNbElementByClasses(labels[valid_idx], msg= "[VALIDATION SET]")
		

		return train_sampler, valid_sampler
		

	def split_validation(self): 
		"""
			we will call this function to get the valid_sampler called 
			in the constructor of this class of the previous function

			We will use another Dataloader to iterate on the valid set, 
			it is one of the specificities of the architecture
		"""
		if self.valid_sampler is None:
			return None
		else:
			return DataLoader(sampler=self.valid_sampler, **self.init_kwargs2)