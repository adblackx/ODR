from torchvision import datasets, transforms
from base.base_data_loader import BaseDataLoader
from data_loader.dataset2 import Dataset
from torch.utils.data.dataloader import default_collate


class odr_data_loader(BaseDataLoader):
	"""
	MNIST data loading demo using BaseDataLoader
	"""
	def __init__(self, data_dir, filename_list, labels_list, batch_size, shuffle, validation_split, num_workers, collate_fn=default_collate):
		"""trsfm = transforms.Compose([
			transforms.ToTensor(),
			transforms.Normalize((0.1307,), (0.3081,))
		])"""
		

		trsfm  = transforms.Compose([
			transforms.Resize(256),
			transforms.CenterCrop(224),
			transforms.ToTensor(),
			transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
		])

		self.data_dir = data_dir
		self.dataset = Dataset(filename_list, labels_list, trsfm)

		super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)


 	