from torchvision import datasets, transforms
from base.base_data_loader import BaseDataLoader
from data_loader.dataset import Dataset
from torch.utils.data.dataloader import default_collate
import numpy as np
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader


class odr_data_loader(BaseDataLoader):
	"""
	MNIST data loading demo using BaseDataLoader
	"""
	def __init__(self, data_dir, batch_size, shuffle, validation_split, num_workers, collate_fn=default_collate):
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

		self.dataset = Dataset(data_dir, trsfm)


		super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)
		#self.sampler, self.valid_sampler = self._split_sampler(self.validation_split) # inutile car le parent appelle la fonction en dessous

		
	def _split_sampler(self, split):
		print("RANDOM CONTROLE CARRE COMME LE NORD DE LA COREE")
		

		if split == 0.0:
			return None, None

		idx_full = np.arange(self.n_samples)

		np.random.seed(0)
		np.random.shuffle(idx_full)

		if isinstance(split, int):
			assert split > 0
			assert split < self.n_samples, "validation set size is configured to be larger than entire dataset."
			len_valid = split
		else:
			len_valid = int(self.n_samples * split)

		valid_idx = idx_full[0:len_valid]
		train_idx = np.delete(idx_full, np.arange(0, len_valid))

		train_sampler = SubsetRandomSampler(train_idx)
		valid_sampler = SubsetRandomSampler(valid_idx)

		#train_sampler, valid_sampler = self.splitTrainTest(split) #TODO


		# turn off shuffle option which is mutually exclusive with sampler
		self.shuffle = False
		self.n_samples = len(train_idx)

		return train_sampler, valid_sampler

	def split_validation(self): 
	# on va appeler cette fonction pour récupérer le valid_sampler appele dans le constructeur de cette classe de la fonction précédente
		if self.valid_sampler is None:
			return None
		else:
			return DataLoader(sampler=self.valid_sampler, **self.init_kwargs)



def splitTrainTest(self,split):
    """
	Fonction qui renvoie deux array liste correspondant aux indices dans Dataset
	split c'est le pourcentage 
	Sachant que self.dataSet()
    """
    #TODO

    train_sampler = []
    valid_sampler = []
    return train_sampler, valid_sampler    


def splitTrainTest(filepath, test_size=0.05, test_equal_repartition = True, Shuffle = False):
    
    data = pd.read_csv(filepath)
        
    #On extrait les labels et le filename de notre dataset
    filename_list = data['filename'].to_numpy()
    labels_list = data['labels'].to_numpy()
    size = len(labels_list)
    
    #On transforme les labels en chiffres
    labels_unique = np.unique(labels_list).tolist()
    labels = []
    for i in labels_list:
        labels.append(labels_unique.index(i))
    labels = np.array(labels)
    
    ind = np.arange(size)
    if(Shuffle):
        np.random.shuffle(ind)
    
    
    #filename_list = [my_dir]*len(filename_list) + filename_list #Pour avoir le chemin exact de nos images et pas juste le nom
    
    if(test_equal_repartition):
        nb_img_by_classes = int(test_size * len(labels)/len(np.unique(labels)))
        
        print("image by classes :", nb_img_by_classes)
        nb_img = np.zeros(len(np.unique(labels))) + nb_img_by_classes
        
        test_list, y_test, train_list, y_train = [],[],[],[]
        
        for i in ind:
            idx = labels[i]
            if nb_img[idx] > 0:
                nb_img[idx]-=1
                test_list.append(filename_list[i])
                y_test.append(labels[i])
            else:
                train_list.append(filename_list[i])
                y_train.append(labels[i])
    else:
        #On va créer notre ensemble d'netrainement et de test
        #ind = np.arange(size)
        train_size = 1.0 - test_size
        end_train = int(train_size*len(filename_list))
        
        print(ind)
        index_train = ind[:end_train]
        index_test = ind[end_train:]
        
        train_list = filename_list[index_train]
        test_list = filename_list[index_test]
        
        #Ci-dessous nos labels
        y_train = labels[index_train]
        y_test = labels[index_test]
    
    print("train size:", len(y_train),"-- test size:", len(y_test) )
    countNbElementByClasses(y_train, msg = "[TRAIN SET]")
    countNbElementByClasses(y_test, msg = "[TEST SET]")
    
    return train_list, y_train, test_list, y_test    
    