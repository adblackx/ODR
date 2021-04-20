import numpy as np
import torch
from torchvision.utils import make_grid
from base.base_trainer import BaseTrainer
from utils.util import inf_loop, MetricTracker


class Trainer(BaseTrainer):
	"""
	Trainer class
	"""
	def __init__(self, model, criterion, metric_ftns, optimizer, config, device,
				 data_loader, valid_data_loader=None, lr_scheduler=None, len_epoch=None):
		super().__init__(model, criterion, metric_ftns, optimizer, config)
		self.config = config
		self.device = device
		self.data_loader = data_loader
		if len_epoch is None:
			# epoch-based training
			self.len_epoch = len(self.data_loader)
		else:
			# iteration-based training
			self.data_loader = inf_loop(data_loader)
			self.len_epoch = len_epoch
		self.valid_data_loader = valid_data_loader
		self.do_validation = self.valid_data_loader is not None
		self.lr_scheduler = lr_scheduler
		self.log_step = int(np.sqrt(data_loader.batch_size))
		self.best_valid = 0
		self.train_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns], writer=self.writer)
		self.valid_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns], writer=self.writer)

		self.file_metrics = str(self.checkpoint_dir / 'metrics.csv')
		


	def _train_epoch(self, epoch):
		"""
		Training logic for an epoch
		:param epoch: Integer, current training epoch.
		:return: A log that contains average loss and metric in this epoch.
		"""
		
		self.model.train()
		self.train_metrics.reset()
		for batch_idx, (data, target) in enumerate(self.data_loader):
			data, target = data.to(self.device), target.to(self.device)

			self.optimizer.zero_grad()
			output = self.model(data)
			loss = self.criterion(output, target)
			loss.backward()
			self.optimizer.step()

			self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx)
			self.train_metrics.update('loss', loss.item())
			for met in self.metric_ftns:
				self.train_metrics.update(met.__name__, met(output, target))

			if batch_idx % self.log_step == 0:
				"""self.logger.debug('Train Epoch: {} {} Loss: {:.6f}'.format(
					epoch,
					self._progress(batch_idx),
					loss.item()))
				"""
				self.writer.add_image('input', make_grid(data.cpu(), nrow=8, normalize=True))

			if batch_idx == self.len_epoch:
				break
		log = self.train_metrics.result()

		if self.do_validation:
			val_log = self._valid_epoch(epoch)
			log.update(**{'val_'+k : v for k, v in val_log.items()})

		if self.lr_scheduler is not None:
			self.lr_scheduler.step()


		output = self.model(data)
		loss = self.criterion(output, target)	

		self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx)
		self.writer.add_scalar('Loss',  loss)

		self._save_csv(epoch, log)

		return log

	def _valid_epoch(self, epoch):
		"""
		Validate after training an epoch
		:param epoch: Integer, current training epoch.
		:return: A log that contains information about validation
		"""
		self.model.eval()
		self.valid_metrics.reset()
		with torch.no_grad():
			for batch_idx, (data, target) in enumerate(self.valid_data_loader):
				data, target = data.to(self.device), target.to(self.device)

				output = self.model(data)
				loss = self.criterion(output, target)

				self.writer.set_step((epoch - 1) * len(self.valid_data_loader) + batch_idx, 'valid')
				self.valid_metrics.update('loss', loss.item())
				for met in self.metric_ftns:
					self.valid_metrics.update(met.__name__, met(output, target))
				self.writer.add_image('input', make_grid(data.cpu(), nrow=8, normalize=True))

		# add histogram of model parameters to the tensorboard
		for name, p in self.model.named_parameters():
			self.writer.add_histogram(name, p, bins='auto')

		# we added spùe custom here
		output = self.model(data)
		loss = self.criterion(output, target)
		self.writer.set_step((epoch - 1) * len(self.valid_data_loader) + batch_idx, 'valid')
		self.writer.add_scalar('Loss',  loss)

		val_log = self.valid_metrics.result()
		actual_accu = val_log['accuracy']
		if(actual_accu - self.best_valid > 0.0025 and self.save):
			self.best_valid = actual_accu
			if self.tensorboard: # is true you can use tensorboard
				self._save_checkpoint(epoch, save_best=True)
			filename = str(self.checkpoint_dir / 'checkpoint-best-epoch.pth')
			torch.save(self.model.state_dict(), filename)
			self.logger.info("Saving checkpoint: {} ...".format(filename))

		return val_log

	def _progress(self, batch_idx):
		base = '[{}/{} ({:.0f}%)]'
		if hasattr(self.data_loader, 'n_samples'):
			current = batch_idx * self.data_loader.batch_size
			total = self.data_loader.n_samples
		else:
			current = batch_idx
			total = self.len_epoch
		return base.format(current, total, 100.0 * current / total)


	def _save_csv(self, epoch ,log):
		"""
			Saving checkpoints
			:param epoch: current epoch number
			:param log: logging information of the epoch
		"""

		fichier = open(self.file_metrics, "a")

		if epoch == 1:
			fichier.write("epoch,")
			for key in log:
				fichier.write(str(key) +",")
			fichier.write("\n")

		fichier.write(str(epoch) +",")
		for key in log:
			fichier.write(str(log[key]) + ",")
		fichier.write("\n")
		fichier.close()