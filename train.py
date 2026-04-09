from tqdm import tqdm
from tensorboardX import SummaryWriter
import os
import numpy as np
import torch
from datetime import datetime
from algos import generate_algo
from data_generator import get_loader
from utils import merge_trees, shrink_tree, unfold_dicts, compute_grad_norm, set_seeds
import yaml
import shutil
from metrics import get_metric
from math import ceil
from typing import List
import hashlib

class Trainer:
	def __init__(self,
			    name_expe:str,
				lr:float,
				batch_size:int,
				epoch:int,
				dataset:dict,
				seed:int,
				metrics:List[str],
				algo_specs:dict,
				log_interval:int = 200,
				load:str = None,
				clip_grad:float = None,
				device: str = None,
				weight_decay:float = 0,
				lr_scheduler:dict = None):
		
		self.epoch = epoch
		self.log_interval = log_interval
		self.clip_grad = clip_grad

		self.model_folder = f"./expe/{name_expe}/"

		# chose device
		if device is None :
			if torch.cuda.is_available() :
				device = "cuda"
			else :
				device = "cpu"

		# Load the dataset
		print(f"{device=}")

		self.loader = get_loader(loader_specs=dataset,
								       batch_size=batch_size,
									   device=device)
		self.nfo = self.loader.nfo
		self.nfo["dataname"] = dataset["name"]

		# Set seeds
		if type(seed) == int :
			set_seeds(seed)
		
		# Generate the algo
		algo_specs["device"] = device
		self.algo = generate_algo(algo_specs, self.nfo)
		print(f"n parameters: {sum(p.numel() for p in self.algo.parameters())}")
		self.algo.to(device)

		# Load params if necessary
		if load is not None :
			path = "expe/"+load+"/last_model"
			self.algo.load(path)

		# Create optimizer
		params_lr_coeff = self.algo.params_lr_coeff
		for p in params_lr_coeff :
			p["lr"] *= lr
		self.optimizer = torch.optim.AdamW(params_lr_coeff,
									 lr=lr,
									 weight_decay=weight_decay)
		if lr_scheduler is not None :
			if lr_scheduler["type"] == "linear" :
				Scheduler = torch.optim.lr_scheduler.LinearLR
			elif lr_scheduler["type"] == "step" :
				Scheduler = torch.optim.lr_scheduler.StepLR
			else :
				raise ValueError("Unknown lr_scheduler")
			del lr_scheduler["type"]
			self.lr_scheduler = Scheduler(self.optimizer, **lr_scheduler)
		else :
			self.lr_scheduler = None

		# Tensorboard for monitoring
		self.writer = SummaryWriter(self.model_folder, flush_secs=30)

		# Metrics
		self.metrics = [get_metric(metric, self.algo, self.nfo, self.loader) for metric in metrics]

	def train_logging(self,
					  coeffs) :
			losses=merge_trees(*self.L_losses)
			for k,v in unfold_dicts(losses).items() :
				self.writer.add_scalar('loss/train'+k, v, self.iter)
			loss = shrink_tree(merge_trees(losses,coeffs,f=lambda x:x[0]*x[1]))
			self.writer.add_scalar('loss/train/total', loss, self.iter)

			self.writer.add_scalar('metrics/grad_norms/max', np.max(self.grad_norms), self.iter)
			self.writer.add_scalar('metrics/grad_norms/mean', np.mean(self.grad_norms), self.iter)

	def eval_step(self):
			L_losses = []
			for X, A in self.loader :
				losses, coeffs = self.algo.compute_loss(X, A, eval=True)
				L_losses.append(losses)

			losses = merge_trees(*L_losses)

			for k,v in unfold_dicts(losses).items() :
				self.writer.add_scalar('loss/eval'+k, v, self.iter)
			loss = shrink_tree(merge_trees(losses,coeffs,f=lambda x:x[0]*x[1]))
			self.writer.add_scalar('loss/eval/total', loss, self.iter)

			for metric in self.metrics :
				m = metric.compute_metrics()
				for k,v in m.items() :
					self.writer.add_scalar(f'metrics/{metric}/{k}', v, self.iter)

			self.algo.save(self.model_folder+'last_model')
			


	def reset_metrics(self) :
		self.grad_norms = []
		self.L_losses = []

	def train(self) :
		self.iter=0
		self.reset_metrics()
		for _ in tqdm(range((ceil(self.epoch/len(self.loader))))) :
			for X, A in self.loader:
				self.optimizer.zero_grad()

				losses, coeffs = self.algo.compute_loss(X, A)
				loss = shrink_tree(merge_trees(losses,coeffs,f=lambda x:x[0]*x[1]))
				self.L_losses.append(losses)

				loss.backward()
				if self.clip_grad is not None :
					torch.nn.utils.clip_grad_norm_(self.algo.parameters(), self.clip_grad)
				self.grad_norms.append(compute_grad_norm(self.algo))
				self.optimizer.step()

				if (self.iter+1) % self.log_interval == 0 :
					self.train_logging(coeffs)
					self.eval_step()
					self.reset_metrics()

				if self.lr_scheduler is not None :
					self.lr_scheduler.step()

				if self.iter >= self.epoch :
					self.writer.close()
					return

				self.iter += 1

		self.writer.close()



def train(config_file):
	with open(config_file, 'r') as file: 
		specs = yaml.safe_load(file)
	if specs.get("name_expe", None) is None :
		specs["name_expe"] = str(datetime.now())
	os.makedirs(f"./expe/{specs['name_expe']}", exist_ok=True)
	shutil.copyfile(config_file, f"./expe/{specs['name_expe']}/config.yaml")

	trainer = Trainer(**specs)
	trainer.train()

if __name__ == "__main__":
	train("./test.yaml")
