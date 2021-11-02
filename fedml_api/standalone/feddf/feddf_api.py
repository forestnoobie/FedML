import copy
import logging
import random
import os
from _collections import defaultdict

import numpy as np
import torch
import wandb

from fedml_api.standalone.feddf.client import Client


class FeddfAPI(object):
    def __init__(self, dataset, unlabeled_dataset, device, args, model_trainer):
        self.device = device
        self.args = args
        [train_data_num, test_data_num, train_data_global, test_data_global,
         train_data_local_num_dict, train_data_local_dict, test_data_local_dict, class_num, \
                                                                        *valid_data_global] = dataset
        self.train_global = train_data_global
        self.test_global = test_data_global
        self.val_global = None
        if valid_data_global:
            self.val_global = valid_data_global[0]
        self.train_data_num_in_total = train_data_num
        self.test_data_num_in_total = test_data_num
        self.class_num = class_num

        self.client_list = []
        self.train_data_local_num_dict = train_data_local_num_dict
        self.train_data_local_dict = train_data_local_dict
        self.test_data_local_dict = test_data_local_dict

        ## For tracking best acc
        self._stats = defaultdict(int)

        ## For saving model directory
        self.save_model_dir = os.path.join(args.wandb_save_dir, "./model_parameters")
        os.makedirs(self.save_model_dir, exist_ok=True)

        [server_model_trainer, client_model_trainer] = model_trainer
        self.model_trainer = server_model_trainer
        self.model_trainer.save_model_dir = self.save_model_dir
        self.model_trainer.class_num = class_num
        self._setup_clients(train_data_local_num_dict, train_data_local_dict, test_data_local_dict, client_model_trainer)

        ## Distillation Fusion
        [train_data_num, test_data_num, unlabeled_train_dl, unlabeled_test_dl] = unlabeled_dataset
        self.unlabeled_train_data = unlabeled_train_dl
        self.unlabeled_test_data = unlabeled_test_dl
        self.unlabeled_train_data_num = train_data_num
        
        ## Fedmix
        self.fedmix = args.fedmix
        if self.fedmix :
            self.mean_dl = self.get_image_label_mean()
    
    def get_image_label_mean(self):
        
        image_means, label_means = torch.Tensor().to(args.device), torch.Tensor().to(args.device)
        
        for client_idx in client_idxs:
            image_mean, label_mean = self.generate_mean(client)
            images_means = torch.cat([image_means, image_mean])
            label_means = torch.cat([label_means, label_mean])
             
        return images_means, label_means

    def generate_mean(self, client_idx):
        
        # Setup client
        client_idx = client_indexes[idx]
        client.update_local_dataset(client_idx, self.train_data_local_dict[client_idx],
                                    self.test_data_local_dict[client_idx],
                                            self.train_data_local_num_dict[client_idx])
        local_training_data = client.local_training_data
        
        # Get mean
        images_means, labels_means = torch.Tensor().to(self.args.device), torch.Tensor().to(self.args.device)
        for batch_idx, (images, labels) in enumerate(local_training_data):
            images, labels = images.to(self.args.device), labels.to(self.args.device)
            images_mean = torch.mean(images, dim=0).unsqueeze(0)
            labels_mean = torch.mean(F.one_hot(labels, num_classes=self.args.num_classes).float(), dim=0).unsqueeze(0)
            images_means = torch.cat([images_means, images_mean], dim=0)
            labels_means = torch.cat([labels_means, labels_mean], dim=0)

        return images_means, labels_means
    
    def _setup_clients(self, train_data_local_num_dict, train_data_local_dict, test_data_local_dict, model_trainer):
        logging.info("############setup_clients (START)#############")
        for client_idx in range(self.args.client_num_per_round):
            c = Client(client_idx, train_data_local_dict[client_idx], test_data_local_dict[client_idx],
                       train_data_local_num_dict[client_idx], self.args, self.device, model_trainer)
            self.client_list.append(c)
        logging.info("############setup_clients (END)#############")


    def _init_logits(self):
        init_logits = torch.zeros(self.unlabeled_train_data_num, self.class_num, device=self.device)
        return init_logits

    def _update_stats(self, stats):
        past_stats = self._stats

        for k, v in stats.items():
            if "best_" + k not in past_stats.keys(): # Default value
                past_stats["best_" + k] = v

            if 'acc' in k :
                max_value = max(v, past_stats["best_" + k])
                past_stats["best_" + k] = max_value
            elif 'loss' in k :
                min_value = min(v, past_stats["best_" + k])
                past_stats["best_" + k] = min_value

        logging.info(past_stats)

    def train(self):
        w_global = self.model_trainer.get_model_params()
        for round_idx in range(self.args.comm_round):
            logging.info("################Communication round : {}".format(round_idx))

            w_locals = [] # N * C \ N : number of unlabeled dataset,  C : number of classes of labeled dataset
            avg_logits = self._init_logits() # For offline dataloader

            """
            for scalability: following the original FedAvg algorithm, we uniformly sample a fraction of clients in each round.
            Instead of changing the 'Client' instances, our implementation keeps the 'Client' instances and then updates their local dataset 
            """
            client_indexes = self._client_sampling(round_idx, self.args.client_num_in_total,
                                                   self.args.client_num_per_round)
            logging.info("client_indexes = " + str(client_indexes))

            for idx, client in enumerate(self.client_list):
                # update dataset
                client_idx = client_indexes[idx]
                client.update_local_dataset(client_idx, self.train_data_local_dict[client_idx],
                                            self.test_data_local_dict[client_idx],
                                            self.train_data_local_num_dict[client_idx])

                # train on new dataset
                w = client.train(copy.deepcopy(w_global))
                w_locals.append((client.get_sample_number(), copy.deepcopy(w)))

                # Save model
                client.model_trainer.save_model(self.save_model_dir)
                
                # Gather average Logits for offline logits
                # avg_logits += client.get_logits(self.unlabeled_train_data)

            # Initialize model fusion with aggregated w_global
            w_global = self._aggregate(w_locals)
            self.model_trainer.set_model_params(w_global)

            # update global weights with average logits
            avg_logits /= len(self.client_list)
            self._ensemble_distillation(round_idx, avg_logits)
            w_global = self.model_trainer.get_model_params()

            # test results
            # at last round
            if round_idx == self.args.comm_round - 1:
                self._local_test_on_all_clients(round_idx)
            # per {frequency_of_the_test} round
            elif round_idx % self.args.frequency_of_the_test == 0:
                if self.args.dataset.startswith("stackoverflow"):
                    self._local_test_on_validation_set(round_idx)
                else:
                    self._local_test_on_all_clients(round_idx)

    def _client_sampling(self, round_idx, client_num_in_total, client_num_per_round):
        if client_num_in_total == client_num_per_round:
            client_indexes = [client_index for client_index in range(client_num_in_total)]
        else:
            num_clients = min(client_num_per_round, client_num_in_total)
            np.random.seed(round_idx)  # make sure for each comparison, we are selecting the same clients each round
            client_indexes = np.random.choice(range(client_num_in_total), num_clients, replace=False)
        logging.info("client_indexes = %s" % str(client_indexes))
        self.model_trainer.client_indexes = client_indexes

        return client_indexes

    def _aggregate(self, w_locals):
        training_num = 0
        for idx in range(len(w_locals)):
            (sample_num, averaged_params) = w_locals[idx]
            training_num += sample_num

        (sample_num, averaged_params) = w_locals[0]
        for k in averaged_params.keys():
            for i in range(0, len(w_locals)):
                local_sample_number, local_model_params = w_locals[i]
                w = local_sample_number / training_num
                if i == 0:
                    averaged_params[k] = local_model_params[k] * w
                else:
                    averaged_params[k] += local_model_params[k] * w
        return averaged_params


    # def _generate_validation_set(self, num_samples=10000):
    #     test_data_num = len(self.test_global.dataset)
    #     sample_indices = random.sample(range(test_data_num), min(num_samples, test_data_num))
    #     subset = torch.utils.data.Subset(self.test_global.dataset, sample_indices)
    #     sample_testset = torch.utils.data.DataLoader(subset, batch_size=self.args.batch_size)
    #     self.val_global = sample_testset
        
    def _ensemble_distillation(self, round_idx, avg_logits):

        stats = {
            'round_idx' : round_idx,
            'server_val_acc' : 0
        }

        unlabeled_dataloader = self.unlabeled_train_data
        unlabeled_dataloader.dataset.target = avg_logits.detach().cpu().numpy()
        round_server_val_acc = self.model_trainer.train(unlabeled_dataloader, self.val_global, self.device, self.args)

        wandb.log({"Server/val/Acc": round_server_val_acc, "round": round_idx})
        stats['server_val_acc'] = round_server_val_acc
        logging.info(stats)


    def _local_test_on_all_clients(self, round_idx):

        logging.info("################local_test_on_all_clients : {}".format(round_idx))

        train_metrics = {
            'num_samples': [],
            'num_correct': [],
            'losses': []
        }

        test_metrics = {
            'num_samples': [],
            'num_correct': [],
            'losses': []
        }

        client = self.client_list[0]

        for client_idx in range(self.args.client_num_in_total):
            """
            Note: for datasets like "fed_CIFAR100" and "fed_shakespheare",
            the training client number is larger than the testing client number
            """
            if self.test_data_local_dict[client_idx] is None:
                continue
            client.update_local_dataset(0, self.train_data_local_dict[client_idx],
                                        self.test_data_local_dict[client_idx],
                                        self.train_data_local_num_dict[client_idx])
            # train data
            train_local_metrics = client.local_test(False)
            train_metrics['num_samples'].append(copy.deepcopy(train_local_metrics['test_total']))
            train_metrics['num_correct'].append(copy.deepcopy(train_local_metrics['test_correct']))
            train_metrics['losses'].append(copy.deepcopy(train_local_metrics['test_loss']))

            # test data
            test_local_metrics = client.local_test(True)
            test_metrics['num_samples'].append(copy.deepcopy(test_local_metrics['test_total']))
            test_metrics['num_correct'].append(copy.deepcopy(test_local_metrics['test_correct']))
            test_metrics['losses'].append(copy.deepcopy(test_local_metrics['test_loss']))

            """
            Note: CI environment is CPU-based computing. 
            The training speed for RNN training is to slow in this setting, so we only test a client to make sure there is no programming error.
            """
            if self.args.ci == 1:
                break

        # test on training dataset
        train_acc = sum(train_metrics['num_correct']) / sum(train_metrics['num_samples'])
        train_loss = sum(train_metrics['losses']) / sum(train_metrics['num_samples'])

        # test on test dataset
        test_acc = sum(test_metrics['num_correct']) / sum(test_metrics['num_samples'])
        test_loss = sum(test_metrics['losses']) / sum(test_metrics['num_samples'])

        stats = {'training_acc': train_acc, 'training_loss': train_loss}
        wandb.log({"Train/Acc": train_acc, "round": round_idx})
        wandb.log({"Train/Loss": train_loss, "round": round_idx})
        logging.info(stats)
        self._update_stats(stats)

        stats = {'test_acc': test_acc, 'test_loss': test_loss}
        wandb.log({"Test/Acc": test_acc, "round": round_idx})
        wandb.log({"Test/Loss": test_loss, "round": round_idx})
        logging.info(stats)
        self._update_stats(stats)

    def _local_test_on_validation_set(self, round_idx):

        logging.info("################local_test_on_validation_set : {}".format(round_idx))

        if self.val_global is None:
            self._generate_validation_set()

        client = self.client_list[0]
        client.update_local_dataset(0, None, self.val_global, None)
        # test data
        test_metrics = client.local_test(True)

        if self.args.dataset == "stackoverflow_nwp":
            test_acc = test_metrics['test_correct'] / test_metrics['test_total']
            test_loss = test_metrics['test_loss'] / test_metrics['test_total']
            stats = {'test_acc': test_acc, 'test_loss': test_loss}
            wandb.log({"Test/Acc": test_acc, "round": round_idx})
            wandb.log({"Test/Loss": test_loss, "round": round_idx})
        elif self.args.dataset == "stackoverflow_lr":
            test_acc = test_metrics['test_correct'] / test_metrics['test_total']
            test_pre = test_metrics['test_precision'] / test_metrics['test_total']
            test_rec = test_metrics['test_recall'] / test_metrics['test_total']
            test_loss = test_metrics['test_loss'] / test_metrics['test_total']
            stats = {'test_acc': test_acc, 'test_pre': test_pre, 'test_rec': test_rec, 'test_loss': test_loss}
            wandb.log({"Test/Acc": test_acc, "round": round_idx})
            wandb.log({"Test/Pre": test_pre, "round": round_idx})
            wandb.log({"Test/Rec": test_rec, "round": round_idx})
            wandb.log({"Test/Loss": test_loss, "round": round_idx})
        else:
            raise Exception("Unknown format to log metrics for dataset {}!" % self.args.dataset)

        logging.info(stats)