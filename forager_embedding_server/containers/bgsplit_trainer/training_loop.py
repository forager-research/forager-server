import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import sklearn.metrics
import os
import os.path
import logging
import time
import math
import shutil
import scipy.special
import copy
from typing import Dict, List, Any, Callable
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter

from config import NUM_WORKERS
from model import Model
from dataset import AuxiliaryDataset
from warmup_scheduler import GradualWarmupScheduler
from util import download, EMA

logger = logging.getLogger("bgsplit")
logger.setLevel(logging.DEBUG)

class TrainingLoop():
    def __init__(
            self,
            model_kwargs,
            train_positive_paths,
            train_negative_paths,
            train_unlabeled_paths,
            val_positive_paths,
            val_negative_paths,
            val_unlabeled_paths,
            data_cache_dir: str,
            notify_callback: Callable[[Dict[str, Any]], None]=lambda x: None):
        '''The training loop for background splitting models.'''
        self.data_cache_dir = data_cache_dir
        self.notify_callback = notify_callback

        self._setup_model_kwargs(model_kwargs)

        # Setup dataset
        self._setup_dataset(
            train_positive_paths, train_negative_paths, train_unlabeled_paths,
            val_positive_paths, val_negative_paths, val_unlabeled_paths)

        # Setup model
        self._setup_model()

        # Setup optimizer

        # Resume if requested
        resume_from = model_kwargs.get('resume_from', None)
        if resume_from:
            resume_training = model_kwargs.get('resume_training', False)
            self.load_checkpoint(resume_from, resume_training=resume_training)

        self.writer = SummaryWriter(log_dir=model_kwargs['log_dir'])

        # Variables for estimating run-time
        self.train_batch_time = EMA(0)
        self.val_batch_time = EMA(0)
        self.train_batches_per_epoch = (
            len(self.train_dataloader.dataset) /
            self.train_dataloader.batch_size)
        self.val_batches_per_epoch = (
            len(self.val_dataloader.dataset) /
            self.val_dataloader.batch_size)
        self.train_batch_idx = 0
        self.val_batch_idx = 0
        self.train_epoch_loss = 0
        self.train_epoch_main_loss = 0
        self.train_epoch_aux_loss = 0

    def _setup_model_kwargs(self, model_kwargs):
        self.model_kwargs = copy.deepcopy(model_kwargs)
        self.num_workers = NUM_WORKERS
        self.val_frequency = model_kwargs.get('val_frequency', 1)
        self.checkpoint_frequency = model_kwargs.get('checkpoint_frequency', 1)
        self.use_cuda = bool(model_kwargs.get('use_cuda', True))
        assert 'model_dir' in model_kwargs
        self.model_dir = model_kwargs['model_dir']
        assert 'aux_labels' in model_kwargs
        self.aux_weight = float(model_kwargs.get('aux_weight', 0.1))
        assert 'log_dir' in model_kwargs

    def _setup_dataset(
            self,
            train_positive_paths, train_negative_paths, train_unlabeled_paths,
            val_positive_paths, val_negative_paths, val_unlabeled_paths):
        assert self.model_kwargs
        aux_labels = self.model_kwargs['aux_labels']
        image_input_size = self.model_kwargs.get('input_size', 224)
        batch_size = int(self.model_kwargs.get('batch_size', 64))
        num_workers = self.num_workers
        restrict_aux_labels = bool(self.model_kwargs.get('restrict_aux_labels', True))
        cache_images_on_disk = self.model_kwargs.get('cache_images_on_disk', False)

        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(image_input_size),
            transforms.RandomHorizontalFlip(),
            transforms.ConvertImageDtype(torch.float32),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ])
        resize_size = int(image_input_size * 1.15)
        resize_size += int(resize_size % 2)
        val_transform = transforms.Compose([
            transforms.Resize(resize_size),
            transforms.CenterCrop(image_input_size),
            transforms.ConvertImageDtype(torch.float32),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ])
        self.train_dataloader = DataLoader(
            AuxiliaryDataset(
                positive_paths=train_positive_paths,
                negative_paths=train_negative_paths,
                unlabeled_paths=train_unlabeled_paths,
                auxiliary_labels=aux_labels,
                restrict_aux_labels=restrict_aux_labels,
                cache_images_on_disk=cache_images_on_disk,
                data_cache_dir=self.data_cache_dir,
                transform=train_transform),
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers)
        self.val_dataloader = DataLoader(
            AuxiliaryDataset(
                positive_paths=val_positive_paths,
                negative_paths=val_negative_paths,
                unlabeled_paths=val_unlabeled_paths,
                auxiliary_labels=aux_labels,
                restrict_aux_labels=restrict_aux_labels,
                cache_images_on_disk=cache_images_on_disk,
                data_cache_dir=self.data_cache_dir,
                transform=val_transform),
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers)

    def _setup_model(self):
        num_classes = 2
        num_aux_classes = self.train_dataloader.dataset.num_auxiliary_classes
        freeze_backbone = self.model_kwargs.get('freeze_backbone', False)
        self.model_kwargs['num_aux_classes'] = num_aux_classes
        self.model = Model(num_main_classes=num_classes,
                           num_aux_classes=num_aux_classes,
                           freeze_backbone=freeze_backbone)
        if self.model_kwargs.get('aux_labels_type', None) == "imagenet":
            # Initialize auxiliary head to imagenet fc
            self.model.auxiliary_head.weight = self.model.backbone.fc.weight
            self.model.auxiliary_head.bias = self.model.backbone.fc.bias
        if self.use_cuda:
            self.model = self.model.cuda()
        self.model = nn.DataParallel(self.model)
        self.main_loss = nn.CrossEntropyLoss()
        self.auxiliary_loss = nn.CrossEntropyLoss()
        self.start_epoch = 0
        self.end_epoch = self.model_kwargs.get('epochs_to_run', 1)
        self.current_epoch = 0
        self.global_train_batch_idx = 0
        self.global_val_batch_idx = 0

        lr = float(self.model_kwargs.get('initial_lr', 0.01))
        endlr = float(self.model_kwargs.get('endlr', 0.0))
        optim_params = dict(
            lr=lr,
            momentum=float(self.model_kwargs.get('momentum', 0.9)),
            weight_decay=float(self.model_kwargs.get('weight_decay', 0.0001)),
        )
        self.optimizer = optim.SGD(self.model.parameters(), **optim_params)
        max_epochs = int(self.model_kwargs.get('max_epochs', 90))
        warmup_epochs = int(self.model_kwargs.get('warmup_epochs', 0))
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, max_epochs - warmup_epochs,
            eta_min=endlr)
        self.optimizer_scheduler = GradualWarmupScheduler(
            optimizer=self.optimizer,
            multiplier=1.0,
            warmup_epochs=warmup_epochs,
            after_scheduler=scheduler)

    def _notify(self):
        epochs_left = self.end_epoch - self.current_epoch - 1
        num_train_batches_left = (
            epochs_left * self.train_batches_per_epoch +
            max(0, self.train_batches_per_epoch - self.train_batch_idx - 1)
        )
        num_val_batches_left = (
            (1 + round(epochs_left / self.val_frequency)) * self.val_batches_per_epoch +
            max(0, self.val_batches_per_epoch - self.val_batch_idx - 1)
        )
        time_left = (
            num_train_batches_left * self.train_batch_time.value +
            num_val_batches_left * self.val_batch_time.value)
        self.notify_callback(**{"training_time_left": time_left})

    def setup_resume(
            self,
            train_positive_paths, train_negative_paths, train_unlabeled_paths,
            val_positive_paths, val_negative_paths, val_unlabeled_paths
    ):
        self._setup_dataset(train_positive_paths, train_negative_paths, train_unlabeled_paths,
                           val_positive_paths, val_negative_paths, val_unlabeled_paths)
        self.start_epoch = self.end_epoch
        self.current_epoch = self.start_epoch
        self.end_epoch = self.start_epoch + self.model_kwargs.get('epochs_to_run', 1)

    def load_checkpoint(self, path: str, resume_training: bool=False):
        checkpoint_state = torch.load(path)
        self.model.load_state_dict(checkpoint_state['state_dict'])
        if resume_training:
            self.global_train_batch_idx = checkpoint_state['global_train_batch_idx']
            self.global_val_batch_idx = checkpoint_state['global_val_batch_idx']
            self.start_epoch = checkpoint_state['epoch'] + 1
            self.current_epoch = self.start_epoch
            self.end_epoch = (
                self.start_epoch + self.model_kwargs.get('epochs_to_run', 1))
            self.optimizer.load_state_dict(
                checkpoint_state['optimizer'])
            self.optimizer_scheduler.load_state_dict(
                checkpoint_state['optimizer_scheduler'])
            # Copy tensorboard state
            prev_log_dir = checkpoint_state['model_kwargs']['log_dir']
            curr_log_dir = self.model_kwargs['log_dir']
            shutil.copytree(prev_log_dir, curr_log_dir)

    def save_checkpoint(self, epoch, checkpoint_path: str):
        kwargs = dict(self.model_kwargs)
        del kwargs['aux_labels']
        state = dict(
            global_train_batch_idx=self.global_train_batch_idx,
            global_val_batch_idx=self.global_val_batch_idx,
            model_kwargs=kwargs,
            epoch=epoch,
            state_dict=self.model.state_dict(),
            optimizer=self.optimizer.state_dict(),
            optimizer_scheduler=self.optimizer_scheduler.state_dict(),
        )
        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
        torch.save(state, checkpoint_path)

    def _validate(self, dataloader):
        self.model.eval()
        loss_value = 0
        main_gts = []
        aux_gts = []
        main_preds = []
        aux_preds = []
        for batch_idx, (images, main_labels, aux_labels) in enumerate(
                dataloader):
            batch_start = time.perf_counter()
            self.val_batch_idx = batch_idx
            if self.use_cuda:
                images = images.cuda()
                main_labels = main_labels.cuda()
                aux_labels = aux_labels.cuda()
            main_logits, aux_logits = self.model(images)
            valid_main_labels = main_labels != -1
            valid_aux_labels = aux_labels != -1
            main_loss_value = self.main_loss(
                main_logits[valid_main_labels],
                main_labels[valid_main_labels])
            aux_loss_value = self.aux_weight * self.auxiliary_loss(
                aux_logits[valid_aux_labels],
                aux_labels[valid_aux_labels])
            loss_value = torch.zeros_like(main_loss_value)
            if valid_main_labels.sum() > 0:
                loss_value += main_loss_value
            if valid_aux_labels.sum() > 0:
                loss_value += aux_loss_value
            loss_value = loss_value.item()
 
            if valid_main_labels.sum() > 0:
                main_pred = F.softmax(main_logits[valid_main_labels])
                main_preds += list(main_pred.argmax(dim=1)[valid_main_labels].cpu().numpy())
                main_gts += list(main_labels[valid_main_labels].cpu().numpy())
            if valid_aux_labels.sum() > 0:
                aux_pred = F.softmax(main_logits[valid_main_labels])
                aux_preds += list(aux_pred.argmax(dim=1)[valid_aux_labels].cpu().numpy())
                aux_gts += list(aux_labels[valid_aux_labels].cpu().numpy())
            batch_end = time.perf_counter()
            self.val_batch_time += (batch_end - batch_start)
            self.global_val_batch_idx += 1
        # Compute F1 score
        if len(dataloader) > 0:
            loss_value /= (len(dataloader) + 1e-10)
            main_prec, main_recall, main_f1, _ = \
                sklearn.metrics.precision_recall_fscore_support(
                    main_gts, main_preds, average='binary')
            aux_prec, aux_recall, aux_f1, _ = \
                sklearn.metrics.precision_recall_fscore_support(
                    aux_gts, aux_preds, average='micro')
        else:
            loss_value = 0
            main_prec = -1
            main_recall = -1
            main_f1 = -1
            aux_prec = -1
            aux_recall = -1
            aux_f1 = -1

        summary_data = [
            ('loss', loss_value),
            ('f1/main_head', main_f1),
            ('prec/main_head', main_prec),
            ('recall/main_head', main_recall),
            ('f1/aux_head', aux_f1),
            ('prec/aux_head', aux_prec),
            ('recall/aux_head', aux_recall),
        ]
        for k, v in [('val/epoch/' + tag, v) for tag, v in summary_data]:
            self.writer.add_scalar(k, v, self.current_epoch)

    def validate(self):
        self._validate(self.val_dataloader)

    def train(self):
        self.model.train()
        logger.info('Starting train epoch')
        load_start = time.perf_counter()
        self.train_epoch_loss = 0
        self.train_epoch_main_loss = 0
        self.train_epoch_aux_loss = 0
        main_gts = []
        aux_gts = []
        main_logits_all = []
        main_preds = []
        aux_preds = []
        for batch_idx, (images, main_labels, aux_labels) in enumerate(
                self.train_dataloader):
            load_end = time.perf_counter()
            batch_start = time.perf_counter()
            self.train_batch_idx = batch_idx
            logger.debug('Train batch')
            if self.use_cuda:
                images = images.cuda()
                main_labels = main_labels.cuda()
                aux_labels = aux_labels.cuda()

            main_logits, aux_logits = self.model(images)
            # Compute loss
            valid_main_labels = main_labels != -1
            valid_aux_labels = aux_labels != -1

            main_loss_value = self.main_loss(
                main_logits[valid_main_labels],
                main_labels[valid_main_labels])
            aux_loss_value = self.aux_weight * self.auxiliary_loss(
                aux_logits[valid_aux_labels],
                aux_labels[valid_aux_labels])

            loss_value = torch.zeros_like(main_loss_value)
            if valid_main_labels.sum() > 0:
                loss_value += main_loss_value
            if valid_aux_labels.sum() > 0:
                loss_value += aux_loss_value

            self.train_epoch_loss += loss_value.item()
            if torch.sum(valid_main_labels) > 0:
                self.train_epoch_main_loss += main_loss_value.item()
            if torch.sum(valid_aux_labels) > 0:
                self.train_epoch_aux_loss += aux_loss_value.item()
            # Update gradients
            self.optimizer.zero_grad()
            loss_value.backward()
            self.optimizer.step()

            if valid_main_labels.sum() > 0:
                main_pred = F.softmax(main_logits[valid_main_labels], dim=1)
                main_logits_all += list(main_logits[valid_main_labels].detach().cpu().numpy())
                main_preds += list(main_pred[valid_main_labels].argmax(dim=1).cpu().numpy())
                main_gts += list(main_labels[valid_main_labels].cpu().numpy())
            if valid_aux_labels.sum() > 0:
                aux_pred = F.softmax(aux_logits[valid_aux_labels], dim=1)
                aux_preds += list(aux_pred[valid_aux_labels].argmax(dim=1).cpu().numpy())
                aux_gts += list(aux_labels[valid_aux_labels].cpu().numpy())

            batch_end = time.perf_counter()
            total_batch_time = (batch_end - batch_start)
            total_load_time = (load_end - load_start)
            self.train_batch_time += total_batch_time + total_load_time
            logger.debug(f'Train batch time: {self.train_batch_time.value}, '
                         f'this batch time: {total_batch_time}, '
                         f'this load time: {total_load_time}, '
                         f'batch epoch loss: {loss_value.item()}, '
                         f'main loss: {main_loss_value.item()}, '
                         f'aux loss: {aux_loss_value.item()}')
            summary_data = [
                ('loss', loss_value.item()),
                ('loss/main_head', main_loss_value.item()),
                ('loss/aux_head', aux_loss_value.item()),
            ]
            for k, v in [('train/batch/' + tag, v) for tag, v in summary_data]:
                self.writer.add_scalar(k, v, self.global_train_batch_idx)

            self._notify()
            self.global_train_batch_idx += 1
            load_start = time.perf_counter()

        model_lr = self.optimizer.param_groups[-1]['lr']
        self.optimizer_scheduler.step()
        logger.debug(f'Train epoch loss: {self.train_epoch_loss}, '
                     f'main loss: {self.train_epoch_main_loss}, '
                     f'aux loss: {self.train_epoch_aux_loss}')
        main_prec, main_recall, main_f1, _ = \
            sklearn.metrics.precision_recall_fscore_support(
                main_gts, main_preds, average='binary')
        aux_prec, aux_recall, aux_f1, _ = \
            sklearn.metrics.precision_recall_fscore_support(
                aux_gts, aux_preds, average='micro')
        logger.debug(f'Train epoch main: {main_prec}, {main_recall}, {main_f1}, '
                     f'aux: {aux_prec}, {aux_recall}, {aux_f1}'
                     f'main loss: {self.train_epoch_main_loss}, '
                     f'aux loss: {self.train_epoch_aux_loss}')
        summary_data = [
            ('lr', model_lr),
            ('loss', self.train_epoch_loss),
            ('loss/main_head', self.train_epoch_main_loss),
            ('loss/aux_head', self.train_epoch_aux_loss),
            ('f1/main_head', main_f1),
            ('prec/main_head', main_prec),
            ('recall/main_head', main_recall),
            ('f1/aux_head', aux_f1),
            ('prec/aux_head', aux_prec),
            ('recall/aux_head', aux_recall)
        ]
        for k, v in [('train/epoch/' + tag , v) for tag, v in summary_data]:
            self.writer.add_scalar(k, v, self.current_epoch)

        if len(main_logits_all):
            self.writer.add_histogram(
                'train/epoch/softmax/main_head',
                scipy.special.softmax(main_logits_all, axis=1)[:, 1])

    def run(self):
        self.last_checkpoint_path = None
        for i in range(self.start_epoch, self.end_epoch):
            logger.info(f'Train: Epoch {i}')
            self.current_epoch = i
            self.train()
            if i % self.val_frequency == 0 or i == self.end_epoch - 1:
                logger.info(f'Validate: Epoch {i}')
                self.validate()
            if i % self.checkpoint_frequency == 0 or i == self.end_epoch - 1:
                logger.info(f'Checkpoint: Epoch {i}')
                self.last_checkpoint_path = os.path.join(
                    self.model_dir, f'checkpoint_{i:03}.pth')
                self.save_checkpoint(i, self.last_checkpoint_path)
        return self.last_checkpoint_path
