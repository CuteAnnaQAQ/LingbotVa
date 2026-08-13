# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
import os

from easydict import EasyDict

from .va_robomme_cfg import va_robomme_cfg


va_robomme_train_cfg = EasyDict(__name__='Config: VA RoboMME train')
va_robomme_train_cfg.update(va_robomme_cfg)

va_robomme_train_cfg.dataset_path = os.getenv(
    'ROBOMME_LEROBOT_DATASET_PATH', '/DATA/disk0/yry/robomme_data_lerobot')
va_robomme_train_cfg.empty_emb_path = os.path.join(
    va_robomme_train_cfg.dataset_path, 'empty_emb.pt')
va_robomme_train_cfg.require_action_stats = True
va_robomme_train_cfg.enable_wandb = os.getenv(
    'ROBOMME_ENABLE_WANDB', '1') == '1'
va_robomme_train_cfg.load_worker = int(os.getenv('ROBOMME_LOAD_WORKER', '16'))
va_robomme_train_cfg.save_interval = int(
    os.getenv('ROBOMME_SAVE_INTERVAL', '200'))
va_robomme_train_cfg.gc_interval = 50
va_robomme_train_cfg.cfg_prob = 0.1

va_robomme_train_cfg.learning_rate = 1e-5
va_robomme_train_cfg.beta1 = 0.9
va_robomme_train_cfg.beta2 = 0.95
va_robomme_train_cfg.weight_decay = 1e-1
va_robomme_train_cfg.warmup_steps = 10
va_robomme_train_cfg.batch_size = int(os.getenv('ROBOMME_BATCH_SIZE', '1'))
va_robomme_train_cfg.gradient_accumulation_steps = int(
    os.getenv('ROBOMME_GRADIENT_ACCUMULATION_STEPS', '32'))
va_robomme_train_cfg.num_steps = int(os.getenv('ROBOMME_NUM_STEPS', '50000'))
va_robomme_train_cfg.resume_from = os.getenv(
    'ROBOMME_RESUME_FROM', '').strip() or None
