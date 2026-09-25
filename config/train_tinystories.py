out_dir = 'out-tinystories'
init_from = 'scratch'
eval_interval = 1000
eval_iters = 100
log_interval = 50
always_save_checkpoint = True
wandb_log = False

dataset = 'tinystories'
gradient_accumulation_steps = 1
batch_size = 128
block_size = 256

n_layer = 6
n_head = 4
n_embd = 256
dropout = 0.1
bias = False

learning_rate = 6e-4
max_iters = 100000
lr_decay_iters = 100000
min_lr = 6e-5
beta2 = 0.95
warmup_iters = 1000
weight_decay = 1e-1
