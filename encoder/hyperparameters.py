# Mel-filterbank
mel_window_length = 25 # In milliseconds
mel_window_step = 10 # In milliseconds
mel_n_channels = 40

# Model
hidden_nodes = 768
projection_size = 256
num_layers = 3

speakers_per_batch = 64
utterances_per_speaker = 10

initial_learning_rate = 0.01
steps_per_halving = 30_000_000

l2_norm_clip = 3

projection_gradient_scale = 0.5

# Apparently this helped smooth convergence
scaling_factor_loss_w_init = 10
scaling_factor_loss_b_init = -5