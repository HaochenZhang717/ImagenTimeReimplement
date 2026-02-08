# """Time-series Generative Adversarial Networks (TimeGAN) Codebase.
#
# Reference: Jinsung Yoon, Daniel Jarrett, Mihaela van der Schaar,
# "Time-series Generative Adversarial Networks,"
# Neural Information Processing Systems (NeurIPS), 2019.
#
# Paper link: https://papers.nips.cc/paper/8789-time-series-generative-adversarial-networks
#
# Last updated Date: April 24th 2020
# Code author: Jinsung Yoon (jsyoon0823@gmail.com)
#
# -----------------------------
#
# predictive_metrics.py
#
# Note: Use Post-hoc RNN to predict one-step ahead (last feature)
# """
#
# # Necessary Packages
# import tensorflow as tf
# import numpy as np
# from sklearn.metrics import mean_absolute_error
# from utils.utils import extract_time
#
# tf.compat.v1.disable_eager_execution()
#
#
# def predictive_score_metrics(ori_data, generated_data):
#     """Report the performance of Post-hoc RNN one-step ahead prediction.
#
#     Args:
#       - ori_data: original data
#       - generated_data: generated synthetic data
#
#     Returns:
#       - predictive_score: MAE of the predictions on the original data
#     """
#     # Initialization on the Graph
#     tf.compat.v1.reset_default_graph()
#
#     # Basic Parameters
#     no, seq_len, dim = np.asarray(ori_data).shape
#
#     # Set maximum sequence length and each sequence length
#     ori_time, ori_max_seq_len = extract_time(ori_data)
#     generated_time, generated_max_seq_len = extract_time(ori_data)
#     max_seq_len = max([ori_max_seq_len, generated_max_seq_len])
#
#     ## Builde a post-hoc RNN predictive network
#     # Network parameters
#     hidden_dim = int(dim / 2)
#     iterations = 5000
#     batch_size = 128
#
#     # Input place holders
#     X = tf.compat.v1.placeholder(tf.float32, [None, max_seq_len - 1, dim - 1], name="myinput_x")
#     T = tf.compat.v1.placeholder(tf.int32, [None], name="myinput_t")
#     Y = tf.compat.v1.placeholder(tf.float32, [None, max_seq_len - 1, 1], name="myinput_y")
#
#     # Predictor function
#     def predictor(x, t):
#         """Simple predictor function.
#
#         Args:
#           - x: time-series data
#           - t: time information
#
#         Returns:
#           - y_hat: prediction
#           - p_vars: predictor variables
#         """
#         with tf.compat.v1.variable_scope("predictor", reuse=tf.compat.v1.AUTO_REUSE) as vs:
#             p_cell = tf.compat.v1.nn.rnn_cell.GRUCell(num_units=hidden_dim, activation=tf.nn.tanh, name='p_cell')
#             p_outputs, p_last_states = tf.compat.v1.nn.dynamic_rnn(p_cell, x, dtype=tf.float32, sequence_length=t)
#             y_hat_logit = tf.compat.v1.layers.dense(p_outputs, 1, activation=None)
#             y_hat = tf.nn.sigmoid(y_hat_logit)
#             p_vars = [v for v in tf.compat.v1.all_variables() if v.name.startswith(vs.name)]
#
#         return y_hat, p_vars
#
#     y_pred, p_vars = predictor(X, T)
#     # Loss for the predictor
#     p_loss = tf.compat.v1.losses.absolute_difference(Y, y_pred)
#     # optimizer
#     p_solver = tf.compat.v1.train.AdamOptimizer().minimize(p_loss, var_list=p_vars)
#
#     ## Training
#     # Session start
#     sess = tf.compat.v1.Session()
#     sess.run(tf.compat.v1.global_variables_initializer())
#
#     # Training using Synthetic data
#     for itt in range(iterations):
#         # Set mini-batch
#         idx = np.random.permutation(len(generated_data))
#         train_idx = idx[:batch_size]
#
#         X_mb = list(generated_data[i][:-1, :(dim - 1)] for i in train_idx)
#         T_mb = list(generated_time[i] - 1 for i in train_idx)
#         Y_mb = list(
#             np.reshape(generated_data[i][1:, (dim - 1)], [len(generated_data[i][1:, (dim - 1)]), 1]) for i in train_idx)
#
#         # Train predictor
#         _, step_p_loss = sess.run([p_solver, p_loss], feed_dict={X: X_mb, T: T_mb, Y: Y_mb})
#
#         ## Test the trained model on the original data
#     idx = np.random.permutation(len(ori_data))
#     train_idx = idx[:no]
#
#     X_mb = list(ori_data[i][:-1, :(dim - 1)] for i in train_idx)
#     T_mb = list(ori_time[i] - 1 for i in train_idx)
#     Y_mb = list(np.reshape(ori_data[i][1:, (dim - 1)], [len(ori_data[i][1:, (dim - 1)]), 1]) for i in train_idx)
#
#     # Prediction
#     pred_Y_curr = sess.run(y_pred, feed_dict={X: X_mb, T: T_mb})
#
#     # Compute the performance in terms of MAE
#     MAE_temp = 0
#     for i in range(no):
#         MAE_temp = MAE_temp + mean_absolute_error(Y_mb[i], pred_Y_curr[i, :, :])
#
#     predictive_score = MAE_temp / no
#
#     return predictive_score

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import mean_absolute_error
from utils.utils import extract_time



class GRUPredictor(nn.Module):
    def __init__(self, input_dim, hidden_dim, use_sigmoid=True):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)
        self.use_sigmoid = use_sigmoid
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, lengths):
        """
        x: (B, T, input_dim)
        lengths: (B,)
        """
        packed = nn.utils.rnn.pack_padded_sequence(
            x, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        packed_out, _ = self.gru(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True)

        y_hat = self.fc(out)
        if self.use_sigmoid:
            y_hat = self.sigmoid(y_hat)

        return y_hat


def predictive_score_metrics(
    ori_data,
    generated_data,
    iterations=5000,
    batch_size=128,
    lr=1e-3,
    device="cuda",
    use_sigmoid=True,
):
    """
    PyTorch implementation of TimeGAN predictive_score_metrics.

    Args:
        ori_data: list/np array, (N, T, D)
        generated_data: list/np array, (N, T, D)
        extract_time: your existing extract_time function
    Returns:
        predictive_score: MAE on original data
    """

    # Convert to list of np arrays for compatibility with extract_time
    ori_data = list(np.asarray(x, dtype=np.float32) for x in ori_data)
    generated_data = list(np.asarray(x, dtype=np.float32) for x in generated_data)

    no, _, dim = np.asarray(ori_data).shape

    # Use your extract_time
    ori_time, ori_max_seq_len = extract_time(ori_data)
    generated_time, generated_max_seq_len = extract_time(generated_data)
    max_seq_len = max([ori_max_seq_len, generated_max_seq_len])

    hidden_dim = int(dim / 2)

    model = GRUPredictor(input_dim=dim - 1, hidden_dim=hidden_dim, use_sigmoid=use_sigmoid).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # loss per element (so we can mask)
    loss_fn = nn.L1Loss(reduction="none")

    # ==========================
    # Training on generated data
    # ==========================
    model.train()

    for itt in range(iterations):
        idx = np.random.permutation(len(generated_data))
        train_idx = idx[:batch_size]

        X_list, Y_list, T_list = [], [], []

        for i in train_idx:
            seq = generated_data[i]  # (T, D)

            # one-step prediction setup (T-1 steps)
            X_i = seq[:-1, :(dim - 1)]          # (T-1, dim-1)
            Y_i = seq[1:, (dim - 1):(dim)]      # (T-1, 1)
            T_i = generated_time[i] - 1         # scalar

            X_list.append(X_i)
            Y_list.append(Y_i)
            T_list.append(T_i)

        # pad to max length in this batch
        max_len = max(T_list)
        B = len(train_idx)

        X_pad = np.zeros((B, max_len, dim - 1), dtype=np.float32)
        Y_pad = np.zeros((B, max_len, 1), dtype=np.float32)

        for b in range(B):
            L = T_list[b]
            X_pad[b, :L] = X_list[b]
            Y_pad[b, :L] = Y_list[b]

        X_pad = torch.tensor(X_pad, device=device)
        Y_pad = torch.tensor(Y_pad, device=device)
        T_tensor = torch.tensor(T_list, device=device)

        pred = model(X_pad, T_tensor)

        # mask out padded positions
        mask = (torch.arange(max_len, device=device)[None, :] < T_tensor[:, None]).float()
        mask = mask.unsqueeze(-1)  # (B, T, 1)

        loss = loss_fn(pred, Y_pad)
        loss = (loss * mask).sum() / mask.sum()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # ======================
    # Testing on ori_data
    # ======================
    model.eval()

    MAE_temp = 0.0
    with torch.no_grad():
        for i in range(no):
            seq = ori_data[i]
            T_i = ori_time[i] - 1

            X_i = seq[:-1, :(dim - 1)]
            Y_i = seq[1:, (dim - 1):(dim)]

            X_tensor = torch.tensor(X_i[None, :, :], device=device)
            T_tensor = torch.tensor([T_i], device=device)

            pred = model(X_tensor, T_tensor)  # (1, T-1, 1)
            pred = pred.squeeze(0).cpu().numpy()  # (T-1, 1)

            MAE_temp += mean_absolute_error(Y_i, pred)

    predictive_score = MAE_temp / no
    return predictive_score