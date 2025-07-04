# Filename: network.py
# License: LICENSES/LICENSE_UVIC_EPFL
import os
import sys

import logging 
logging.disable(logging.WARNING) 
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import numpy as np
# import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

from parse import parse
from tqdm import trange
import pdb
import itertools

from .tf_utils import pre_x_in, topk
from .ops import tf_skew_symmetric 
from .tests import test_process

class MyNetwork(object):
    """Network class """

    def __init__(self, config):

        self.config = config

        # Initialize thensorflow session
        self._init_tensorflow()

        # Build the network
        self._build_placeholder()
        self._build_preprocessing()
        self._build_model()
        # self._build_loss()
        self._build_optim()
        self._build_summary()
        self._build_writer()

    def _init_tensorflow(self):
        # limit CPU threads with OMP_NUM_THREADS
        num_threads = os.getenv("OMP_NUM_THREADS", "")
        if num_threads != "":
            num_threads = int(num_threads)
            # print("limiting tensorflow to {} threads!".format(num_threads))
            # Limit
            tfconfig = tf.ConfigProto(
                intra_op_parallelism_threads=num_threads,
                inter_op_parallelism_threads=num_threads,
            )
        else:
#           tfconfig = tf.compat.v1.ConfigProto() 
            tfconfig = tf.ConfigProto()

        tfconfig.gpu_options.allow_growth = True

#       self.sess = tf.compat.v1.Session()
        self.sess = tf.Session(config=tfconfig)

    def _build_placeholder(self):
        """Build placeholders."""

        # Make tensforflow placeholder
        self.x_in = tf.placeholder(tf.float32, [None, 1, None, 4], name="x_in")
        self.y_in = tf.placeholder(tf.float32, [None, None, 2], name="y_in")
        self.R_in = tf.placeholder(tf.float32, [None, 9], name="R_in")
        self.t_in = tf.placeholder(tf.float32, [None, 3], name="t_in")
        self.is_training = tf.placeholder(tf.bool, (), name="is_training")
        # Input uncalibration and normalization parameters
        self.T1_in = tf.placeholder(tf.float32, [None, 3, 3], name="T1_in") # norm mat
        self.T2_in = tf.placeholder(tf.float32, [None, 3, 3], name="T2_in") # norm mat
        self.K1_in = tf.placeholder(tf.float32, [None, 3, 3], name="K1_in") # calib mat
        self.K2_in = tf.placeholder(tf.float32, [None, 3, 3], name="K2_in") # calib mat

        # Global step for optimization
        self.global_step = tf.get_variable(
            "global_step", shape=(),
            initializer=tf.zeros_initializer(),
            dtype=tf.int64,
            trainable=False)

    def _build_preprocessing(self):
        """Build preprocessing related graph."""

        # For now, do nothing
        pass

    def _build_model(self):
        """Build our MLP network."""

        with tf.variable_scope("Matchnet", reuse=tf.AUTO_REUSE):
            # For intermediate visualization 
            self.fetch_vis = {}
            # -------------------- Network archintecture --------------------
            # Import correct build_graph function
            from .cvpr2020 import build_graph
            # Build graph
            # print("Building Graph")
            # Preprocessing input, currently doing nothing
            x_in = pre_x_in(self.x_in, self.config.pre_x_in)
            y_in = self.y_in
            self.fetch_vis["x_in"] = self.x_in
            self.fetch_vis["y_in"] = self.y_in

            logits = []
            indexs = []
            e_hats = []
            losses = []
            # Framework for iterative top-k strategy.
            # We currently disable iterative top-k by set num_phase=1.
            for i in range(self.config.num_phase):
                # Weight local is the wegiht matrix for incorporating locality into network
                # But we currently disable it by set it as None.
                weight_local = None 
                x_shp = tf.shape(x_in)
                logit, vis_dict = build_graph(x_in, self.is_training, self.config, weight_local)
                tf.summary.histogram("logit", logit)
                self.fetch_vis = {**self.fetch_vis, **vis_dict} # For visualizing intermediate layers
                self.fetch_vis["logits"] = logit[:, None, :, None]
                
                self.bool_use_weight_for_score = self.config.bool_use_weight_for_score 

                # Support different output weight for 8-point algorithm
                if self.config.weight_opt == "relu_tanh":
                    weights = tf.nn.relu(tf.tanh(logit))
                elif self.config.weight_opt == "sigmoid_softmax":
                    logit_softmax = vis_dict["logit_softmax"]
                    self.logit_softmax = logit_softmax
                    mask = tf.nn.sigmoid(logit)
                    if self.config.bool_hard_attention:
                        mask = tf.cast(logit > 0, tf.float32)
                    weights = tf.exp(logit_softmax) * mask
                    weights = weights / tf.reduce_sum(weights, -1, keepdims=True) 
                else:
                    raise ValueError("Don't support it")


                # Make input data (num_img_pair x num_corr x 4)
                xx = tf.transpose(tf.reshape(
                    x_in, (x_shp[0], x_shp[2], 4)), (0, 2, 1))

                # Create the matrix to be used for the eight-point algorithm
                X = tf.transpose(tf.stack([
                    xx[:, 2] * xx[:, 0], xx[:, 2] * xx[:, 1], xx[:, 2],
                    xx[:, 3] * xx[:, 0], xx[:, 3] * xx[:, 1], xx[:, 3],
                    xx[:, 0], xx[:, 1], tf.ones_like(xx[:, 0])
                ], axis=1), (0, 2, 1))
                self.fetch_vis["X"] = X[:, None]
                # print("X shape = {}".format(X.shape))
                wX = tf.reshape(weights, (x_shp[0], x_shp[2], 1)) * X
                # print("wX shape = {}".format(wX.shape))
                XwX = tf.matmul(tf.transpose(X, (0, 2, 1)), wX)
                # print("XwX shape = {}".format(XwX.shape))

                # Recover essential matrix from self-adjoing eigen
                e, v = tf.self_adjoint_eig(XwX)
                e_hat = tf.reshape(v[:, :, 0], (x_shp[0], 9))
                # in case you want to directly output F
                self.out_e_hat = e_hat

                if self.config.use_fundamental > 0:
                    # Go back Essential Matrix with input norm and calibration matrix
                    e_hat = tf.reshape(e_hat, (x_shp[0], 3, 3)) 
                    e_hat = tf.matmul(
                        tf.matmul(tf.transpose(self.T2_in, (0, 2, 1)), e_hat), self.T1_in)
                    e_hat = tf.matmul(
                        tf.matmul(tf.transpose(self.K2_in, (0, 2, 1)), e_hat), self.K1_in)
                    e_hat = tf.reshape(e_hat, (x_shp[0], 9))

                e_hat /= tf.norm(e_hat, axis=1, keepdims=True)
                last_e_hat = e_hat
                last_logit = logit
                last_x_in = x_in
                last_weights = weights

                e_hats += [e_hat]
                losses += [self._build_loss(e_hat, logit, x_in, y_in, weights, name=str(i))]
                logits += [logit]
                num_top_k = tf.cast(x_shp[2] * 5 / 10, tf.int32) # top 50% points
                # update x_in and y_in according to the logit
                x_in, index = topk(x_in, logit[:, None], num_top_k)
                y_in = tf.squeeze(tf.gather_nd(y_in[:, None], index), 1)
                indexs += [index]
            # L2 loss
            for var in tf.trainable_variables():
                if "weights" in var.name:
                    # print(var.name)
                    tf.add_to_collection("l2_losses", tf.reduce_sum(var**2))
            l2_loss = tf.add_n(tf.get_collection("l2_losses"))
            tf.summary.scalar("l2_loss", l2_loss)
            # Check global_step and add essential loss
            loss = self.config.loss_decay * l2_loss
            self.loss = loss + tf.reduce_mean(tf.stack(losses))
            # repalce self.logit and self.e_hat with self.last_e_hat, 
            # self.last_logit, self.last_x_in
            self.e_hat = None
            self.logits = None
            self.last_e_hat = last_e_hat
            self.last_logit = last_logit
            self.last_x_in = last_x_in
            self.last_weights = last_weights
    
    def _build_loss(self, e_hat, logit, x_in, y_in, weights, name=""):
        """Build our cross entropy loss."""

        with tf.variable_scope("Loss_{}".format(name), reuse=tf.AUTO_REUSE):
            x_shp = tf.shape(self.x_in)
            # The groundtruth epi sqr
            gt_geod_d = y_in[:, :, 0]
            # tf.summary.histogram("gt_geod_d", gt_geod_d)

            # Get groundtruth Essential matrix
            e_gt_unnorm = tf.reshape(tf.matmul(
                tf.reshape(tf_skew_symmetric(self.t_in), (x_shp[0], 3, 3)),
                tf.reshape(self.R_in, (x_shp[0], 3, 3))
            ), (x_shp[0], 9))
            e_gt = e_gt_unnorm / tf.norm(e_gt_unnorm, axis=1, keepdims=True)

            # e_hat = tf.reshape(tf.matmul(
            #     tf.reshape(t_hat, (-1, 3, 3)),
            #     tf.reshape(r_hat, (-1, 3, 3))
            # ), (-1, 9))

            # Essential matrix loss
            essential_loss = tf.reduce_mean(tf.minimum(
                tf.reduce_sum(tf.square(e_hat - e_gt), axis=1),
                tf.reduce_sum(tf.square(e_hat + e_gt), axis=1)
            ))
            
            tf.summary.scalar("essential_loss", essential_loss)

            # Classification loss
            is_pos = tf.cast(
                gt_geod_d < self.config.obj_geod_th, tf.float32
            )
            is_neg = tf.cast(
                gt_geod_d >= self.config.obj_geod_th, tf.float32
            )
            c = is_pos - is_neg

            classif_losses = -tf.log(tf.nn.sigmoid(c * logit))

            # balance
            num_pos = tf.nn.relu(tf.reduce_sum(is_pos, axis=1) - 1.0) + 1.0
            num_neg = tf.nn.relu(tf.reduce_sum(is_neg, axis=1) - 1.0) + 1.0
            classif_loss_p = tf.reduce_sum(
                classif_losses * is_pos, axis=1
            )
            classif_loss_n = tf.reduce_sum(
                classif_losses * is_neg, axis=1
            )


            classif_loss = tf.reduce_mean(
                classif_loss_p * 0.5 / num_pos +
                classif_loss_n * 0.5 / num_neg
            )
            tf.summary.scalar("classif_loss", classif_loss)
            tf.summary.scalar(
                "classif_loss_p",
                tf.reduce_mean(classif_loss_p * 0.5 / num_pos))
            tf.summary.scalar(
                "classif_loss_n",
                tf.reduce_mean(classif_loss_n * 0.5 / num_neg))
            precision = tf.reduce_mean(
                tf.reduce_sum(tf.cast(logit > 0, tf.float32) * is_pos, axis=1) /
                tf.reduce_sum(tf.cast(logit > 0, tf.float32) *
                              (is_pos + is_neg), axis=1)
            )
            tf.summary.scalar("precision", precision)
            recall = tf.reduce_mean(
                tf.reduce_sum(tf.cast(logit > 0, tf.float32) * is_pos, axis=1) /
                tf.reduce_sum(is_pos, axis=1)
            )
            tf.summary.scalar("recall", recall)
            self.precision = precision
            self.recall = recall

            loss = 0
            
            if self.config.loss_essential > 0:
                loss += (
                    self.config.loss_essential * essential_loss * tf.cast(
                        self.global_step >= tf.cast(
                            self.config.loss_essential_init_iter, tf.int64), tf.float32))
            if self.config.loss_classif > 0:
                loss += self.config.loss_classif * classif_loss

            if self.config.loss_multi_logit > 0:
                # init value is 0
                th_logit = self.config.th_logit
                classif_multi_logit = []
                logit_attentions = tf.get_collection("logit_attention") #  Get logits for local attention
                for logit_attention in logit_attentions:
                    # print("attention : {}".format(logit_attention.name))
                    logit_i = tf.squeeze(logit_attention - th_logit, [1, 3])
                    gt_geod_d = y_in[:, :, 0]
                    is_pos = tf.cast(gt_geod_d < self.config.obj_geod_th, tf.float32)
                    is_neg = tf.cast(gt_geod_d >= self.config.obj_geod_th, tf.float32)
                    c = is_pos - is_neg
                    classif_losses = -tf.log_sigmoid(c * logit_i)
                    num_pos = tf.nn.relu(tf.reduce_sum(is_pos, axis=1) - 1.0) + 1.0
                    num_neg = tf.nn.relu(tf.reduce_sum(is_neg, axis=1) - 1.0) + 1.0
                    classif_loss_p = tf.reduce_sum(
                        classif_losses * is_pos, axis=1
                    )
                    classif_loss_n = tf.reduce_sum(
                        classif_losses * is_neg, axis=1
                    )
                    classif_loss = tf.reduce_mean(
                        classif_loss_p * 0.5 / num_pos +
                        classif_loss_n * 0.5 / num_neg
                    )
                    classif_multi_logit += [classif_loss]
                classif_multi_logit = tf.reduce_mean(tf.stack(classif_multi_logit))
                loss += classif_multi_logit * self.config.loss_multi_logit

            tf.summary.scalar("loss", loss)
            return loss
            

    def _build_optim(self):
        """Build optimizer related ops and vars."""

        with tf.variable_scope("Optimization", reuse=tf.AUTO_REUSE):
            learning_rate = self.config.train_lr
            max_grad_norm = None
            optim = tf.train.AdamOptimizer(learning_rate)
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                grads_and_vars = optim.compute_gradients(self.loss)

                # gradient clipping
                if max_grad_norm is not None:
                    new_grads_and_vars = []
                    for idx, (grad, var) in enumerate(grads_and_vars):
                        if grad is not None:
                            new_grads_and_vars.append((
                                tf.clip_by_norm(grad, max_grad_norm), var))
                    grads_and_vars = new_grads_and_vars

                # Check numerics and report if something is going on. This
                # will make the backward pass stop and skip the batch
                new_grads_and_vars = []
                for idx, (grad, var) in enumerate(grads_and_vars):
                    if grad is not None:
                        grad = tf.check_numerics(
                            grad, "Numerical error in gradient for {}"
                            "".format(var.name))
                    new_grads_and_vars.append((grad, var))

                # Should only apply grads once they are safe
                self.optim = optim.apply_gradients(
                    new_grads_and_vars, global_step=self.global_step)

            # # Summarize all gradients
            # for grad, var in grads_and_vars:
            #     if grad is not None:
            #         tf.summary.histogram(var.name + '/gradient', grad)

    def _build_summary(self):
        """Build summary ops."""

        # Merge all summary op
        self.summary_op = tf.summary.merge_all()

    def _build_writer(self):
        """Build the writers and savers"""

        # Create suffix automatically if not provided
        # suffix_tr = self.config.log_dir
        # if suffix_tr == "":
        #     suffix_tr = "-".join(sys.argv)
        # suffix_te = self.config.test_log_dir
        # if suffix_te == "":
        #     suffix_te = suffix_tr

        ppath = os.path.join(os.path.split(__file__)[0], 'logs')

        # Directories for train/test
        # self.res_dir_tr = os.path.join(self.config.res_dir, suffix_tr)[:-3]
        # self.res_dir_va = os.path.join(self.config.res_dir, suffix_te)[:-3]
        # self.res_dir_te = os.path.join(self.config.res_dir, suffix_te)[:-3]

        self.res_dir_tr = ppath
        self.res_dir_va = ppath
        self.res_dir_te = ppath

        # Create summary writers
        if self.config.run_mode == "train":
            self.summary_tr = tf.summary.FileWriter(
                os.path.join(self.res_dir_tr, "train", "logs"))
        if self.config.run_mode != "comp":
            self.summary_va = tf.summary.FileWriter(
                os.path.join(self.res_dir_va, "valid", "logs"))
        if self.config.run_mode == "test":
            self.summary_te = tf.summary.FileWriter(
                os.path.join(self.res_dir_te, "test", "logs"))

        # Create savers (one for current, one for best)
        self.saver_cur = tf.train.Saver()
        self.saver_best = tf.train.Saver()
        # Save file for the current model
        self.save_file_cur = os.path.join(
            self.res_dir_tr, "model")
        # Save file for the best model
        self.save_file_best = os.path.join(
            self.res_dir_tr, "models-best")
        self.save_file_best_ours_ransac = os.path.join(
            self.res_dir_tr, "models-best-ours-ransac")

        # Other savers
        self.va_res_file = os.path.join(self.res_dir_va, "valid", "va_res.txt")
        self.va_res_file_ours_ransac = os.path.join(self.res_dir_va, "valid", "va_res_ours_ransac.txt")

    def train(self, data):
        """Training function.

        Parameters
        ----------
        data_tr : tuple
            Training data.

        data_va : tuple
            Validation data.

        x_va : ndarray
            Validation data.

        y_va : ndarray
            Validation labels.

        """

        # print("Initializing...")
        self.sess.run(tf.global_variables_initializer())
        
        # ----------------------------------------
        # Resume data if it already exists
        latest_checkpoint = tf.train.latest_checkpoint(
            self.res_dir_tr)
        b_resume = latest_checkpoint is not None
        if b_resume:
            # Restore network
            # print("Restoring from {}...".format(self.res_dir_tr))
            self.saver_cur.restore(
                self.sess,
                latest_checkpoint
            )
            # restore number of steps so far
            step = self.sess.run(self.global_step)
            # restore best validation result
            if os.path.exists(self.va_res_file):
                with open(self.va_res_file, "r") as ifp:
                    dump_res = ifp.read()
                dump_res = parse(
                    "{best_va_res:e}\n", dump_res)
                best_va_res = dump_res["best_va_res"]
            if os.path.exists(self.va_res_file_ours_ransac):
                with open(self.va_res_file_ours_ransac, "r") as ifp:
                    dump_res = ifp.read()
                dump_res = parse(
                    "{best_va_res:e}\n", dump_res)
                best_va_res_ours_ransac = dump_res["best_va_res"]
        else:
            # print("Starting from scratch...")
            step = 0
            best_va_res = -1
            best_va_res_ours_ransac = -1

        # ----------------------------------------
        if self.config.data_name.startswith("oan"):
            data_loader = iter(data["train"])
        else: 
            # Unpack some data for simple coding
            xs_tr = data["train"]["xs"]
            ys_tr = data["train"]["ys"]
            Rs_tr = data["train"]["Rs"]
            ts_tr = data["train"]["ts"]
            T1s_tr = data["train"]["T1s"]
            T2s_tr = data["train"]["T2s"]
            K1s_tr = data["train"]["K1s"]
            K2s_tr = data["train"]["K2s"]

        # ----------------------------------------
        # The training loop
        batch_size = self.config.train_batch_size
        max_iter = self.config.train_iter
        
        for step in trange(step, max_iter, ncols=self.config.tqdm_width):
            # ----------------------------------------
            # Batch construction

            # Get a random training batch
            if self.config.data_name.startswith("oan"):
                try:
                    data_dict = next(data_loader)
                except StopIteration:
                    data_loader = iter(data["train"])
                    data_dict = next(data_loader)
            
                xs_b = data_dict["xs"] 
                ys_b = data_dict["ys"]
                Rs_b = data_dict["Rs"].reshape(-1, 9)
                ts_b = data_dict["ts"].reshape(-1, 3)
                T1s_b = data_dict["T1s"] 
                T2s_b = data_dict["T2s"]
                K1s_b = data_dict["K1s"]
                K2s_b = data_dict["K2s"]
            else:
                ind_cur = np.random.choice(
                    len(xs_tr), batch_size, replace=False)
                # Use minimum kp in batch to construct the batch
                numkps = np.array([xs_tr[_i].shape[1] for _i in ind_cur])
                cur_num_kp = numkps.min()
                # Actual construction of the batch
                xs_b = np.array(
                    [xs_tr[_i][:, :cur_num_kp, :] for _i in ind_cur]
                ).reshape(batch_size, 1, cur_num_kp, 4)
                ys_b = np.array(
                    [ys_tr[_i][:cur_num_kp, :] for _i in ind_cur]
                ).reshape(batch_size, cur_num_kp, 2)
                Rs_b = np.array(
                    [Rs_tr[_i] for _i in ind_cur]
                ).reshape(batch_size, 9)
                ts_b = np.array(
                    [ts_tr[_i] for _i in ind_cur]
                ).reshape(batch_size, 3)
                if self.config.use_fundamental > 0:
                    T1s_b = np.array(
                        [T1s_tr[_i] for _i in ind_cur])
                    T2s_b = np.array(
                        [T2s_tr[_i] for _i in ind_cur])
                    K1s_b = np.array(
                        [K1s_tr[_i] for _i in ind_cur])
                    K2s_b = np.array(
                        [K2s_tr[_i] for _i in ind_cur])
            # ----------------------------------------
            # Train

            # Feed Dict
            feed_dict = {
                self.x_in: xs_b,
                self.y_in: ys_b,
                self.R_in: Rs_b,
                self.t_in: ts_b,
                self.is_training: True,
            }

            # add use_fundamental
            if self.config.use_fundamental > 0:
                feed_dict[self.T1_in] = T1s_b
                feed_dict[self.T2_in] = T2s_b
                feed_dict[self.K1_in] = K1s_b
                feed_dict[self.K2_in] = K2s_b

            # Fetch
            fetch = {
                "optim": self.optim,
                "loss": self.loss,
                "precision": self.precision,
                "recall": self.recall,
            }
            # Check if we want to write summary and check validation
            b_write_summary = ((step + 1) % self.config.report_intv) == 0
            b_validate = ((step + 1) % self.config.val_intv) == 0
            if b_write_summary or b_validate:
                fetch["summary"] = self.summary_op
                fetch["global_step"] = self.global_step
            # Run optimization
            # res = self.sess.run(fetch, feed_dict=feed_dict)
            try:
                res = self.sess.run(fetch, feed_dict=feed_dict)
            except (ValueError, tf.errors.InvalidArgumentError):
                # print("Backward pass had numerical errors. "
                #      "This training batch is skipped!")
                continue
            # Write summary and save current model
            if b_write_summary:
                self.summary_tr.add_summary(
                    res["summary"], global_step=res["global_step"])
                self.saver_cur.save(
                    self.sess, self.save_file_cur,
                    global_step=self.global_step,
                    write_meta_graph=False)

            # ----------------------------------------
            # Validation
            if b_validate:
                va_res = 0
                cur_global_step = res["global_step"]
                score = self.last_logit # defaul score: local attention
                if self.config.weight_opt == "sigmoid_softmax":
                    score = [self.last_logit, self.logit_softmax, self.last_weights]

                test_process_ins = [self.x_in, self.y_in, self.R_in, self.t_in, self.is_training] 

                if self.config.use_fundamental > 0:
                    test_process_ins += [self.T1_in, self.T2_in, self.K1_in, self.K2_in]
                    
                va_res, va_res_ours_ransac = test_process(
                    "valid", self.sess, cur_global_step,
                    self.summary_op, self.summary_va,
                    test_process_ins,
                    None, None, None,
                    self.logits, self.e_hat, self.loss, self.precision, self.recall,
                    self.last_e_hat, score, self.last_x_in,
                    data["valid"],
                    self.res_dir_va, self.config, True)
                # Higher the better
                if va_res > best_va_res:
                    # print("Saving best model with va_res = {}".format(va_res))
                    best_va_res = va_res
                    # Save best validation result
                    with open(self.va_res_file, "w") as ofp:
                        ofp.write("{:e}\n".format(best_va_res))
                    # Save best model
                    self.saver_best.save(
                        self.sess, self.save_file_best,
                        write_meta_graph=False,
                    )
                if va_res_ours_ransac > best_va_res_ours_ransac:
                    # print("Saving best model with va_res_ours_ransac = {}".format(va_res_ours_ransac))
                    best_va_res_ours_ransac = va_res_ours_ransac
                    # Save best validation result
                    with open(self.va_res_file_ours_ransac, "w") as ofp:
                        ofp.write("{:e}\n".format(best_va_res_ours_ransac))
                    # Save best model
                    self.saver_best.save(
                        self.sess, self.save_file_best_ours_ransac,
                        write_meta_graph=False,
                    )

    def test(self, data):
        """Test routine"""

        # Check if model exists
        if not os.path.exists(self.save_file_best + ".index"):
            # print("Model File {} does not exist! Quiting".format(self.save_file_best))
            exit(1)

        # Restore model
        # print("Restoring from {}...".format(self.save_file_best))
        self.saver_best.restore(
            self.sess,
            self.save_file_best)
        # Run Test
        cur_global_step = 0 # dummy
        test_mode_list = ["test"] # Only evaluate on test set
        for test_mode in test_mode_list:
            score = self.last_logit
            if self.bool_use_weight_for_score:
                # print("score is from weights!")
                score = self.last_weights
            if self.config.weight_opt == "sigmoid_softmax":
                score = [self.last_logit, self.logit_softmax, self.last_weights] 

            test_process_ins = [self.x_in, self.y_in, self.R_in, self.t_in, self.is_training] 

            if self.config.use_fundamental > 0:
                test_process_ins += [self.T1_in, self.T2_in, self.K1_in, self.K2_in]
            
            test_process(
                test_mode, self.sess,
                cur_global_step,
                self.summary_op, getattr(self, "summary_" + test_mode[:2]),
                test_process_ins,
                None, None, None,
                self.logits, self.e_hat, self.loss, self.precision, self.recall,
                self.last_e_hat, score, self.last_x_in,
                data[test_mode],
                getattr(self, "res_dir_" + test_mode[:2]), self.config)

#
# network.py ends here
