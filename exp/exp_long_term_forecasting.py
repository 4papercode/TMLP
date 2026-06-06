from data_provider.data_factory import data_provider
from data_provider.data_loader import Dataset_WECC
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, visual
from utils.metrics import metric
import torch
import torch.nn as nn
from torch import optim
import os
import time
import warnings
import numpy as np
import json
from utils.dtw_metric import dtw,accelerated_dtw
from utils.augmentation import run_augmentation,run_augmentation_single

warnings.filterwarnings('ignore')

class Exp_Long_Term_Forecast(Exp_Basic):
    def __init__(self, args):
        super(Exp_Long_Term_Forecast, self).__init__(args)

    def _build_model(self):
        model = self.model_dict[self.args.model].Model(self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        # WECC dataset requires special construction (neighbor data, sim weights)
        if getattr(self.args, 'data', None) == 'WECC':
            return self._get_wecc_data(flag)
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _get_wecc_data(self, flag):
        """Build Dataset_WECC and its DataLoader for the target BA."""
        import pandas as pd
        ba_names       = json.load(open(self.args.ba_names_path))
        neighbors_dict = json.load(open(self.args.neighbors_path))

        # Load full adjacency matrix (needed for subgraph_adj in GNN)
        adj_df = pd.read_csv(self.args.adj_path, index_col=0)
        adj_df.index   = [n.replace("LADWP", "LDWP") for n in adj_df.index]
        adj_df.columns = [n.replace("LADWP", "LDWP") for n in adj_df.columns]
        full_adj_np = adj_df.values.astype(np.float32)

        ba_idx           = ba_names.index(self.args.ba_name)
        neighbor_indices = neighbors_dict[self.args.ba_name]

        timeenc = 0 if self.args.embed != 'timeF' else 1
        data_set = Dataset_WECC(
            args             = self.args,
            ba_name          = self.args.ba_name,
            neighbor_indices = neighbor_indices,
            ba_names         = ba_names,
            data_root        = self.args.wecc_data_root,
            full_adj_np      = full_adj_np,
            ba_idx           = ba_idx,
            flag             = flag,
            size             = [self.args.seq_len, self.args.label_len, self.args.pred_len],
            features         = self.args.features,
            target           = self.args.target,
            timeenc          = timeenc,
            freq             = self.args.freq,
            year             = getattr(self.args, 'year', None),
            pi_root_1        = getattr(self.args, 'pi_root_1', None),
            pi_root_2        = getattr(self.args, 'pi_root_2', None),
        )
        shuffle = flag == 'train'
        data_loader = torch.utils.data.DataLoader(
            data_set,
            batch_size  = self.args.batch_size,
            shuffle     = shuffle,
            num_workers = self.args.num_workers,
            drop_last   = False,
        )
        print(f"[WECC] {flag}: {self.args.ba_name}, "
              f"{len(neighbor_indices)} neighbors, {len(data_set)} samples")
        return data_set, data_loader

    @staticmethod
    def _unpack_batch(batch):
        """Return (batch_x, batch_y, batch_x_mark, batch_y_mark, neighbor_list, subgraph_adj, pi_images).
        For non-WECC loaders the last three are None."""
        if len(batch) == 8:
            bx, by, bxm, bym, nb_xs, subgraph_adj, pi_imgs_1, pi_imgs_2 = batch
            neighbor_list = [nb_xs[:, i] for i in range(nb_xs.shape[1])]
            return bx, by, bxm, bym, neighbor_list, subgraph_adj[0], pi_imgs_1, pi_imgs_2
        elif len(batch) == 7:
            bx, by, bxm, bym, nb_xs, subgraph_adj, pi_imgs_1 = batch
            neighbor_list = [nb_xs[:, i] for i in range(nb_xs.shape[1])]
            return bx, by, bxm, bym, neighbor_list, subgraph_adj[0], pi_imgs_1, None
        elif len(batch) == 6:
            bx, by, bxm, bym, nb_xs, subgraph_adj = batch
            neighbor_list = [nb_xs[:, i] for i in range(nb_xs.shape[1])]
            return bx, by, bxm, bym, neighbor_list, subgraph_adj[0], None, None
        else:
            bx, by, bxm, bym = batch
            return bx, by, bxm, bym, None, None, None, None

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion

    def _call_model(self, batch_x, batch_x_mark, dec_inp, batch_y_mark,
                    neighbor_xs, sim_w, pi_imgs_1, pi_imgs_2):
        """Call model with extra WECC args for TimeVLM, plain call for others.
        Returns (outputs, align_gt, align_mm) for TimeVLM, (outputs, 0, 0) for others."""
        if self.args.model == 'TimeVLM':
            outputs, align_gt, align_mm = self.model(
                batch_x, batch_x_mark, dec_inp, batch_y_mark,
                neighbor_x_encs=neighbor_xs, subgraph_adj=sim_w,
                pi_images_1=pi_imgs_1, pi_images_2=pi_imgs_2)
            return outputs, align_gt, align_mm
        else:
            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
            zero = torch.tensor(0.0, device=batch_x.device)
            return outputs, zero, zero

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for i, batch in enumerate(vali_loader):
                batch_x, batch_y, batch_x_mark, batch_y_mark, neighbor_xs, sim_w, pi_imgs_1, pi_imgs_2 = \
                    self._unpack_batch(batch)

                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)
                if pi_imgs_1 is not None:
                    pi_imgs_1 = pi_imgs_1.float().to(self.device)
                if pi_imgs_2 is not None:
                    pi_imgs_2 = pi_imgs_2.float().to(self.device)
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs, _, _ = self._call_model(batch_x, batch_x_mark, dec_inp, batch_y_mark,
                                                         neighbor_xs, sim_w, pi_imgs_1, pi_imgs_2)
                else:
                    outputs, _, _ = self._call_model(batch_x, batch_x_mark, dec_inp, batch_y_mark,
                                                     neighbor_xs, sim_w, pi_imgs_1, pi_imgs_2)

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                pred = outputs.detach().cpu()
                true = batch_y.detach().cpu()
                loss = criterion(pred, true)
                total_loss.append(loss)
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            for i, batch in enumerate(train_loader):
                batch_x, batch_y, batch_x_mark, batch_y_mark, neighbor_xs, sim_w, pi_imgs_1, pi_imgs_2 = \
                    self._unpack_batch(batch)

                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)
                if pi_imgs_1 is not None:
                    pi_imgs_1 = pi_imgs_1.float().to(self.device)
                if pi_imgs_2 is not None:
                    pi_imgs_2 = pi_imgs_2.float().to(self.device)

                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                lambda1 = getattr(self.args, 'align_lambda1', 0.0)
                lambda2 = getattr(self.args, 'align_lambda2', 0.0)

                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs, align_gt, align_mm = self._call_model(
                            batch_x, batch_x_mark, dec_inp, batch_y_mark,
                            neighbor_xs, sim_w, pi_imgs_1, pi_imgs_2)
                        f_dim = -1 if self.args.features == 'MS' else 0
                        outputs = outputs[:, -self.args.pred_len:, f_dim:]
                        batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                        loss = criterion(outputs, batch_y) + lambda1 * align_gt + lambda2 * align_mm
                        train_loss.append(loss.item())
                else:
                    outputs, align_gt, align_mm = self._call_model(
                        batch_x, batch_x_mark, dec_inp, batch_y_mark,
                        neighbor_xs, sim_w, pi_imgs_1, pi_imgs_2)
                    f_dim = -1 if self.args.features == 'MS' else 0
                    outputs = outputs[:, -self.args.pred_len:, f_dim:]
                    batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                    loss = criterion(outputs, batch_y) + lambda1 * align_gt + lambda2 * align_mm
                    train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            adjust_learning_rate(model_optim, epoch + 1, self.args)

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')
        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth'), map_location=self.device))

        preds = []
        trues = []
        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        with torch.no_grad():
            for i, batch in enumerate(test_loader):
                batch_x, batch_y, batch_x_mark, batch_y_mark, neighbor_xs, sim_w, pi_imgs_1, pi_imgs_2 = \
                    self._unpack_batch(batch)

                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)
                if pi_imgs_1 is not None:
                    pi_imgs_1 = pi_imgs_1.float().to(self.device)
                if pi_imgs_2 is not None:
                    pi_imgs_2 = pi_imgs_2.float().to(self.device)

                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs, _, _ = self._call_model(batch_x, batch_x_mark, dec_inp, batch_y_mark,
                                                         neighbor_xs, sim_w, pi_imgs_1, pi_imgs_2)
                else:
                    outputs, _, _ = self._call_model(batch_x, batch_x_mark, dec_inp, batch_y_mark,
                                                     neighbor_xs, sim_w, pi_imgs_1, pi_imgs_2)

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, :]
                batch_y = batch_y[:, -self.args.pred_len:, :].to(self.device)
                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()
                if test_data.scale and self.args.inverse:
                    shape = outputs.shape
                    outputs = test_data.inverse_transform(outputs.reshape(shape[0] * shape[1], -1)).reshape(shape)
                    batch_y = test_data.inverse_transform(batch_y.reshape(shape[0] * shape[1], -1)).reshape(shape)
        
                outputs = outputs[:, :, f_dim:]
                batch_y = batch_y[:, :, f_dim:]

                pred = outputs
                true = batch_y

                preds.append(pred)
                trues.append(true)
                if i % 20 == 0:
                    input = batch_x.detach().cpu().numpy()
                    if test_data.scale and self.args.inverse:
                        shape = input.shape
                        input = test_data.inverse_transform(input.reshape(shape[0] * shape[1], -1)).reshape(shape)
                    gt = np.concatenate((input[0, :, -1], true[0, :, -1]), axis=0)
                    pd = np.concatenate((input[0, :, -1], pred[0, :, -1]), axis=0)
                    visual(gt, pd, os.path.join(folder_path, str(i) + '.pdf'))

        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)
        print('test shape:', preds.shape, trues.shape)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        print('test shape:', preds.shape, trues.shape)

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        
        # dtw calculation
        if self.args.use_dtw:
            dtw_list = []
            manhattan_distance = lambda x, y: np.abs(x - y)
            for i in range(preds.shape[0]):
                x = preds[i].reshape(-1,1)
                y = trues[i].reshape(-1,1)
                if i % 100 == 0:
                    print("calculating dtw iter:", i)
                d, _, _, _ = accelerated_dtw(x, y, dist=manhattan_distance)
                dtw_list.append(d)
            dtw = np.array(dtw_list).mean()
        else:
            dtw = 'not calculated'
            

        mae, mse, rmse, mape, mspe = metric(preds, trues)
        print('mse: {}, mae: {}, dtw: {}'.format(mse, mae, dtw))
        f = open("result_long_term_forecast.txt", 'a')
        f.write(setting + "  \n")
        f.write('mse: {}, mae: {}, dtw: {}'.format(mse, mae, dtw))
        f.write('\n')
        f.write('\n')
        f.close()

        np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe]))
        np.save(folder_path + 'pred.npy', preds)
        np.save(folder_path + 'true.npy', trues)

        return
