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

warnings.filterwarnings('ignore')

class Exp_Few_Shot_Forecast(Exp_Basic):
    def __init__(self, args):
        super(Exp_Few_Shot_Forecast, self).__init__(args)

    def _build_model(self):
        model = self.model_dict[self.args.model].Model(self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        if getattr(self.args, 'data', None) == 'WECC':
            return self._get_wecc_data(flag)
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _get_wecc_data(self, flag):
        """Build Dataset_WECC with optional few-shot subsampling on train split."""
        import pandas as pd
        ba_names       = json.load(open(self.args.ba_names_path))
        neighbors_dict = json.load(open(self.args.neighbors_path))

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
        )

        # Few-shot subsampling: only subsample train split
        percent = getattr(self.args, 'percent', 1.0)
        if flag == 'train' and percent < 1.0:
            num_samples = int(len(data_set) * percent)
            indices = np.random.choice(len(data_set), num_samples, replace=False)
            data_set = torch.utils.data.Subset(data_set, indices)
            print(f"[WECC few-shot] {flag}: {self.args.ba_name}, "
                  f"{percent*100:.0f}% → {len(data_set)} samples")
        else:
            print(f"[WECC] {flag}: {self.args.ba_name}, {len(data_set)} samples")

        shuffle = flag == 'train'
        data_loader = torch.utils.data.DataLoader(
            data_set,
            batch_size  = self.args.batch_size,
            shuffle     = shuffle,
            num_workers = self.args.num_workers,
            drop_last   = False,
        )
        return data_set, data_loader

    @staticmethod
    def _unpack_batch(batch):
        """Return (batch_x, batch_y, batch_x_mark, batch_y_mark, neighbor_list, subgraph_adj).
        For non-WECC loaders the last two are None."""
        if len(batch) == 6:
            bx, by, bxm, bym, nb_xs, subgraph_adj = batch
            neighbor_list = [nb_xs[:, i] for i in range(nb_xs.shape[1])]
            return bx, by, bxm, bym, neighbor_list, subgraph_adj[0]
        else:
            bx, by, bxm, bym = batch
            return bx, by, bxm, bym, None, None

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for i, batch in enumerate(vali_loader):
                batch_x, batch_y, batch_x_mark, batch_y_mark, neighbor_xs, sim_w = \
                    self._unpack_batch(batch)

                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark,
                                             neighbor_x_encs=neighbor_xs, subgraph_adj=sim_w)
                else:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark,
                                         neighbor_x_encs=neighbor_xs, subgraph_adj=sim_w)

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
                batch_x, batch_y, batch_x_mark, batch_y_mark, neighbor_xs, sim_w = \
                    self._unpack_batch(batch)

                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark,
                                             neighbor_x_encs=neighbor_xs, subgraph_adj=sim_w)
                        f_dim = -1 if self.args.features == 'MS' else 0
                        outputs = outputs[:, -self.args.pred_len:, f_dim:]
                        batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                        loss = criterion(outputs, batch_y)
                        train_loss.append(loss.item())
                else:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark,
                                         neighbor_x_encs=neighbor_xs, subgraph_adj=sim_w)
                    f_dim = -1 if self.args.features == 'MS' else 0
                    outputs = outputs[:, -self.args.pred_len:, f_dim:]
                    batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                    loss = criterion(outputs, batch_y)
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
        
        if isinstance(test_data, torch.utils.data.Subset):
            data_scaling = test_data.dataset.scale
        else:
            data_scaling = test_data.scale
        
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
                batch_x, batch_y, batch_x_mark, batch_y_mark, neighbor_xs, sim_w = \
                    self._unpack_batch(batch)

                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark,
                                             neighbor_x_encs=neighbor_xs, subgraph_adj=sim_w)
                else:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark,
                                         neighbor_x_encs=neighbor_xs, subgraph_adj=sim_w)

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, :]
                batch_y = batch_y[:, -self.args.pred_len:, :].to(self.device)
                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()
                if data_scaling and self.args.inverse:
                    shape = outputs.shape
                    if isinstance(test_data, torch.utils.data.Subset):
                        outputs = test_data.dataset.inverse_transform(outputs.reshape(shape[0] * shape[1], -1)).reshape(shape)
                        batch_y = test_data.dataset.inverse_transform(batch_y.reshape(shape[0] * shape[1], -1)).reshape(shape)
                    else:
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
                    if data_scaling and self.args.inverse:
                        shape = input.shape
                        if isinstance(test_data, torch.utils.data.Subset):
                            input = test_data.dataset.inverse_transform(input.reshape(shape[0] * shape[1], -1)).reshape(shape)
                        else:
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
