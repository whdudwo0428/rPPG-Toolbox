"""Trainer for MTTS-CAN (BVP + Respiration)."""

import logging
import os
from collections import OrderedDict

import numpy as np
import torch
import torch.optim as optim
from evaluation.metrics import calculate_metrics
from neural_methods.model.MTTS_CAN import MTTS_CAN
from neural_methods.trainer.BaseTrainer import BaseTrainer
from tqdm import tqdm


def _split_labels_for_multitask(labels):
    """
    Returns (label_bvp, label_resp) shaped as [N*D, 1].
    Accepts:
      - dict with keys like 'T1','T2' (or 'BVP','RESP')
      - tensor [..., 1 or 2]
    """
    if isinstance(labels, dict):
        # try common keys
        bvp = labels.get('T1', labels.get('BVP', None))
        resp = labels.get('T2', labels.get('RESP', None))
        if bvp is None:
            raise ValueError("Label dict must contain 'T1'/'BVP' for BVP.")
        # reshape
        bvp = bvp.view(-1, 1)
        resp = resp.view(-1, 1) if resp is not None else None
        return bvp, resp
    else:
        # tensor: [N, D, C] or [N*D, C]
        if labels.dim() == 3:
            N, D, C = labels.shape
            labels = labels.view(N * D, C)
        elif labels.dim() == 2:
            pass
        elif labels.dim() == 1:
            labels = labels.unsqueeze(-1)
        else:
            raise ValueError(f"Unsupported label shape: {labels.shape}")
        if labels.size(-1) == 1:
            return labels, None
        elif labels.size(-1) >= 2:
            return labels[:, [0]], labels[:, [1]]
        else:
            raise ValueError("Label tensor has invalid last dim.")

def _reshape_labels_like_data(labels, N, D):
    """
    labels: torch.Tensor of shape
      - [N, D] or [N, D, 1] -> flatten to [N*D, 1]
      - [N, 1] or [N]       -> repeat each item D times -> [N*D, 1]
      - [N*D, 1]            -> as-is
    returns: [N*D, 1]
    """
    if labels.dim() == 3:                         # [N, D, 1]
        assert labels.shape[0] == N and labels.shape[1] == D
        labels = labels.reshape(N * D, labels.size(-1))
    elif labels.dim() == 2:
        if labels.shape[0] == N and labels.shape[1] == D:   # [N, D]
            labels = labels.reshape(N * D, 1)
        elif labels.shape[0] == N and labels.shape[1] == 1: # [N, 1]
            labels = labels.repeat_interleave(D, dim=0)     # -> [N*D, 1]
        elif labels.shape[0] == N * D and labels.shape[1] == 1:
            pass  # already good
        else:
            raise ValueError(f"Unexpected label shape {labels.shape} for N={N}, D={D}")
    elif labels.dim() == 1:
        if labels.shape[0] == N:
            labels = labels.repeat_interleave(D, dim=0).unsqueeze(-1)
        elif labels.shape[0] == N * D:
            labels = labels.unsqueeze(-1)
        else:
            raise ValueError(f"Unexpected label shape {labels.shape} for N={N}, D={D}")
    else:
        raise ValueError(f"Unsupported label dim: {labels.dim()}")
    return labels


class MTTscanTrainer(BaseTrainer):

    def __init__(self, config, data_loader):
        super().__init__()  # BaseTrainer는 인자 없는 __init__을 가짐
        self.config = config
        self.device = torch.device(config.DEVICE)
        self.frame_depth     = config.MODEL.TSCAN.FRAME_DEPTH
        self.max_epoch_num   = config.TRAIN.EPOCHS
        self.model_dir       = config.MODEL.MODEL_DIR
        self.model_file_name = config.TRAIN.MODEL_FILE_NAME
        self.batch_size      = config.TRAIN.BATCH_SIZE
        self.num_of_gpu      = config.NUM_OF_GPU_TRAIN
        self.base_len        = self.num_of_gpu * self.frame_depth
        self.chunk_len       = config.TRAIN.DATA.PREPROCESS.CHUNK_LENGTH

        # 손실 가중치 (기본값: 논문 1.0, 0.5)
        if hasattr(config, "LOSS") and hasattr(config.LOSS, "WEIGHTS"):
            self.loss_w_bvp, self.loss_w_resp = config.LOSS.WEIGHTS
        else:
            self.loss_w_bvp, self.loss_w_resp = 1.0, 0.5

        # 모델
        self.model = MTTS_CAN(
            frame_depth=self.frame_depth,
            img_size=config.TRAIN.DATA.PREPROCESS.RESIZE.H
        ).to(self.device)
        self.model = torch.nn.DataParallel(self.model, device_ids=list(range(self.num_of_gpu)))

        # 데이터 로더
        self.data_loader = data_loader
        self.num_train_batches = len(data_loader["train"]) if "train" in data_loader else 0

        # 손실 함수 (논문: L1)
        self.crit_bvp  = torch.nn.L1Loss()
        self.crit_resp = torch.nn.L1Loss()

        # 옵티마이저 / 스케줄러 (TSCAN 방식)
        self.optimizer = optim.AdamW(self.model.parameters(), lr=config.TRAIN.LR, weight_decay=0.0)
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=config.TRAIN.LR,
            epochs=config.TRAIN.EPOCHS,
            steps_per_epoch=self.num_train_batches if self.num_train_batches > 0 else 1
        )

        # 모드 체크
        if config.TOOLBOX_MODE not in ("train_and_test", "only_test"):
            raise ValueError("MTTS-CAN trainer initialized in incorrect toolbox mode!")

    def train(self, data_loader):
        if data_loader["train"] is None:
            raise ValueError("No data for train")
        mean_training_losses, mean_valid_losses, lrs = [], [], []
        for epoch in range(self.max_epoch_num):
            print(f"\n====Training Epoch: {epoch}====")
            running_loss = 0.0
            train_loss = []
            self.model.train()
            tbar = tqdm(data_loader["train"], ncols=80)
            for idx, batch in enumerate(tbar):
                tbar.set_description("Train epoch %s" % epoch)
                data, labels = batch[0].to(self.device), batch[1]

                # 1) N, D 먼저 확보
                N, D, C, H, W = data.shape

                # 2) 레이블을 디바이스로 옮기고, 형태 맞추기
                if isinstance(labels, dict):
                    labels = {k: v.to(self.device) for k, v in labels.items()}
                    # bvp/resp 분리
                    bvp = labels.get('T1', labels.get('BVP', None))
                    resp = labels.get('T2', labels.get('RESP', None))
                    if bvp is None:
                        raise ValueError("Label dict must contain 'T1'/'BVP' for BVP.")
                    label_bvp = _reshape_labels_like_data(bvp, N, D)  # -> [N*D,1]
                    label_resp = _reshape_labels_like_data(resp, N, D) if resp is not None else None
                else:
                    labels = labels.to(self.device)
                    # 단일태스크 텐서도 동일 규칙으로 전개
                    label_bvp = _reshape_labels_like_data(labels, N, D)
                    label_resp = None

                # 3) 이제 데이터 전개
                data = data.view(N * D, C, H, W)

                # 4) base_len(=num_gpu * frame_depth)로 잘라내기 (둘 다 같은 길이로!)
                limit = (N * D) // self.base_len * self.base_len
                data = data[:limit]
                label_bvp = label_bvp[:limit]
                if label_resp is not None:
                    label_resp = label_resp[:limit]

                # 5) forward & loss
                self.optimizer.zero_grad()

                pred_bvp, pred_resp = self.model(data)  # model output: [N*T,1] each

                loss_bvp = self.crit_bvp(pred_bvp, label_bvp)  # label_bvp shape: [N*T,1]
                if (label_resp is not None) and (pred_resp is not None):
                    loss_resp = self.crit_resp(pred_resp, label_resp)
                    loss = self.loss_w_bvp * loss_bvp + self.loss_w_resp * loss_resp
                else:
                    loss = loss_bvp

                loss.backward()

                # OneCycle LR logging(옵션)
                if hasattr(self, "scheduler") and self.scheduler is not None:
                    lrs.append(self.scheduler.get_last_lr()[0])
                self.optimizer.step()
                if hasattr(self, "scheduler") and self.scheduler is not None:
                    self.scheduler.step()

                running_loss += loss.item()
                train_loss.append(loss.item())
                if idx % 100 == 99:
                    print(f'[{epoch}, {idx + 1:5d}] loss: {running_loss / 100:.3f}')
                    running_loss = 0.0
                tbar.set_postfix(loss=loss.item())

            mean_training_losses.append(np.mean(train_loss))
            self.save_model(epoch)

            if not self.config.TEST.USE_LAST_EPOCH:
                valid_loss = self.valid(data_loader)
                mean_valid_losses.append(valid_loss)
                print('validation loss: ', valid_loss)
                if self.min_valid_loss is None or valid_loss < self.min_valid_loss:
                    self.min_valid_loss = valid_loss
                    self.best_epoch = epoch
                    print("Update best model! Best epoch: {}".format(self.best_epoch))

        if not self.config.TEST.USE_LAST_EPOCH:
            print("best trained epoch: {}, min_val_loss: {}".format(self.best_epoch, self.min_valid_loss))
        if self.config.TRAIN.PLOT_LOSSES_AND_LR:
            self.plot_losses_and_lrs(mean_training_losses, mean_valid_losses, lrs, self.config)

    def valid(self, data_loader):
        if data_loader["valid"] is None:
            raise ValueError("No data for valid")

        print("\n===Validating===")
        valid_loss = []
        self.model.eval()
        with torch.no_grad():
            vbar = tqdm(data_loader["valid"], ncols=80)
            for _, batch in enumerate(vbar):
                data_valid, labels_valid = batch[0].to(self.device), batch[1]

                # [N, D, C, H, W] -> [N*D, C, H, W]
                N, D, C, H, W = data_valid.shape
                data_valid = data_valid.view(N * D, C, H, W)
                # base_len 단위로 잘라서 TSM 윈도와 DP에 맞춤
                data_valid = data_valid[: (N * D) // self.base_len * self.base_len]

                # 라벨: dict 또는 tensor 모두 지원 → (bvp, resp) 각 [N*D,1]
                if isinstance(labels_valid, dict):
                    labels_valid = {k: v.to(self.device) for k, v in labels_valid.items()}
                else:
                    labels_valid = labels_valid.to(self.device)
                label_bvp, label_resp = _split_labels_for_multitask(labels_valid)

                # data_valid 길이에 정확히 맞춤(브로드캐스팅 경고/에러 방지)
                label_bvp = label_bvp[: data_valid.size(0)]
                if label_resp is not None:
                    label_resp = label_resp[: data_valid.size(0)]

                # 추론 및 손실
                pred_bvp, pred_resp = self.model(data_valid)

                loss_bvp = torch.nn.functional.l1_loss(pred_bvp, label_bvp)
                if (label_resp is not None) and (pred_resp is not None):
                    loss_resp = torch.nn.functional.l1_loss(pred_resp, label_resp)
                    loss = self.loss_w_bvp * loss_bvp + self.loss_w_resp * loss_resp
                else:
                    loss = loss_bvp

                valid_loss.append(loss.item())
                vbar.set_postfix(loss=loss.item())
        return float(np.mean(valid_loss))

    def test(self, data_loader):
        if data_loader["test"] is None:
            raise ValueError("No data for test")

        print("\n===Testing===")
        self.chunk_len = self.config.TEST.DATA.PREPROCESS.CHUNK_LENGTH

        predictions_bvp, labels_bvp = dict(), dict()
        predictions_resp, labels_resp = dict(), dict()

        if self.config.TOOLBOX_MODE == "only_test":
            if not os.path.exists(self.config.INFERENCE.MODEL_PATH):
                raise ValueError("Inference model path error! Please check INFERENCE.MODEL_PATH in yaml.")
            self.model.load_state_dict(torch.load(self.config.INFERENCE.MODEL_PATH))
            print("Testing uses pretrained model!")
        else:
            if self.config.TEST.USE_LAST_EPOCH:
                last_epoch_model_path = os.path.join(self.model_dir, self.model_file_name + '_Epoch' + str(self.max_epoch_num - 1) + '.pth')
                print("Testing uses last epoch as non-pretrained model!")
                print(last_epoch_model_path)
                self.model.load_state_dict(torch.load(last_epoch_model_path))
            else:
                best_model_path = os.path.join(self.model_dir, self.model_file_name + '_Epoch' + str(self.best_epoch) + '.pth')
                print("Testing uses best epoch selected using model selection as non-pretrained model!")
                print(best_model_path)
                self.model.load_state_dict(torch.load(best_model_path))

        self.model = self.model.to(self.config.DEVICE)
        self.model.eval()
        print("Running model evaluation on the testing dataset!")

        with torch.no_grad():
            for _, test_batch in enumerate(tqdm(data_loader["test"], ncols=80)):
                batch_size = test_batch[0].shape[0]
                data_test, labels_test = test_batch[0].to(self.config.DEVICE), test_batch[1]

                # [N, D, C, H, W] -> [N*D, C, H, W]
                N, D, C, H, W = data_test.shape
                data_test = data_test.view(N * D, C, H, W)
                data_test = data_test[: (N * D) // self.base_len * self.base_len]

                # 라벨 쪼개고 길이 맞추기
                if isinstance(labels_test, dict):
                    labels_test = {k: v.to(self.config.DEVICE) for k, v in labels_test.items()}
                else:
                    labels_test = labels_test.to(self.config.DEVICE)
                label_bvp, label_resp = _split_labels_for_multitask(labels_test)
                label_bvp = label_bvp[: data_test.size(0)]
                if label_resp is not None:
                    label_resp = label_resp[: data_test.size(0)]

                # 추론
                pred_bvp, pred_resp = self.model(data_test)

                # 필요 시 CPU 이동
                if self.config.TEST.OUTPUT_SAVE_DIR:
                    pred_bvp = pred_bvp.cpu()
                    label_bvp = label_bvp.cpu()
                    if (pred_resp is not None) and (label_resp is not None):
                        pred_resp = pred_resp.cpu()
                        label_resp = label_resp.cpu()

                # 클립 단위로 저장 (각 클립 길이는 chunk_len)
                for idx in range(batch_size):
                    subj_index = test_batch[2][idx]
                    sort_index = int(test_batch[3][idx])
                    if subj_index not in predictions_bvp:
                        predictions_bvp[subj_index] = dict()
                        labels_bvp[subj_index] = dict()
                        predictions_resp[subj_index] = dict()
                        labels_resp[subj_index] = dict()

                    start = idx * self.chunk_len
                    end = (idx + 1) * self.chunk_len

                    predictions_bvp[subj_index][sort_index] = pred_bvp[start:end]
                    labels_bvp[subj_index][sort_index] = label_bvp[start:end]

                    if (pred_resp is not None) and (label_resp is not None):
                        predictions_resp[subj_index][sort_index] = pred_resp[start:end]
                        labels_resp[subj_index][sort_index] = label_resp[start:end]

        print("\nCalculating metrics (BVP only by default)!")
        calculate_metrics(predictions_bvp, labels_bvp, self.config)

        if self.config.TEST.OUTPUT_SAVE_DIR:
            self.save_test_outputs(predictions_bvp, labels_bvp, self.config)
            self.save_test_outputs(predictions_resp, labels_resp, self.config, tag='_resp')

    def save_model(self, index):
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        model_path = os.path.join(self.model_dir, self.model_file_name + '_Epoch' + str(index) + '.pth')
        torch.save(self.model.state_dict(), model_path)
        print('Saved Model Path: ', model_path)