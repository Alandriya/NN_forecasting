import numpy as np
from torch import nn
import torch
from config import cfg


class Loss(nn.Module):
    def __init__(self):
        super().__init__()
        pass

    def forward(self, truth, pred, decouple_loss):
        # print(truth.shape)
        # print(pred.shape)
        differ = truth - pred  # b s c h w
        mse = torch.sum(differ ** 2, (2, 3, 4))  # b s
        mae = torch.sum(torch.abs(differ), (2, 3, 4))  # b s
        mse = torch.mean(mse)  # 1
        mae = torch.mean(mae)  # 1
        loss = mse + mae
        if 'PredRNN-V2' in cfg.model_name:
            decouple_loss = torch.sum(decouple_loss, (1, 3))  # s l b c -> s b
            decouple_loss = torch.mean(decouple_loss)  # 1
            loss = loss + cfg.decouple_loss_weight * decouple_loss
        return loss


class Loss2(nn.Module):
    def __init__(self):
        super().__init__()
        pass

    def forward(self, truth, pred):
        # print(truth.shape)
        # print(pred.shape)
        differ = truth - pred  # b s c h w
        mse = torch.sum(differ ** 2, (2, 3, 4))  # b s
        mae = torch.sum(torch.abs(differ), (2, 3, 4))  # b s
        mse = torch.mean(mse)  # 1
        mae = torch.mean(mae)  # 1
        loss = mse + mae
        return loss


class Loss_MAE(nn.Module):
    def __init__(self):
        super().__init__()
        pass

    def forward(self, truth, pred):
        # print(truth.shape)
        # print(pred.shape)
        differ = truth - pred  # b s c h w
        # print(differ)
        mae = torch.sum(torch.abs(differ), (2, 3, 4))  # b s
        mae = torch.mean(mae)  # 1
        return mae


class Loss_MSE(nn.Module):
    def __init__(self):
        super().__init__()
        pass

    def forward(self, truth, pred, mask):
        # print(truth.shape)
        # print(pred.shape)
        differ = truth - pred  # b s c h w
        differ[:, :, :, np.where(mask == 0)[0], np.where(mask == 0)[1]] = 0
        mse = torch.sum(differ ** 2, (2, 3, 4))  # b s
        mse = torch.mean(mse)  # 1
        return mse


class Loss_MSE_eigenvalues(nn.Module):
    def __init__(self):
        super().__init__()
        pass

    def forward(self, truth, pred, mask, eigens, alpha = 0.0):
        # print(truth.shape)
        # print(pred.shape)
        # print(eigens.shape)

        differ = truth - pred  # b s c h w
        differ[:, :, :, np.where(mask == 0)[0], np.where(mask == 0)[1]] = 0
        # mse = torch.sum(differ ** 2, (2, 3, 4))
        tmp = (differ ** 2) * (1-alpha) + eigens * alpha
        # print(torch.sum(tmp-differ**2))
        mse = torch.sum(tmp, (2, 3, 4))  # b s
        mse = torch.mean(mse)  # 1
        return mse


class Loss_MSE_informed(nn.Module):
    def __init__(self):
        super().__init__()
        pass

    def forward(self, truth, pred, features, alpha = 0.0, beta=0.0):
        # print(truth.shape)
        # print(pred.shape)
        # print(features.shape)

        A_model = features[:, :, :3]
        B_model = torch.zeros_like(A_model)
        for i in range(3):
            B_model[:, :, i] = features[:, :, 3 + i] + features[:, :, 6 + i]
        # print(A_model.shape)
        # print(B_model.shape)


        differ = truth - pred  # b s c h w
        loss_drift = pred - A_model

        loss_variance = pred - A_model - B_model
        # loss_variance = 0
        # raise ValueError
        tmp = differ ** 2 + (loss_drift ** 2) * alpha + (loss_variance ** 2) * beta

        # print(torch.sum(tmp-differ**2))
        mse = torch.sum(tmp, (2, 3, 4))  # b s
        mse = torch.mean(mse)  # 1
        return mse