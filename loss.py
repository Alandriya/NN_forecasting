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

    def forward(self, truth, pred):
        # print(truth.shape)
        # print(pred.shape)
        differ = truth - pred  # b s c h w
        # print(differ)
        mse = torch.sum(differ ** 2, (2, 3, 4))  # b s
        mse = torch.mean(mse)  # 1
        return mse


class Loss_MSE_eigenvalues(nn.Module):
    def __init__(self):
        super().__init__()
        pass

    def forward(self, truth, pred, eigens, alpha = 0.0):
        # print(truth.shape)
        # print(pred.shape)
        # print(eigens.shape)

        differ = truth - pred  # b s c h w
        mse = torch.sum(differ ** 2 * (1-alpha) + eigens * alpha, (2, 3, 4))  # b s
        mse = torch.mean(mse)  # 1
        return mse