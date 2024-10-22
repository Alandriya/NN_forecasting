from models.convlstm import *
from models.ms_lstm import *
from models.predrnn_v2 import *
from models.ms_predrnn import *
from models.attetion_unet import *
from models import *
# from models.preciplstm import *
from config import cfg
from collections import OrderedDict

b = cfg.batch
h = cfg.height
w = cfg.width

hs = cfg.lstm_hidden_state
if cfg.kernel_size == 5:
    k, s, p = 5, 1, 2
elif cfg.kernel_size == 3:
    k, s, p = 3, 1, 1
else:
    k, s, p = 3, 1, 'same'

if cfg.model_name == 'ConvLSTM':
    rnn = ConvLSTM
elif cfg.model_name == 'MS-LSTM':
    rnn = MS_LSTM
elif cfg.model_name == 'MS-PredRNN':
    rnn = MS_PredRNN
elif cfg.model_name == 'PredRNN-V2':
    rnn = PredRNN_V2
elif cfg.model_name == 'Att-Unet':
    rnn = AttU_Net
# elif cfg.model_name == 'PrecipLSTM':
#     rnn = PrecipLSTM

# elif cfg.model_name == 'TrajGRU':
#     rnn = TrajGRU
# elif cfg.model_name == 'PredRNN':
#     rnn = PredRNN
# elif cfg.model_name == 'PredRNN++':
#     rnn = PredRNN_Plus2
# elif cfg.model_name == 'MIM':
#     rnn = MIM
# elif cfg.model_name == 'MotionRNN':
#     rnn = MotionRNN
# elif cfg.model_name == 'CMS-LSTM':
#     rnn = CMS_LSTM
# elif cfg.model_name == 'MoDeRNN':
#     rnn = MoDeRNN
# elif cfg.model_name == 'PredRNN-V2':
#     rnn = PredRNN_V2

#
# elif cfg.model_name == 'MS-ConvLSTM-WO-Skip':
#     rnn = MS_ConvLSTM_WO_Skip
# elif cfg.model_name == 'MS-ConvLSTM':
#     rnn = MS_ConvLSTM
# elif cfg.model_name == 'MS-ConvLSTM-UNet3+':
#     rnn = MS_ConvLSTM_UNet_Plus3
# elif cfg.model_name == 'MS-ConvLSTM-FC':
#     rnn = MS_ConvLSTM_FC
#
# elif cfg.model_name == 'MS-TrajGRU':
#     rnn = MS_TrajGRU

# elif cfg.model_name == 'MS-PredRNN++':
#     rnn = MS_PredRNN_Plus2
# elif cfg.model_name == 'MS-MIM':
#     rnn = MS_MIM
# elif cfg.model_name == 'MS-MotionRNN':
#     rnn = MS_MotionRNN
# elif cfg.model_name == 'MS-CMS-LSTM':
#     rnn = MS_CMS_LSTM
# elif cfg.model_name == 'MS-MoDeRNN':
#     rnn = MS_MoDeRNN
# elif cfg.model_name == 'MS-PredRNN-V2':
#     rnn = MS_PredRNN_V2
# elif cfg.model_name == 'MS-PrecipLSTM':
#     rnn = MS_PrecipLSTM
# elif cfg.model_name == 'MS-LSTM':
#     rnn = MS_LSTM
# elif cfg.model_name == 'MK-LSTM':
#     rnn = MK_LSTM
# else:
#     rnn = None

rnn = rnn(input_channel=hs, output_channel=hs, b_h_w=(b, 96, 96), kernel_size=k, stride=s, padding=p)
# nets = [OrderedDict({'conv_embed': [cfg.features_amount*cfg.in_len, hs, 1, 1, 0, 1]}),
#         rnn,
#         OrderedDict({'conv_fc': [hs, 3*cfg.out_len, 1, 1, 0, 1]})]
nets = [OrderedDict({'conv_embed': [cfg.features_amount*cfg.in_len, hs, 1, 1, 0, 1]}),
        rnn,
        OrderedDict({'conv_fc': [hs, 3*cfg.out_len, 1, 1, 0, 1]})]
# in_channels=v[0], out_channels=v[1],kernel_size=v[2], stride=v[3],padding=v[4], dilation=v[5]
