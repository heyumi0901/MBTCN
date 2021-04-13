import torch.nn.functional as F
from torch import nn
from tcn import TemporalConvNet
import torch

class TCN(nn.Module):
    def __init__(self, input_size1,input_size2,input_size3,input_size4
                 , output_size, num_channels, kernel_size, dropout):
        super(TCN, self).__init__()
        self.tcn1 = TemporalConvNet(input_size1, num_channels, kernel_size=kernel_size, dropout=dropout)
        self.tcn2 = TemporalConvNet(input_size2, num_channels, kernel_size=kernel_size, dropout=dropout)
        self.tcn3 = TemporalConvNet(input_size3, num_channels, kernel_size=kernel_size, dropout=dropout)
        self.tcn4 = TemporalConvNet(input_size4, num_channels, kernel_size=kernel_size, dropout=dropout)
        # self.tcn5 = TemporalConvNet(input_size5, num_channels, kernel_size=kernel_size, dropout=dropout)
        # self. = nn.Linear(25,output_size)
        self.linear1 = nn.Sequential(nn.Linear(num_channels[-1] * 4,18), nn.SELU())

        self.dimZ = 100


    def encoder_result(self, inputs1,inputs2,inputs3,inputs4):
        #这里的batch是指 x_batch
        output1 = self.tcn1(inputs1)
        output2 = self.tcn2(inputs2)
        output3 = self.tcn3(inputs3)
        output4 = self.tcn4(inputs4)
        # output5 = self.tcn5(inputs5)
        output = torch.cat(
            [output1[:, :, -1], output2[:, :, -1], output3[:, :, -1], output4[:, :, -1], ], 1).reshape(
            output1.shape[0], -1) # input should have dimension (N, C, L)
        # o = y1[:, :, -1]
        mu =self.linear1(output)

        return output, mu
    def forward(self, inputs1,inputs2,inputs3,inputs4):
        """Inputs have to have dimension (N, C_in, L_in)"""
        # to_decoder = self.sample_encoder_Z(num_samples=5, batch=inputs)
        # 采样了num_samples 次，然后求了均值
        # decoder_logits_mean: [1000,10]
        # [batch_size, y_dim]
        # decoder_logits_mean = self.decoder_logits(to_decoder)

        inner_o, output = self.encoder_result(inputs1,inputs2,inputs3,inputs4)
        # o =self.linear3(output)
        # d = F.softmax(o,dim=1)
        return  F.log_softmax(output, dim=1)
    def testforward(self, inputs1,inputs2,inputs3,inputs4):
        """Inputs have to have dimension (N, C_in, L_in)"""
        # y1 = self.tcn(inputs).reshape(20,-1)
        # y2 = self.linear1(y1)
        inner_o ,to_decoder = self.encoder_result(inputs1,inputs2,inputs3,inputs4)
        # o =self.linear3(to_decoder)
        # 采样了num_samples 次，然后求了均值
        # decoder_logits_mean: [1000,10]
        # [batch_size, y_dim]
        # decoder_logits_mean = self.decoder_logits(y2)
        return inner_o, to_decoder