import torch
import torch.nn as nn


# Define GAN loss: [vanilla | lsgan ]
class GAN_Loss(nn.Module):
    def __init__(self, gan_type, real_label_val=1.0, fake_label_val=0.0):
        super(GAN_Loss, self).__init__()
        self.GAN_Type = gan_type
        self.Real = real_label_val
        self.Fake = fake_label_val

        if self.GAN_Type == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()
        elif self.GAN_Type == 'lsgan':
            self.loss = nn.MSELoss()

        else:
            raise NotImplementedError('GAN type [{:s}] is not found'.format(self.GAN_Type))

    def get_target_label(self, i_n, target_is_real):
        if target_is_real:
            return torch.empty_like(i_n).fill_(self.Real)
        else:
            return torch.empty_like(i_n).fill_(self.Fake)

    def forward(self, input, target_is_real):
        target_label = self.get_target_label(input,target_is_real)
        loss = self.loss(input,target_label)
        return loss

