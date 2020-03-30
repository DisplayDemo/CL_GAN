import os
import logging
from collections import OrderedDict

import torch
import torch.nn as nn
from torch.optim import lr_scheduler

import modules.networks as networks
from .base_model import BaseModel
from modules.Loss import GAN_Loss

logger = logging.getLogger('base')


class CL_Model(BaseModel):
    def __init__(self, opt):
        super(CL_Model, self).__init__(opt)
        train_opt = opt['train']

        # define networks and load pretrained models

        self.network_G = networks.define_G(opt).to(self.device)
        if self.is_train:
            self.network_D = networks.define_D(opt).to(self.device)
            self.network_G.train()
            self.network_D.train()

        self.load()

        if self.is_train:
            # get networks G pixel loss
            if train_opt['pixel_weight'] > 0:
                l_pix_type = train_opt['pixel_criterion']
                if l_pix_type == 'l1':
                    self.cri_pix = nn.L1Loss().to(self.device)
                elif l_pix_type == 'l2':
                    self.cri_pix = nn.MSELoss().to(self.device)
                self.l_pix_w = train_opt['pixel_weight']
            else:
                logging.info('Remove pixel loss.')
                self.cri_pix = None

            # get networks G feature loss
            if train_opt['feature_weight'] > 0:
                l_fea_type = train_opt['feature_criterion']
                if l_fea_type == 'l1':
                    self.cri_fea = nn.L1Loss().to(self.device)
                elif l_fea_type == 'l2':
                    self.cri_fea = nn.MSELoss().to(self.device)
                self.l_fea_w = train_opt['feature_weight']
            else:
                logging.info('Remove feature loss.')
                self.cri_fea = None

            if self.cri_fea:
                self.net_Feature = networks.define_F(opt, True).to(self.device)

            # GAN Loss
            self.cri_gan = GAN_Loss(train_opt['gan_type'], 1.0, 0.0).to(self.device)
            self.l_gan_W = train_opt['gan_weight']

            # D_update_ratio and D_init_iters are for WGAN
            self.D_update_ratio = train_opt['D_update_ratio'] if train_opt['D_update_ratio'] else 1
            self.D_init_iters = train_opt['D_init_iters'] if train_opt['D_init_iters'] else 0

            # D class segmentation loss
            self.cri_ce = nn.CrossEntropyLoss(ignore_index=0).to(self.device)

            # optimizers G

            wd_G = train_opt['weight_decay_G']
            optim_params_CL = []
            optim_params_other = []
            for k, v in self.network_G.named_parameters():
                if 'CL' in k or 'Cond' in k:
                    optim_params_CL.append(v)
                else:
                    optim_params_other.append(v)

            self.optimizer_G_CL = torch.optim.Adam(optim_params_CL, lr=train_opt['lr_G'] * 5,
                                                   weight_decay=wd_G, betas=(0.9, 0.999))
            self.optimizer_G_other = torch.optim.Adam(optim_params_other, lr=train_opt['lr_G'],
                                                      weight_decay=wd_G, betas=(0.9, 0.999))
            self.optimizers.append(self.optimizer_G_CL)
            self.optimizers.append(self.optimizer_G_other)

            # optimizers D
            wd_D = train_opt['weight_decay_D']
            self.optimizer_D = torch.optim.Adam(self.network_D.parameters(), lr=train_opt['lr_D'],
                                                weight_decay=wd_D, betas=(0.9, 0.999))
            self.optimizers.append(self.optimizer_D)

            # lr decay
            if train_opt['lr_scheme'] == 'MultiStepLR':
                for optimizer in self.optimizers:
                    self.schedulers.append(
                        lr_scheduler.MultiStepLR(optimizer, train_opt['lr_steps'], train_opt['lr_gamma']))

            else:
                raise NotImplementedError('MultiStepLR learning rate scheme is enough.')

            self.log_dict = OrderedDict()
        self.print_network()

    def feed_data(self, data, need_HR=True):
        self.Var_LR = data['LR'].to(self.device)
        self.Var_Seg = data['seg'].to(self.device)
        self.Var_Class = data['category'].to(self.device)

        if need_HR:
            self.Var_HR = data['HR'].long().to(self.device)

    def optimize_parameters(self, step):
        # network G
        self.optimizer_G_CL.zero_grad()
        self.optimizer_G_other.zero_grad()
        self.Fake_HR = self.network_G((self.Var_LR,self.Var_Seg))

        loss_G_Total = 0
        if step%self.D_update_ratio == 0 and step>self.D_init_iters:
            if self.cri_pix: # pixel loss
                l_g_pix = self.l_pix_w * self.cri_pix(self.Fake_HR, self.Var_HR)
                loss_G_Total+=l_g_pix
            if self.cri_fea:
                real_fea = self.net_Feature(self.Var_HR).detach()
                fake_fea = self.net_Feature(self.Fake_HR).detach()
                l_g_fea = self.l_fea_w * self.cri_fea(fake_fea, real_fea)
                loss_G_Total+=l_g_fea

            # GAN G and Class Segmentation Losses

            Pred_G_Fake, ClassS_G_Fake = self.network_D(self.Fake_HR)

            l_g_gan = self.l_gan_W * self.cri_gan(Pred_G_Fake, True)
            l_g_cls = self.l_gan_W * self.cri_ce(ClassS_G_Fake, self.Var_Seg)
            loss_G_Total+=l_g_gan
            loss_G_Total+=l_g_cls

            loss_G_Total.backward()
            self.optimizer_G_CL.step()
        if step > 20000:
            self.optimizer_G_other.step()

        # network D
        self.optimizer_D.zero_grad()
        loss_D_Total = 0

        #real data
        Pred_D_Real,ClassS_D_Real = self.network_D(self.Var_HR)
        l_d_real = self.cri_gan(Pred_D_Real,True)
        l_d_cls_real = self.cri_ce(ClassS_D_Real, self.Var_Seg)

        #Fake Data
        Pred_D_Fake, ClassS_D_Fake = self.network_D(self.Fake_HR.detach())  # detach to avoid BP to G
        l_d_fake = self.cri_gan(Pred_D_Fake, False)
        l_d_cls_fake = self.cri_ce(ClassS_D_Fake, self.Var_Seg)

        loss_D_Total = l_d_real + l_d_cls_real + l_d_fake + l_d_cls_fake

        loss_D_Total.backward()
        self.optimizer_D.step()

        #writing log
        if step % self.D_update_ratio == 0 and step > self.D_init_iters:
            # G
            if self.cri_pix:
                self.log_dict['loss_G_Pix'] = l_g_pix.item()
            if self.cri_fea:
                self.log_dict['loss_G_Fea'] = l_g_fea.item()
            self.log_dict['loss_G_GAN'] = l_g_gan.item()
            # D
        self.log_dict['loss_D_Real'] = l_d_real.item()
        self.log_dict['loss_D_Fake'] = l_d_fake.item()
        self.log_dict['loss_D_ClassS_Real'] = l_d_cls_real.item()
        self.log_dict['loss_D_ClassS_Fake'] = l_d_cls_fake.item()

        # D outputs
        self.log_dict['D_Real'] = torch.mean(Pred_D_Real.detach())
        self.log_dict['D_Fake'] = torch.mean(Pred_D_Fake.detach())


    def test(self):
        self.network_G.eval()
        with torch.no_grad():
            self.Fake_HR = self.network_G((self.Var_LR,self.Var_Seg))
        self.network_G.train()

    def get_current_log(self):
        return self.log_dict

    def get_current_visuals(self, need_HR=True):
        out_dict = OrderedDict()
        out_dict['LR'] = self.Var_LR.detach()[0].float().cpu()
        out_dict['SR'] = self.Fake_HR.detach()[0].float().cpu()
        if need_HR:
            out_dict['HR'] = self.Var_HR.detach()[0].float().cpu()
        return out_dict

    def print_network(self):
        # G
        s, n = self.get_network_description(self.network_G)
        if isinstance(self.network_G, nn.DataParallel):
            net_struc_str = '{} - {}'.format(self.network_G.__class__.__name__,
                                             self.network_G.module.__class__.__name__)
        else:
            net_struc_str = '{}'.format(self.network_G.__class__.__name__)

        logger.info('Network G structure: {}, with parameters: {:,d}'.format(net_struc_str, n))
        logger.info(s)
        if self.is_train:
            # D
            s, n = self.get_network_description(self.network_D)
            if isinstance(self.network_D, nn.DataParallel):
                net_struc_str = '{} - {}'.format(self.network_D.__class__.__name__,
                                                 self.network_D.module.__class__.__name__)
            else:
                net_struc_str = '{}'.format(self.network_D.__class__.__name__)

            logger.info('Network D structure: {}, with parameters: {:,d}'.format(net_struc_str, n))
            logger.info(s)

            if self.cri_fea:  # F, Perceptual Network
                s, n = self.get_network_description(self.net_Feature)
                if isinstance(self.net_Feature, nn.DataParallel):
                    net_struc_str = '{} - {}'.format(self.net_Feature.__class__.__name__,
                                                     self.net_Feature.module.__class__.__name__)
                else:
                    net_struc_str = '{}'.format(self.net_Feature.__class__.__name__)

                logger.info('Network F structure: {}, with parameters: {:,d}'.format(net_struc_str, n))
                logger.info(s)

    def load(self):
        load_path_G = self.opt['path']['pretrain_model_G']
        if load_path_G is not None:
            logger.info('Loading pretrained model for G [{:s}] ...'.format(load_path_G))
            self.load_network(load_path_G, self.network_G)
        load_path_D = self.opt['path']['pretrain_model_D']
        if self.opt['is_train'] and load_path_D is not None:
            logger.info('Loading pretrained model for D [{:s}] ...'.format(load_path_D))
            self.load_network(load_path_D, self.network_D)

    def save(self, iter_step):
        self.save_network(self.network_G, 'G', iter_step)
        self.save_network(self.network_D, 'D', iter_step)








