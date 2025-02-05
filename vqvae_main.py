import torch
import random
import numpy as np

from configs import FLAGS
from models.vqvae import VQVAE
from datasets.dataset import TinyImageNet

from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.optim import AdamW

import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

import torch.nn as nn
from pytorch_fid import fid_score
import torchvision



class VQVAESolver:
    def __init__(self, FLAGS):
        self.FLAGS = FLAGS          # config

        self.start_epoch            = 1
        self.model                  = None                         # torch.nn.Module
        self.optimizer              = None                         # torch.optim.Optimizer
        self.scheduler              = None                         # torch.optim.lr_scheduler._LRScheduler
        self.summary_writer         = None                         # torch.utils.tensorboard.SummaryWriter
        self.train_loader           = None                         # dataloader for training dataset
        self.test_loader            = None                         # dataloader for testing dataset
        self.training_set           = self.get_dataset('train')
        self.testing_set            = self.get_dataset('test')
        self.batch_size_train       = FLAGS.data.train.batch_size
        self.batch_size_test        = FLAGS.data.test.batch_size
        self.model_save_path        = 'vae.pt'
        self.model_load_path        = 'vae222.pt'
        self.real_image_folder      = 'Real_images/'
        self.generated_image_folder = 'Generated_images/'
        self.image_tot:int          = 0
        self.image_totR:int         = 0
        self.image_totG:int         = 0
        print(FLAGS)
        self.max_epoch = FLAGS.max_epoch
        # choose device for train or test 
        if FLAGS.device == 'mps':       
           self.device = torch.device('mps')
        else:
           self.device = torch.device('cpu')
        # ......

    def config_model(self):
        self.model = VQVAE(in_channels=FLAGS.model.in_channels,embedding_dim=FLAGS.model.d_embedding,
                           num_embeddings=FLAGS.model.n_embedding,hidden_dims=FLAGS.model.channels_list,
                           beta=FLAGS.model.beta).to(self.device)
        

    def get_dataset(self, flag='train'):
        if flag=='train':
            return TinyImageNet(type='train',transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))]),
                                root_dir=FLAGS.data.train.root_dir,filelist=FLAGS.data.train.filelist)
        else:
            return TinyImageNet(type='test',transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))]),
                                root_dir=FLAGS.data.train.root_dir,filelist=FLAGS.data.train.filelist)
    
    def config_dataloader(self, disable_train=False):
        self.train_loader=DataLoader(dataset=self.training_set, batch_size=self.batch_size_train, shuffle=True)
        # self.test_loader=DataLoader(dataset=self.testing_set, batch_size=self.batch_size_test, shuffle=False)

    def config_optimizer(self):
        if FLAGS.optimizer.type=='adamw':
            self.optimizer = AdamW(self.model.parameters(),lr=FLAGS.optimizer.base_lr,betas=(FLAGS.optimizer.betas[0],FLAGS.optimizer.betas[1]),
                                   weight_decay=FLAGS.optimizer.weight_decay)

    def config_scheduler(self):
        if FLAGS.scheduler=='CosineAnnealingLR':
            self.scheduler = CosineAnnealingLR(optimizer=self.optimizer, T_max=self.max_epoch)

    def train(self):
        self.manual_seed()
        self.config_model()
        self.config_dataloader()
        self.config_optimizer()
        self.config_scheduler()
        # set model as train mode
        self.model.train()
        criterion = criterion = nn.MSELoss()
        for _ in tqdm(range(self.max_epoch)):
            # read data
            for i, (y,x) in enumerate(self.train_loader):
                # model forward process
                logits, vq_loss, min_indices = self.model(x.float().to(self.device))
                
                # compute loss
                loss_ae=0.6*criterion(logits.to(self.device), x.float().to(self.device))+0.4*vq_loss.to(self.device)
                
                # compute gradient
                loss_ae.backward()
                
                # optimize parameters
                self.optimizer.step()
                self.optimizer.zero_grad()
                
                if i%100==0:
                    self.model.eval()
                    g=self.model.generate(x.float().to(self.device)).cpu()
                    new_p=torch.concat((x[0],g[0]),dim=-1)
                    self.save_images(new_p)
                    self.model.train()
                    
            self.scheduler.step()
            torch.save(self.model.state_dict(), self.model_save_path)


    def test(self):
        self.config_model()
        self.config_dataloader(True)
        # set model as eval mode
        # read data
        # model forward process
        # compute loss
        # ......
    def eval(self):
        # self.config_model()
        # self.config_dataloader()
        # print("in1")
        # self.load()
        # print("in2")
        """for i,(y,x) in enumerate(self.train_loader):
            for j in range(x.shape[0]):
                self.save_images(x[j],'real')
            result=self.model.generate(x.float().to(self.device))
            for j in range(x.shape[0]):
                self.save_images(result[j],'gen')"""
            
        # inception_model = torchvision.models.inception_v3(pretrained=True)
        # transform_=transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])
        fid_value = fid_score.calculate_fid_given_paths([self.real_image_folder, self.generated_image_folder],batch_size=32,device='mps',dims=2048)
        print('FID value:', fid_value)
        

    def save_images(self, image_data: torch.Tensor, type='train'):
        # save reconstructed images
        # assuming that image_data: [3,H,W] with value range in [-1, 1]
        if type=='train':
            self.image_tot+=1
            img_pil=transforms.ToPILImage()((image_data + 1) / 2.0)
            img_pil.save(f"result_transformer/image_{self.image_tot}.png")
        elif type=='real':
            self.image_totR+=1
            img_pil=transforms.ToPILImage()((image_data + 1) / 2.0)
            img_pil.save(self.real_image_folder+f"image_{self.image_totR}.png")
        elif type=='gen':
            self.image_totG+=1
            img_pil=transforms.ToPILImage()((image_data + 1) / 2.0)
            img_pil.save(self.generated_image_folder+f"image_{self.image_totG}.png")
            

    def manual_seed(self):
        rand_seed = self.FLAGS.rand_seed
        if rand_seed > 0:
            random.seed(rand_seed)
            np.random.seed(rand_seed)
            torch.manual_seed(rand_seed)
            torch.cuda.manual_seed(rand_seed)
            torch.cuda.manual_seed_all(rand_seed)
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True

    def run(self):
        eval('self.%s()' % self.FLAGS.run)
    
    def save(self):
        torch.save(self.model.state_dict(), self.model_save_path)
    
    def load(self):
        # load vqvae
        checkpoint = torch.load(self.model_load_path)
        self.model = VQVAE(in_channels=FLAGS.model.in_channels,embedding_dim=FLAGS.model.d_embedding,
                           num_embeddings=FLAGS.model.n_embedding,hidden_dims=FLAGS.model.channels_list,
                           beta=FLAGS.model.beta).to(self.device)
        self.model.load_state_dict(checkpoint, strict=False)
        self.model.eval()

    @classmethod
    def main(cls):
        completion = cls(FLAGS)
        # completion.run()
        # completion.train()
        completion.eval()
        
        

if __name__ == '__main__':
    VQVAESolver.main()