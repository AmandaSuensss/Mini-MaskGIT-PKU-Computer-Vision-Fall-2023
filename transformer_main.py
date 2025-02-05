import torch
import random
import numpy as np
import math
from configs import FLAGS
from datasets.dataset import TinyImageNet
from models.transformer import BidirectionalTransformer
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.optim import AdamW
from pytorch_fid import fid_score
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from collections import deque
import torch.nn as nn
from models.vqvae import VQVAE
from torch.optim.lr_scheduler import CosineAnnealingLR
import matplotlib.pyplot as plt
import torchvision.utils as vutils
from torchvision.utils import save_image

class TransformerSolver:
    def __init__(self, FLAGS):
        self.FLAGS = FLAGS

        self.start_epoch = 1
        self.model = None           # torch.nn.Module
        self.optimizer = None       # torch.optim.Optimizer  
        self.scheduler = None       # torch.optim.lr_scheduler._LRScheduler
        self.train_loader = None
        self.test_loader = None
        self.patch_size = FLAGS.data.train.patch_size
        self.training_set     = self.get_dataset('train')
        self.testing_set      = self.get_dataset('test')
        self.batch_size_train = FLAGS.data.train.batch_size
        self.batch_size_test  = FLAGS.data.test.batch_size
        self.image_tot:int          = 0
        self.image_totR:int         = 0
        self.image_totG:int         = 0
        self.grad_cum = FLAGS.grad_cum
        self.mask_value = FLAGS.mask_value
        self.codebook_size = FLAGS.codebook_size
        self.drop_label = FLAGS.drop_label
        self.n = len(self.training_set)
        self.embedding_dim = FLAGS.embbeding_dim_

        # customize the model path
        self.model_save_path  = '/Users/amandala-mando/Desktop/maskgit/transformer.pt'
        self.model_load_path  = '/Users/amandala-mando/Desktop/maskgit/transformer.pt'
        self.model_vae = '/Users/amandala-mando/Desktop/maskgit/vae222.pt'
        self.real_image_folder      = '/Users/amandala-mando/Desktop/maskgit/Real_images/'
        self.generated_image_folder = '/Users/amandala-mando/Desktop/maskgit/Generated_images/'
        
        self.vae=None
        
        #print(FLAGS)
        self.max_epoch = FLAGS.max_epoch
        # choose device for train or test 
        if FLAGS.device == 'mps':       
           self.device = torch.device('mps')
        else:
           self.device = torch.device('cpu')

    def config_model(self):
        self.model = BidirectionalTransformer(img_size=FLAGS.model.transformer.img_size, hidden_dim=FLAGS.model.transformer.hidden_dim, 
                                              codebook_size=FLAGS.model.transformer.codebook_size, depth=FLAGS.model.transformer.depth, 
                                              heads=FLAGS.model.transformer.heads, mlp_dim=FLAGS.model.transformer.mlp_dim, 
                                              dropout=FLAGS.model.transformer.dropout, nclass=FLAGS.model.transformer.nclass).to(self.device)

    def load_vqvae(self):
        # Instantiate the VQ-VAE model
        self.vae = VQVAE(in_channels=FLAGS.model.vae.in_channels,embedding_dim=FLAGS.model.vae.d_embedding,
                            num_embeddings=FLAGS.model.vae.n_embedding,hidden_dims=FLAGS.model.vae.channels_list,
                            beta=FLAGS.model.vae.beta).to(self.device)
        state_dict = torch.load(self.model_vae)
        self.vae.load_state_dict(state_dict, strict=False)
        self.vae.eval()
        

    def get_dataset(self, flag):
        if flag=='train':
            return TinyImageNet(type='train',transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))]),
                                root_dir=FLAGS.data.train.root_dir,filelist=FLAGS.data.train.filelist)
        else:
            return TinyImageNet(type='test',transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))]),
                                root_dir=FLAGS.data.train.root_dir,filelist=FLAGS.data.train.filelist)
    
    def config_dataloader(self, disable_train=False):
        self.train_loader=DataLoader(dataset=self.training_set, batch_size=self.batch_size_train, shuffle=True)

    def config_optimizer(self):
        if FLAGS.optimizer.type=='adamw':
            self.optimizer = AdamW(self.model.parameters(),lr=FLAGS.optimizer.base_lr,betas=(FLAGS.optimizer.betas[0],FLAGS.optimizer.betas[1]),
                                   weight_decay=FLAGS.optimizer.weight_decay)

    def config_scheduler(self):
        if FLAGS.scheduler=='CosineAnnealingLR':
            self.scheduler = CosineAnnealingLR(optimizer=self.optimizer, T_max=self.max_epoch)

    # mask the tokens
    def get_mask_code(self, code, value,mode='cosine'):
        r = torch.rand(code.size(0))
        val_to_mask = torch.arccos(r) / (math.pi * 0.5)
        # arc cosine scheduler
        if mode == "linear":                # linear scheduler
            val_to_mask = r
        elif mode == "square":              # square scheduler
            val_to_mask = (r ** 2)
        elif mode == "cosine":              # cosine scheduler
            val_to_mask = torch.cos(r * math.pi * 0.5)
        elif mode == "arccos":              # arc cosine scheduler
            val_to_mask = torch.arccos(r) / (math.pi * 0.5)
        else:
            val_to_mask = None
        
        mask_code = code.detach().clone()
        
        # Sample the amount of tokens + localization to mask
        mask = torch.rand(size=code.size()) < val_to_mask.view(code.size(0), 1, 1)

        # Mask the selected token by the value
        mask_code[mask] = torch.full_like(mask_code[mask], value)
        #print(mask_code.size(), mask.size())
        
        return mask_code, mask

    # schedule the masking process
    def adap_sche(self, step, leave=False,mode='arccos'):
        r = torch.linspace(1, 0, step)
        # cosine scheduler
        val_to_mask = torch.cos(r * math.pi * 0.5)
        if mode == "root":              # root scheduler
            val_to_mask = 1 - (r ** .5)
        elif mode == "linear":          # linear scheduler
            val_to_mask = 1 - r
        elif mode == "square":          # square scheduler
            val_to_mask = 1 - (r ** 2)
        elif mode == "cosine":          # cosine scheduler
            val_to_mask = torch.cos(r * math.pi * 0.5)
        elif mode == "arccos":          # arc cosine scheduler
            val_to_mask = torch.arccos(r) / (math.pi * 0.5)
        else:
            return
        # fill the scheduler by the ratio of tokens to predict at each step
        sche = (val_to_mask / val_to_mask.sum()) * (self.patch_size * self.patch_size)
        sche = sche.round()
        sche[sche == 0] = 1                                                  # add 1 to predict a least 1 token / step
        sche[-1] += (self.patch_size * self.patch_size) - sche.sum()         # need to sum up nb of code
        return tqdm(sche.int(), leave=leave)
    
    # train seed
    def train(self):
        self.manual_seed()
        self.config_model()
        self.config_dataloader()
        self.config_optimizer()
        self.config_scheduler()
        self.load_vqvae()

        # set model as train mode
        self.model.train()
        criterion = nn.CrossEntropyLoss()
        window_loss = deque(maxlen=self.grad_cum)
        
        # self.load_vqvae()

        for epoch in tqdm(range(self.max_epoch)):
            cum_loss = 0.0
            # progress_bar = tqdm(self.train_loader, leave=False)
            # scheduler = iter(self.adap_sche(self.n, leave=False))  # Create the mask scheduler and wrap it with iter()
            

            for i,(xx, yy) in enumerate(self.train_loader):
                x = xx.clone()
                y = yy.clone()
                x = x.float().to(self.device)
                y = y.to(self.device)
                # if y.shape[0]!=self.batch_size_train: break
                # print(i)
                with torch.no_grad():
                    y = y.float()
                    encoding = self.vae.encode(y)
                    # print("sdsa",encoding.size())
                    _, _, code = self.vae.vq_layer(encoding) # min_indices

                code = code.reshape(y.size(0), self.patch_size, self.patch_size)
                masked_code, mask = self.get_mask_code(code, value=self.mask_value)

                # Perform a forward pass with the masked code
                drop_label = torch.empty(x.size()).uniform_(0, 1) < self.drop_label
                
                pred = self.model(masked_code, x, drop_label=drop_label)
                mask = mask.to(code.device)

                # Calculate the loss
                loss = criterion(pred.reshape(-1, self.codebook_size + 1), code.view(-1))

                # Backward pass and optimization
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
                
                
                cum_loss += loss.item()
                window_loss.append(loss.item())
                window_loss_total = sum(window_loss)
                # progress_bar.set_description(f"Epoch {epoch+1}/{self.max_epoch} | Loss: {cum_loss/self.n:.4f} | Window Loss: {window_loss_total:.4f}")
                
            #print(1)
            self.scheduler.step()
            self.model.eval()
            #print(2)
            new_p2,___,____ = self.sample(nb_sample=10)
            #print("1")
            for j in range(new_p2.shape[0]):
                self.save_images(new_p2[j])
            self.model.train()
            
            # Update the global epoch count
            # global_epoch += 1

            # Save the modepython transformer_main.py --config configs/transformer.yamll
            torch.save(self.model.state_dict(), self.model_save_path)

    def sample( self, init_code=None, nb_sample=50, labels=None, sm_temp=1, w=3,
               randomize="linear", r_temp=4.5, step=12,mode='arccos'):#step=12
        # vqvae=self.load_vqvae()
        # self.eval()
        self.model.eval()
        l_codes = []  # Save the intermediate codes predicted
        l_mask = []   # Save the intermediate masks
        with torch.no_grad():
            # 关于label的初始化
            if labels is None:  # Default classes generated
                # random
                labels = [0,1,2,3,4,5,6,7,8, random.randint(0, 10)] * (nb_sample // 10) # 后面这个操作是什么意思？
                labels = torch.LongTensor(labels)

            #关于code（以及mask）的初始化
            drop = torch.ones(nb_sample, dtype=torch.bool).to(self.device)
            if init_code is not None:  # Start with a pre-define code
                code = init_code
                mask = (init_code == self.codebook_size).float().view(nb_sample, self.patch_size*self.patch_size) #基本上会产生一个全白的画布
            else:  # Initialize a code (直接看它如何应对没有inicode的)
                if self.mask_value < 0:  # Code initialize with random tokens
                    code = torch.randint(0, FLAGS.codebook_size-1, (nb_sample, self.patch_size, self.patch_size)).to(self.device)
                else:  # Code initialize with masked tokens （我没有很搞明白这个mask_value是用来做什么的）
                    code = torch.full((nb_sample, self.patch_size, self.patch_size), self.mask_value).to(self.device)
                mask = torch.ones(nb_sample, self.patch_size*self.patch_size).to(self.device) # 一开始全是1

            # Instantiate scheduler（每一步揭开多少面纱）
            scheduler = self.adap_sche(step,mode=mode)

            # Beginning of sampling, t = number of token to predict a step "indice"
            for indice, t in enumerate(scheduler):
                if mask.sum() < t:  # Cannot predict more token than 16*16 or 32*32
                    t = int(mask.sum().item())
                if mask.sum() == 0:  # Break if code is fully predicted
                    break
                
                # begin for computing conf
                if w != 0:
                    # Model Prediction 首先这个cat不是左右接在一起，而是上下接在一起，增加的是batchsize)
                    logit = self.model(torch.cat([code.clone(), code.clone()], dim=0).to(self.device),
                                        torch.cat([labels, labels], dim=0).to(self.device), # 这时候labels似乎就起到作用了
                                        torch.cat([~drop, drop], dim=0).to(self.device)) # 唯一的差一点在于~drop与drop，其实可以相当于什么也没有扔掉
                    logit_c, logit_u = torch.chunk(logit, 2, dim=0) # 把它们拆开
                    _w = w * (indice / (len(scheduler)-1)) # 相当于t/T
                    # Classifier Free Guidance (这个logit是用来把(1+_w)*(~drop)-(_w)(drop))（我没有太看懂）
                    logit = (1 + _w) * logit_c - _w * logit_u
                else:
                    logit = self.model(code.clone().to(self.device), labels.to(self.device), drop_label=~drop).to(self.device) # 哦，看来这个drop主要在这里起作用
                # logit 长 (b, p*p, codebbok+1)
                prob = torch.softmax(logit * sm_temp, -1)
                # prob 长 (b, p*p, codebbok+1)
                # Sample the code from the softmax prediction
                distri = torch.distributions.Categorical(probs=prob)
                pred_code = distri.sample()
                # pred_code 长 (b, p*p) from a range(int) of [0, codebook_size+1)

                
                conf = torch.gather(prob, 2, pred_code.view(nb_sample, self.patch_size*self.patch_size, 1))

                if randomize == "linear":  # add gumbel noise decreasing over the sampling process
                    ratio = (indice / (len(scheduler)-1))
                    rand = r_temp * np.random.gumbel(size=(nb_sample, self.patch_size*self.patch_size)) * (1 - ratio)
                    conf = torch.log(conf.squeeze()) + torch.from_numpy(rand).to(self.device, dtype=torch.float32)
                elif randomize == "warm_up":  # chose random sample for the 2 first steps
                    conf = torch.rand_like(conf) if indice < 2 else conf
                elif randomize == "random":   # chose random prediction at each step
                    conf = torch.rand_like(conf)
                
                # logit 长 (b, p*p, codebbok+1)
                logit  = self.model(code.clone().to(self.device), labels.to(self.device), drop_label=~drop).to(self.device)
                conf_  = torch.softmax(logit,-1)
                # [b,p*p,codebook+1]
                pred_code_=torch.argmax(conf_,dim=2).long()
                # [b, p*p]
                #print(pred_code_.size())
                conf_ = torch.max(conf_, dim=2)[0]
                # [b,p*p]
                # print(conf_.size())
                # print(conf_)
                #conf=conf_
                #pred_code=pred_code_
                
                # do not predict on already predicted tokens
                conf[~mask.bool()] = -math.inf
                # [b,p*p]
                # print(conf.size())
                # end for computing conf
                
                # chose the predicted token with the highest confidence
                tresh_conf, indice_mask = torch.topk(conf.view(nb_sample, -1), k=t, dim=-1)
                tresh_conf = tresh_conf[:, -1]
                # tresh_conf：形状为 (nb_sample, t)，包含每个样本中前 t 个最大的概率值。
                # indice_mask：形状为 (nb_sample, t)，包含每个样本中前 t 个最大概率值对应的索引。
                # 然后只保留第t大的，形状变为(batchsize, 1)

                # replace the chosen tokens
                conf = (conf >= tresh_conf.unsqueeze(-1)).view(nb_sample, self.patch_size, self.patch_size)
                f_mask = (mask.view(nb_sample, self.patch_size, self.patch_size).float() * conf.view(nb_sample, self.patch_size, self.patch_size).float()).bool()
                # 已经被遮掩的会是1，没有被遮掩的会是0，1*0=0，这个可以用来揭开面纱
                code[f_mask] = pred_code.view(nb_sample, self.patch_size, self.patch_size)[f_mask]
                # 这里牵扯到一个code和precode之间的联系，我对着整个code都感到有点乱，我感觉它是针对某种随机生成的。

                # update the mask
                for i_mask, ind_mask in enumerate(indice_mask):
                    mask[i_mask, ind_mask] = 0
                l_codes.append(pred_code.view(nb_sample, self.patch_size, self.patch_size).clone())
                l_mask.append(mask.view(nb_sample, self.patch_size, self.patch_size).clone())
                
            # End the for loop
            
            # decode the final prediction
            _code = torch.clamp(code, 0, FLAGS.codebook_size-1)
            # print(_code.size())
            #mixed_zq = model.vq_layer.get_codebook_entry(masked_code.view(-1).unsqueeze(1),(batch_size, patch_size, patch_size, encoding.shape[1]))
            #g = model.decode(mixed_zq.float().to(device)).cpu()
            print(_code.size())
            code_=self.vae.vq_layer.get_codebook_entry(_code.view(-1).unsqueeze(1),(nb_sample,self.patch_size,self.patch_size, FLAGS.model.vae.d_embedding))
            x = self.vae.decode(code_.float())
        #print(x.size())
        self.model.train()
        return x, l_codes, l_mask
    
    def viz(self,x, nrow=10, pad=2, size=(18, 18)):
        """
        Visualize a grid of images.
        Args:
            x (torch.Tensor): Input images to visualize.
            nrow (int): Number of images in each row of the grid.
            pad (int): Padding between the images in the grid.
            size (tuple): Size of the visualization figure.
        """
        nb_img = len(x)
        min_norm = x.min(-1)[0].min(-1)[0].min(-1)[0].view(-1, 1, 1, 1)
        max_norm = x.max(-1)[0].max(-1)[0].max(-1)[0].view(-1, 1, 1, 1)
        x = (x - min_norm) / (max_norm - min_norm)
        x = vutils.make_grid(x.float().cpu(), nrow=nrow, padding=pad, normalize=False)
        plt.figure(figsize = size)
        plt.axis('off')
        plt.imshow(x.permute(1, 2, 0))
        plt.show()
    
    def decoding_viz(self, gen_code, mask):
        """
        Visualize the decoding process of generated images with associated masks.

        Args:
            gen_code (torch.Tensor): Generated code for decoding.
            mask (torch.Tensor): Mask used for decoding.
            maskgit (MaskGIT): MaskGIT instance.
        """
        start = torch.FloatTensor([1, 1, 1]).view(1, 3, 1, 1).expand(1, 3, self.patch_size, self.patch_size) * 0.8
        end = torch.FloatTensor([0.01953125, 0.30078125, 0.08203125]).view(1, 3, 1, 1).expand(1, 3, self.patch_size, self.patch_size) * 1.4
        code = torch.stack((gen_code), dim=0).squeeze()
        print(code.size())
        mask = torch.stack((mask), dim=0).view(-1, 1, self.patch_size, self.patch_size).cpu()
        

        with torch.no_grad():
            # x = maskgit.ae.decode_code(torch.clamp(code, 0, 1023))
            print(torch.clamp(code, 0, 1023).view(-1,code.size(2),code.size(3)).size())
            code_=self.vae.vq_layer.get_codebook_entry(torch.clamp(code, 0, 1023).view(-1,code.size(2),code.size(3)).view(-1).unsqueeze(1),(code.shape[0]*code.shape[1],self.patch_size,self.patch_size, FLAGS.model.vae.d_embedding))
            x = self.vae.decode(code_.float())

        binary_mask = mask * start + (1 - mask) * end
        binary_mask = vutils.make_grid(binary_mask, nrow=len(gen_code), padding=1, pad_value=0.4, normalize=False)
        binary_mask = binary_mask.permute(1, 2, 0)
        print(binary_mask.size())

        plt.figure(figsize = (18, 2))
        plt.gca().invert_yaxis()
        plt.pcolormesh(binary_mask, edgecolors='w', linewidth=.5)
        plt.axis('off')
        plt.show()
        self.viz(x, nrow=len(gen_code))

    def test(self):
        self.manual_seed()
        self.config_model()
        self.config_dataloader(True)
        self.config_optimizer()
        self.config_scheduler()
        self.load_vqvae()
        # set model as eval mode
        # read data
        # model forward process
        # compute loss
        # ......

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
    
    def save(self):
        torch.save(self.model.state_dict(), self.model_save_path)
    
    def load(self):
        # load vqvae
        checkpoint = torch.load(self.model_load_path)
        self.model.load_state_dict(checkpoint, strict=False)
        # for name, param in self.model.named_parameters():
        #     print(f"Parameter name: {name}, Size: {param.size()}")
        self.model.eval()
    
    def research(self):
        self.load_vqvae()
        self.config_model()
        self.config_dataloader()
        self.load()
        new_p2,code_,mask_ = self.sample(nb_sample=10,step=12,sm_temp=1.3,w=9,r_temp=4.5,mode='cosine')
        #print("1")
        for j in range(new_p2.shape[0]):
            self.save_images(new_p2[j])
        # self.viz(new_p2[0])
        self.decoding_viz(code_, mask_)
        
        

    @classmethod
    def main(cls):
        completion = cls(FLAGS)
        completion.run()
        completion.train()
        # completion.eval()
        # completion.research()

if __name__ == '__main__':
    TransformerSolver.main()
