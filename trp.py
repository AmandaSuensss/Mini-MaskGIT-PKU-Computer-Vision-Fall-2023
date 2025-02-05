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

        for epoch in range(self.max_epoch):
            cum_loss = 0.0
            progress_bar = tqdm(self.train_loader, leave=False)
            scheduler = iter(self.adap_sche(self.n, leave=False))  # Create the mask scheduler and wrap it with iter()
            vqvae=self.load_vqvae()

            for x, y in progress_bar:
                x = x.float().to(self.device)
                y = y.to(self.device)
                if y.shape[0]!=self.batch_size_train: continue

                with torch.no_grad():
                    y = y.float()
                    encoding = vqvae.encode(y)
                    # print("sdsa",encoding.size())
                    _, _, code = vqvae.vq_layer(encoding) # min_indices

                code = code.reshape(y.size(0), self.patch_size, self.patch_size)
                masked_code, mask = self.get_mask_code(code, value=self.mask_value)

                # Perform a forward pass with the masked code
                pred = self.model(masked_code, x, drop_label=self.drop_label)
                mask = mask.to(code.device)

                # Calculate the loss
                loss = criterion(pred.reshape(-1, self.codebook_size + 1), code.view(-1))

                # Backward pass and optimization
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                cum_loss += loss.item()
                window_loss.append(loss.item())
                window_loss_total = sum(window_loss)
                progress_bar.set_description(f"Epoch {epoch+1}/{self.max_epoch} | Loss: {cum_loss/self.n:.4f} | Window Loss: {window_loss_total:.4f}")

                new_p2,___,____ = self.sample(nb_sample=10)
                self.save_images(new_p2)
                self.model.train()
        torch.save(self.model.state_dict(), self.model_save_path)

def sample( self, init_code=None, nb_sample=50, labels=None, sm_temp=1, w=3,
            randomize="linear", r_temp=4.5, step=12):
    vqvae=self.load_vqvae()
    # self.eval()
    l_codes = [] 
    l_mask = []  
    with torch.no_grad():
        # 关于label的初始化
        if labels is None:  # Default classes generated
            # random
            labels = [0,1,2,3,4,5,6,7,8, random.randint(0, 10)] * (nb_sample // 10) # 后面这个操作是什么意思？
            labels = torch.LongTensor(labels)

        #关于code（以及mask）的初始化
        drop = torch.ones(nb_sample, dtype=torch.bool).to(self.device)
        if init_code is not None:  
            code = init_code
            mask = (init_code == self.codebook_size).float().view(nb_sample, self.patch_size*self.patch_size) #基本上会产生一个全白的画布
        else:  
            if self.mask_value < 0: 
                code = torch.randint(0, 1024, (nb_sample, self.patch_size, self.patch_size)).to(self.device)
            else: 
                code = torch.full((nb_sample, self.patch_size, self.patch_size), self.mask_value).to(self.device)
            mask = torch.ones(nb_sample, self.patch_size*self.patch_size).to(self.device) # 一开始全是1

        scheduler = self.adap_sche(step)
        for indice, t in enumerate(scheduler):
            if mask.sum() < t:  # Cannot predict more token than 16*16 or 32*32
                t = int(mask.sum().item())
            if mask.sum() == 0:  # Break if code is fully predicted
                break
            if w != 0:
                logit = self.model(torch.cat([code.clone(), code.clone()], dim=0).to(self.device),
                                    torch.cat([labels, labels], dim=0).to(self.device), # 这时候labels似乎就起到作用了
                                    torch.cat([~drop, drop], dim=0).to(self.device)) # 唯一的差一点在于~drop与drop，其实可以相当于什么也没有扔掉
                logit_c, logit_u = torch.chunk(logit, 2, dim=0) # 把它们拆开
                _w = w * (indice / (len(scheduler)-1)) # 相当于t/T
                logit = (1 + _w) * logit_c - _w * logit_u
            else:
                logit = self.model(code.clone(), labels, drop_label=~drop) # 哦，看来这个drop主要在这里起作用
            prob = torch.softmax(logit * sm_temp, -1)
            distri = torch.distributions.Categorical(probs=prob)
            pred_code = distri.sample()
            conf = torch.gather(prob, 2, pred_code.view(nb_sample, self.patch_size*self.patch_size, 1))

            if randomize == "linear":  # add gumbel noise decreasing over the sampling process
                ratio = (indice / (len(scheduler)-1))
                rand = r_temp * np.random.gumbel(size=(nb_sample, self.patch_size*self.patch_size)) * (1 - ratio)
                conf = torch.log(conf.squeeze()) + torch.from_numpy(rand).to(self.device, dtype=torch.float32)
            elif randomize == "warm_up":  # chose random sample for the 2 first steps
                conf = torch.rand_like(conf) if indice < 2 else conf
            elif randomize == "random":   # chose random prediction at each step
                conf = torch.rand_like(conf)

            # do not predict on already predicted tokens
            conf[~mask.bool()] = -math.inf
            # end for computing conf
            
            # chose the predicted token with the highest confidence
            tresh_conf, indice_mask = torch.topk(conf.view(nb_sample, -1), k=t, dim=-1)
            tresh_conf = tresh_conf[:, -1]
            conf = (conf >= tresh_conf.unsqueeze(-1)).view(nb_sample, self.patch_size, self.patch_size)
            f_mask = (mask.view(nb_sample, self.patch_size, self.patch_size).float() * conf.view(nb_sample, self.patch_size, self.patch_size).float()).bool()
            code[f_mask] = pred_code.view(nb_sample, self.patch_size, self.patch_size)[f_mask]

            # update the mask
            for i_mask, ind_mask in enumerate(indice_mask):
                mask[i_mask, ind_mask] = 0
            l_codes.append(pred_code.view(nb_sample, self.patch_size, self.patch_size).clone())
            l_mask.append(mask.view(nb_sample, self.patch_size, self.patch_size).clone())
            
        _code = torch.clamp(code, 0, 1023)
        code_=vqvae.vq_layer.get_codebook_entry(_code.view(-1).unsqueeze(1),(nb_sample,self.patch_size,self.patch_size, 192))
        x = vqvae.decode(code_.float())
    self.train()
    return x, l_codes, l_mask