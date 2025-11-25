def forward(self, x, y,
                candidate_x, candidate_y, is_train,
                ):
        if is_train:
            data_size=candidate_x.shape[0]
            retrival_size=int(data_size*self.sample_rate)
            sample_idx=torch.randperm(data_size)[:retrival_size]
            candidate_x=candidate_x[sample_idx]
            candidate_y =candidate_y[sample_idx]
        if self.num_embeddings is not None and self.d_num >0:
            x_num,x_cat=x[:,:self.d_num],x[:,self.d_num:]
            candidate_x_num,candidate_x_cat=candidate_x[:,:self.d_num],candidate_x[:,self.d_num:]
            x_num=self.num_embeddings(x_num).flatten(1)
            candidate_x_num=self.num_embeddings(candidate_x_num).flatten(1)
            x=torch.cat([x_num,x_cat],dim=-1)
            candidate_x=torch.cat([candidate_x_num,candidate_x_cat],dim=-1)
        # x=x.double()
        # candidate_x=candidate_x.double()
        x = self.encoder(x)
        candidate_x = self.encoder(candidate_x)
        if self.n_blocks > 0:
            x = self.post_encoder(x)
            candidate_x = self.post_encoder(candidate_x)
        if is_train: 
            assert y is not None
            candidate_x = torch.cat([x, candidate_x])
            candidate_y = torch.cat([y, candidate_y])
        
        if self.d_out > 1:
            candidate_y = F.one_hot(candidate_y, self.d_out).to(x.dtype)
        elif len(candidate_y.shape) == 1:
            candidate_y=candidate_y.unsqueeze(-1)

        # calculate distance
        # default we use euclidean distance, however, cosine distance is also a good choice for classification.
        # Using cosine distance, you need to tune the temperature. You can add "temperature":["loguniform",1e-5,1] in the configs/opt_space/modernNCA.json file.
        distances = torch.cdist(x, candidate_x, p=2)
        # following is the code for cosine distance
        # x=F.normalize(x,p=2,dim=-1)   
        # candidate_x=F.normalize(candidate_x,p=2,dim=-1)
        # distances=torch.mm(x,candidate_x.T)
        # distances=-distances
        distances=distances/self.T
        # remove the label of training index
        if is_train:
            distances = distances.fill_diagonal_(torch.inf)     
        distances = F.softmax(-distances, dim=-1)
        logits = torch.mm(distances, candidate_y)
        eps=1e-7
        if self.d_out>1:
            # if task type is classification, since the logit is already normalized, we calculate the log of the logit 
            # and use nll_loss to calculate the loss
            logits=torch.log(logits+eps)
        return logits.squeeze(-1)