# a variant of GRU4REC: change from session-based next-item recommendation to sequential recommendation
import math
import torch
from data_loader import Dataset
from step_sample import create_named_schedule_sampler


class PositionalEncoding(torch.nn.Module):
    def __init__(self, d_model, max_len=100):
        super(PositionalEncoding, self).__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(-math.log(10000.0) * torch.arange(0, d_model, 2).float() / d_model)
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        pe = pe.unsqueeze(0)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        # x: (batch, seq_len, d_model)
        x = x + self.pe[:, :x.size(1)]
        return x

class TransformerEncoder(torch.nn.Module):
    def __init__(self, n_hidden, device, n_head=4, n_layers=4):
        super(TransformerEncoder, self).__init__()
        self.pos_encoder = PositionalEncoding(n_hidden)
        encoder_layer = torch.nn.TransformerEncoderLayer(d_model=n_hidden, nhead=n_head, batch_first=True)
        self.transformer = torch.nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.mlp_output = torch.nn.Sequential(torch.nn.Linear(n_hidden, n_hidden))
        self.to(device)
        self.device = device
    
    def forward(self, xs, length):
        shape = xs.shape
        xs = xs.view(-1, shape[-2], shape[-1])
        xs = self.pos_encoder(xs)

        xs = self.transformer(xs)
        # xs = self.mlp_output(xs[torch.arange(xs.size(0)).to(self.device), length - 1])
        xs = self.mlp_output(xs[:, -1])
        # xs = xs[:, -1]
        return xs.view(*shape[:-2], shape[-1])


class GCDR(torch.nn.Module):  # Conditional Diffusion Recommender Model
    def __init__(self, dataset: Dataset, device, args):
        super(GCDR, self).__init__()
        self.dataset = dataset
        self.device = device

        # Not in the paper
        self.skip_step = args.skip_step
        self.norm = args.norm

        # Won't change in general
        self.n_hidden = args.n_hidden
        self.n_negative = args.n_negative
        self.diffusion_steps = args.diffusion_steps

        # Sensitive Parameters
        self.uncondition_rate = args.uncondition_rate
        self.category_uncondition_rate = args.category_uncondition_rate
        self.tau = args.tau
        self.delta = args.delta
        
        self.lambda_rec_loss = args.lambda_rec_loss  # NOT IN THE PAPER
        self.lambda_mse_loss = args.lambda_mse_loss 
        self.lambda_user_loss = args.lambda_user_loss

        # coefficient of prior
        scale = args.scale / args.diffusion_steps
        beta_start = scale * args.beta_start + args.beta_base
        beta_end = scale * args.beta_end + args.beta_base
        if beta_end > 1:
            beta_end = 1 / args.diffusion_steps + args.beta_base
        self.betas = torch.linspace(beta_start, beta_end, args.diffusion_steps, device=device)
        alphas = 1.0 - self.betas
        self.alphas_bar = torch.cumprod(alphas, dim=0)
        self.alphas_bar_prev = torch.concat((torch.tensor([1.0], device=device), self.alphas_bar[:-1]))

        self.sqrt_alphas_bar = torch.sqrt(self.alphas_bar)
        self.sqrt_one_minus_alphas_bar = torch.sqrt(1.0 - self.alphas_bar)
        
        # coefficient of posterior
        self.p_mu_c1 = self.betas * torch.sqrt(self.alphas_bar_prev) / (1.0 - self.alphas_bar)
        self.p_mu_c2 = (1.0 - self.alphas_bar_prev) * torch.sqrt(alphas) / (1.0 - self.alphas_bar)
        self.p_sqrt_var = torch.sqrt(self.betas * (1.0 - self.alphas_bar_prev) / (1.0 - self.alphas_bar))
        
        # Approximator
        ## Condition Encoding
        self.user_embeddings = torch.nn.Embedding(dataset.n_users, args.n_hidden)
        self.c_l_mlp = torch.nn.Sequential(
            torch.nn.Linear(args.n_hidden, args.n_hidden),
            torch.nn.Sigmoid()
        )

        ## Guidance Encoding
        ### Recent Interaction Sequence
        self.item_embeddings = torch.nn.Embedding(dataset.n_items + 1, args.n_hidden, padding_idx=dataset.n_items)
        self.transformer = TransformerEncoder(args.n_hidden, device, args.n_head, args.n_layer)
        self.none_embedding = torch.nn.Embedding(1, args.n_hidden)

        ### Category Preference
        self.g_c_mlp = torch.nn.Sequential(
            torch.nn.Dropout(args.dropout_g_c),
            torch.nn.Linear(len(self.dataset.cat2id), args.n_hidden),
            torch.nn.Sigmoid()
        )
        self.category_none_embedding = torch.nn.Embedding(1, args.n_hidden)

        ## Timestep Embedding
        self.timestep_embeddings = self.get_timestep_embeddings(torch.arange(args.diffusion_steps, device=device), args.n_hidden)

        self.time_embed = torch.nn.Sequential(
            torch.nn.Linear(args.n_hidden, args.n_hidden), 
            torch.nn.SiLU(),
            torch.nn.Linear(args.n_hidden, args.n_hidden)
        )

        ## xt
        self.xt_mlp = torch.nn.Sequential(
            torch.nn.Dropout(0.3),
            torch.nn.Linear(args.n_hidden, args.n_hidden, bias=True),
            torch.nn.Sigmoid(),
        )

        # Fusing Layer
        self.fusing_layer = torch.nn.Sequential(
            torch.nn.Linear(args.n_hidden * 5, args.n_hidden),
            torch.nn.Sigmoid(),
        )
        
        # diffusion step sampler
        self.diffusion_step_sampler = create_named_schedule_sampler('lossaware', args.diffusion_steps)

        # diffusion mse loss weight
        self.mse_weight = (self.alphas_bar_prev / (1 - self.alphas_bar_prev) - self.alphas_bar / (1 - self.alphas_bar)) / 2

        # reverse steps
        self.reverse_steps = list(range(self.diffusion_steps))[::-self.skip_step]

        self.to(device)

    def get_timestep_embeddings(self, timesteps, dim, max_period=10000):
        half = dim // 2
        freqs = torch.exp(
            -torch.log(torch.tensor(max_period)) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=timesteps.device)
        args = timesteps[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def get_item_embeddings(self, items):
        item_embeddings = self.item_embeddings(torch.LongTensor(items).to(self.device))
        if self.norm:
            item_embeddings = torch.nn.functional.normalize(item_embeddings, dim=-1)
        return item_embeddings,

    def get_original_user_embeddings(self, users):
        user_embeddings = self.user_embeddings(torch.LongTensor(users).to(self.device))
        if self.norm:
            user_embeddings = torch.nn.functional.normalize(user_embeddings, dim=-1)
        return user_embeddings
    
    def evaluate_x0(self, xt, t, user_condition, condition, category_condition, training):
        bs = xt.size(0)
        emb_t = self.time_embed(self.timestep_embeddings[t])

        if training:
            # Unconditional Version
            mask = (torch.rand(bs, device=self.device) <= self.uncondition_rate).float().view(bs, 1)
            condition = (1 - mask) * condition + mask * self.none_embedding(torch.tensor([0], device=self.device))

            category_mask = (torch.rand(bs, device=self.device) <= self.category_uncondition_rate).float().view(bs, 1)
            category_condition = (1 - category_mask) * category_condition + category_mask * self.category_none_embedding(torch.tensor([0], device=self.device))

        # x0_hat = self.fusing_layer(torch.concat((xt, condition, category_condition, user_condition, emb_t), dim=-1))        
        
        x0_hat = self.fusing_layer(torch.concat((xt * self.delta, condition, category_condition, user_condition, emb_t), dim=-1))
        return x0_hat

    def get_user_embeddings(self, users, histories, lengths, x0=None, user_cat_hist=None, _=None):
        # User-specific Embedding -> Long-term Interest
        user_embeddings = self.get_original_user_embeddings(users)
        user_condition = self.c_l_mlp(user_embeddings)

        # Recent Interaction Sequence -> Short-term Interest
        history_embeddings = self.get_item_embeddings(histories)[0]
        lengths = torch.LongTensor(lengths).to(self.device)
        condition = self.transformer.forward(history_embeddings, lengths)

        # Category Distribution -> Category Preference
        if user_cat_hist is None:
            user_cat_hist = self.dataset.user_cat_hist
        
        user_cat_hist = user_cat_hist[torch.LongTensor(users).to(self.device)]
        category_condition = self.g_c_mlp(user_cat_hist)

        bs = len(histories)
        if x0 is not None:
            t, _ = self.diffusion_step_sampler.sample(bs, self.device)
            noise = torch.randn_like(x0)
            xt = self.sqrt_alphas_bar[t].unsqueeze(-1) * x0 + self.sqrt_one_minus_alphas_bar[t].unsqueeze(-1) * noise
            x0_hat = self.evaluate_x0(xt, t, user_condition, condition, category_condition, training=True)
        else:
            noise_xt = torch.rand((bs, self.n_hidden), device=self.device)
            for step in self.reverse_steps:
                t = torch.tensor([step] * bs, device=self.device)
                x0_hat = self.evaluate_x0(noise_xt, t, user_condition, condition, category_condition, training=False)
                model_mean = self.p_mu_c1[t].unsqueeze(-1) * x0_hat + self.p_mu_c2[t].unsqueeze(-1) * noise_xt
                model_sqrt_var = self.p_sqrt_var[t].unsqueeze(-1)
                noise = torch.randn_like(noise_xt)
                noise_xt = model_mean + model_sqrt_var * noise
            x0_hat = noise_xt
        return x0_hat, t, user_embeddings

    def forward_bpr(self, users, histories, lengths, pos_items):
        bs = len(pos_items)
        neg_items = torch.randint(0, self.dataset.n_items, (bs, self.n_negative))  # bs x n_negative
        pos_x0 = self.get_item_embeddings(pos_items)[0]
        neg_x0_list = self.get_item_embeddings(neg_items)[0]  # bs x n_negative x d

        # x0_hat, t, user_embeddings = self.get_user_embeddings(users, histories, lengths, pos_x0, x0_id=pos_items)
        x0_hat, t, user_embeddings = self.get_user_embeddings(users, histories, lengths, pos_x0)
        
        # User Loss
        pos_user_loss = -torch.mean(torch.log(torch.sigmoid(torch.sum(user_embeddings * pos_x0, dim=-1))  + 1e-24 ))
        neg_user_score = torch.sum(user_embeddings.view(bs, 1, -1) * neg_x0_list, dim=-1)  # bs x n_negative
        neg_user_loss = - torch.mean(torch.sum(torch.log(torch.sigmoid(-neg_user_score) + 1e-24), dim=-1))
        user_loss = pos_user_loss + neg_user_loss

        # Recommendation Loss
        tau_coeff = torch.where(t <= self.tau, 1.0, 0.0)
        # pos_loss = -torch.mean(tau_coeff * torch.log(torch.sigmoid(torch.sum(x0_hat * user_embeddings, dim=-1)) + 1e-24))  # TODO: !important
        pos_loss = -torch.mean(tau_coeff * torch.log(torch.sigmoid(torch.sum(x0_hat * pos_x0, dim=-1)) + 1e-24))
        neg_score = torch.sum(x0_hat.view(bs, 1, -1) * neg_x0_list, dim=-1)  # bs x n_negative
        neg_loss = -torch.mean(torch.sum(tau_coeff.view(bs, 1) * torch.log(torch.sigmoid(-neg_score) + 1e-24), dim=-1))
        loss = pos_loss + neg_loss

        # Diffusion Loss
        x0_ = pos_x0
        # x0_ = x0.detach()
        mse = torch.mean((x0_ - x0_hat) ** 2, dim=-1)
        weight = torch.where((t == 0), 1.0, self.mse_weight[t])
        mse_loss = torch.mean(mse * weight)
            
        return user_loss * self.lambda_user_loss, loss * self.lambda_rec_loss, mse_loss * self.lambda_mse_loss
        