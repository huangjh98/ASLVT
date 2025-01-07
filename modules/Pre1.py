""" Uncertainty modules
Reference code:
    PIENet in
    https://github.com/yalesong/pvse/blob/master/model.py
"""
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

import numpy as np

import torch
import torch.nn.functional as F

def l2_normalize(tensor, axis=-1):
    """L2-normalize columns of tensor"""
    return F.normalize(tensor, p=2, dim=axis)

class MultiHeadSelfAttention(nn.Module):
    """Self-attention module by Lin, Zhouhan, et al. ICLR 2017"""

    def __init__(self, n_head, d_in, d_hidden):
        super(MultiHeadSelfAttention, self).__init__()

        self.n_head = n_head
        self.w_1 = nn.Linear(d_in, d_hidden, bias=False)
        self.w_2 = nn.Linear(d_hidden, n_head, bias=False)
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=1)
        self.init_weights()

    def init_weights(self):
        nn.init.xavier_uniform_(self.w_1.weight)
        nn.init.xavier_uniform_(self.w_2.weight)

    def forward(self, x, mask=None):
        # This expects input x to be of size (b x seqlen x d_feat)
        attn = self.w_2(self.tanh(self.w_1(x)))
        if mask is not None:
            mask = mask.repeat(self.n_head, 1, 1).permute(1, 2, 0)
            attn.masked_fill_(mask, -np.inf)
        attn = self.softmax(attn)

        output = torch.bmm(attn.transpose(1, 2), x)
        if output.shape[1] == 1:
            output = output.squeeze(1)
        return output, attn
        
def sample_gaussian_tensors(mu, logsigma, num_samples):
    eps = torch.randn(mu.size(0), num_samples, mu.size(1), dtype=mu.dtype, device=mu.device)

    #samples = eps.mul(torch.exp(logsigma.unsqueeze(1))).add_(mu.unsqueeze(1))
    samples = mu / logsigma.clamp(min=1e-10)
   
    
    return samples
    
class UncertaintyModuleImage(nn.Module):
    def __init__(self, d_in, d_out, d_h):
        super().__init__()

        self.attention = MultiHeadSelfAttention(1, d_in, d_h)

        self.fc = nn.Linear(d_in, d_out)
        self.sigmoid = nn.Sigmoid()
        self.init_weights()

        self.fc2 = nn.Linear(d_in, d_out)
        self.embed_dim = d_in

    def init_weights(self):
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.constant_(self.fc.bias, 0)

    def forward(self, out, x, pad_mask=None):
        residual, attn = self.attention(x, pad_mask)

        fc_out = self.fc2(out)
        out = self.fc(residual) + fc_out

        return {
            'logsigma': out,
            'attention': attn,
        }


class UncertaintyModuleText(nn.Module):
    def __init__(self, d_in, d_out, d_h):
        super().__init__()

        self.attention = MultiHeadSelfAttention(1, d_in, d_h)

        self.fc = nn.Linear(d_in, d_out)
        self.sigmoid = nn.Sigmoid()
        self.init_weights()

        self.rnn = nn.GRU(d_in, d_out // 2, bidirectional=True, batch_first=True)
        self.embed_dim = d_out

    def init_weights(self):
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.constant_(self.fc.bias, 0)

    def forward(self, out, x, pad_mask=None, lengths=None):
        residual, attn = self.attention(x, pad_mask)

        # Forward propagate RNNs
        packed = pack_padded_sequence(out, lengths.cpu(), batch_first=True)
        if torch.cuda.device_count() > 1:
            self.rnn.flatten_parameters()
        rnn_out, _ = self.rnn(packed)
        padded = pad_packed_sequence(rnn_out, batch_first=True)

        # Reshape *final* output to (batch_size, hidden_size)
        I = lengths.expand(self.embed_dim, 1, -1).permute(2, 1, 0) - 1
        gru_out = torch.gather(padded[0], 1, I).squeeze(1)

        out = self.fc(residual) + gru_out

        return {
            'logsigma': out,
            'attention': attn,
        }

class PIENet(nn.Module):
    """Polysemous Instance Embedding (PIE) module"""

    def __init__(self, n_embeds, d_in, d_out, d_h, dropout=0.0):
        super(PIENet, self).__init__()

        self.num_embeds = n_embeds
        self.attention = MultiHeadSelfAttention(n_embeds, d_in, d_h)
        self.fc = nn.Linear(d_in, d_out)
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_out)
        self.init_weights()
        self.fc1 = nn.Linear(d_in, d_out)

    def init_weights(self):
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.constant_(self.fc.bias, 0.0)

    def forward(self, out, x, pad_mask=None):
        residual, attn = self.attention(x, pad_mask)
        residual = self.dropout(self.sigmoid(self.fc(residual)))
        if self.num_embeds > 1:
            #out = out.unsqueeze(1).repeat(1, self.num_embeds, 1)
            eps = torch.randn(out.size(0), num_embeds, out.size(1), dtype=out.dtype, device=out.device)

            out = eps.mul(out.unsqueeze(1))
            
        out = self.layer_norm(out + residual)
        out = self.fc1(out)
        return out, attn, residual        
        
class pre(nn.Module):
      def __init__(self):
          super(pre,self).__init__()
          embed_dim = 512
          
          self.uncertain_net_video = UncertaintyModuleImage(embed_dim, embed_dim, embed_dim // 2)
          self.pie_net_video = PIENet(7, embed_dim, embed_dim, embed_dim // 2)
             
          self.pie_net_text = PIENet(7, embed_dim, embed_dim, embed_dim // 2)
          self.uncertain_net_text = UncertaintyModuleImage(embed_dim, embed_dim, embed_dim // 2)
          
          self.mean_head_video = FCN_head(mid_dim=[256, 256])
          self.var_head_video = FCN_head(mid_dim=[64])
          
          self.mean_head_text = FCN_head(mid_dim=[256, 256])
          self.var_head_text = FCN_head(mid_dim=[64])
          
      def probabilistic_video(self, video_pooled, videos):
          output = {} 
          
          out, attn, residual = self.pie_net_video(video_pooled, videos)        # (B 512) (B 12 512)   multiheadatt + fc + sigmoid + (residual) + laynorm 
        
          #uncertain_out = self.uncertain_net_video(video_pooled, videos)        # (B 512) (B 12 512)   multiheadatt + fc + (residual)         
          #out = uncertain_out['logsigma']#logsigma = uncertain_out['logsigma']
          
          feat_mean = self.mean_head_video(out)
          feat_var = torch.log1p(torch.exp(self.var_head_video(out)))
          logsigma = feat_var#feat_mean / feat_var
          output['logsigma'] = feat_var       # B 512     可以看作是方差

          #out = l2_normalize(out)     # B 512     l2 normalization后 均值
           

          output['embedding'] = sample_gaussian_tensors(feat_mean, logsigma, 7)      # B 7 512    从高斯分布中采样N个embedding  

          return output


      def probabilistic_text(self, text_pooled, text_token):
          output = {}
          
          out, attn, residual = self.pie_net_text(text_pooled, text_token)     # (B 512) (B 32 512)   multiheadatt + fc + sigmoid + (residual) + laynorm
         
          #uncertain_out = self.uncertain_net_text(text_pooled, text_token)     # (B 512) (B 32 512)   multiheadatt + fc + (residual)   
          #out = uncertain_out['logsigma']#logsigma = uncertain_out['logsigma']
          
          feat_mean = self.mean_head_text(out)
          feat_var = torch.log1p(torch.exp(self.var_head_text(out)))
          logsigma = feat_var#feat_mean / feat_var
          output['logsigma'] = feat_var

          #out = l2_normalize(out)
          
          output['embedding'] = sample_gaussian_tensors(feat_mean, logsigma, 7)
          
          return output

class FCN_head(nn.Module):
    def __init__(self, in_dim=512, mid_dim=[64], out_dim=512):
        super(FCN_head, self).__init__()
        BN_MOMENTUM = 0.1
        dim_list = [in_dim] + mid_dim + [out_dim]
        self.layer = nn.Sequential()
        self.bayes_count = len(dim_list) - 1
        
        for i in range(self.bayes_count):
            in_dim, out_dim = dim_list[i], dim_list[i + 1]
            self.layer.add_module('Linear_{}'.format(i),
                                  nn.Linear(in_dim, out_dim))
            if i < self.bayes_count - 1:
                self.layer.add_module('ReLU_{}'.format(i), nn.ReLU(inplace=True))

    def forward(self, x):
        return self.layer(x)
        
                
class KLdivergence(nn.Module):
    def __init__(self):
        super(KLdivergence, self).__init__()
        
        self.distill_criterion = nn.MSELoss()
    
    def kl_divergence(self, mu, logsigma):
        return -0.5 * (1 + logsigma - mu.pow(2) - logsigma.exp()).mean()
    
    def forward(self, sampled_video_features, video_logsigma, sampled_text_features, text_logsigma):
        #vib_loss = self.kl_divergence(sampled_video_features.mean(dim=1), video_logsigma) + self.kl_divergence(sampled_text_features.mean(dim=1), text_logsigma)
        vib_loss = self.distill_criterion(sampled_video_features, sampled_text_features)
        return vib_loss

class MILNCELoss_BoF(nn.Module):
    def __init__(self):
        super(MILNCELoss_BoF, self).__init__()
        # self.batch_size = batch_size
        # self.n_pair = n_pair
        torch_v = float(".".join(torch.__version__.split(".")[:2]))
        self.bool_dtype = torch.bool if torch_v >= 1.3 else torch.uint8

    def forward(self, sim_matrix, batch_size, n_video, n_text):
        if sim_matrix.size(0) // batch_size == n_video:     # from v
            la = np.ones((n_video, n_text))
        else:
            la = np.ones((n_text, n_video))

        mm_mask = np.eye(batch_size)
        mm_mask = np.kron(mm_mask, la)     # 克罗克内积
        mm_mask = torch.tensor(mm_mask).float().bool()
        mm_mask = mm_mask.to(sim_matrix.device)
        
        sim_loss = - (F.log_softmax(sim_matrix, dim=1) * mm_mask).sum(1) / mm_mask.sum(1)
        sim_loss = sim_loss.mean()
        return sim_loss
        
class preLoss(nn.Module):
      def __init__(self):
          super(preLoss,self).__init__()
          self.kl = KLdivergence()
          self.p = pre()
          self.mil = MILNCELoss_BoF()
          
      def forward(self, video_emb, video_token, text_emb, text_token):
          prob_video = self.p.probabilistic_video(video_emb, video_token.contiguous())
          prob_video_embedding = prob_video['embedding']       # 从分布中采样m个embedding
          prob_video_logsigma = prob_video['logsigma']         # 方差

          prob_text = self.p.probabilistic_text(text_emb, text_token.contiguous())
          prob_text_embedding = prob_text['embedding']   # b n 512
          prob_text_logsigma = prob_text['logsigma']     # bs 512
          
          
          loss = self.kl(prob_video_embedding, prob_video_logsigma, prob_text_embedding, prob_text_logsigma)
          
          dim = 512
          bs = prob_video_embedding.size(0)
          prob_sim_matrix_from_v = torch.einsum('ad,bd->ab', [prob_video_embedding.view(-1, dim), prob_text_embedding.view(-1, dim)])
          MIL_loss_v = self.mil(prob_sim_matrix_from_v, bs, 7, 7)
          prob_sim_matrix_from_t = torch.einsum('ad,bd->ab', [prob_text_embedding.view(-1, dim), prob_video_embedding.view(-1, dim)])     # 与.t()等价
          MIL_loss_t = self.mil(prob_sim_matrix_from_t, bs, 7, 7)
          MIL_loss = (MIL_loss_v + MIL_loss_t) / 2
            
          return loss, MIL_loss
'''
a=torch.rand(32,512)
b=torch.rand(32,512)
a1=torch.rand(32,12,512)
b1=torch.rand(32,20,512)
print(preLoss()(a,a1,b,b1))
'''
