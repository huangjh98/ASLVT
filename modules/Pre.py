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
    return tensor / tensor.norm(dim=-1, keepdim=True)

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
        n_embeds = 1
        self.num_embeds = n_embeds
        self.attention = MultiHeadSelfAttention(n_embeds, d_in, d_h)

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
        if self.num_embeds > 1:
            out = out.unsqueeze(1).repeat(1, self.num_embeds, 1)
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

class PIENet1(nn.Module):
    """Polysemous Instance Embedding (PIE) module"""

    def __init__(self, n_embeds, d_in, d_out, d_h, dropout=0.0):
        super(PIENet1, self).__init__()

        self.num_embeds = n_embeds
        self.attention = MultiHeadSelfAttention(n_embeds, d_in, d_h)
        self.fc = nn.Linear(d_in, d_out)
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_out)
        self.fc1 = nn.Linear(d_in, d_out)
        self.init_weights()
        

    def init_weights(self):
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.constant_(self.fc.bias, 0.0)

    def forward(self, out, x, pad_mask=None):
        residual, attn = self.attention(x, pad_mask)
        residual = self.dropout(self.sigmoid(self.fc(residual)))
        if self.num_embeds > 1:
            #eps = torch.randn(out.size(0), self.num_embeds, out.size(1), dtype=out.dtype, device=out.device)
            out = out.unsqueeze(1).repeat(1, self.num_embeds, 1)
            #out = eps.mul(torch.exp(out.unsqueeze(1)))
            
        
        out = self.layer_norm(out + residual)
        out = self.fc1(out)
        
        return out, attn, residual        

class PIENet(nn.Module):
    """Polysemous Instance Embedding (PIE) module"""

    def __init__(self, n_embeds, d_in, d_out, d_h, dropout=0.2):
        super(PIENet, self).__init__()

        self.num_embeds = n_embeds
        self.attention = MultiHeadSelfAttention(n_embeds, d_in, d_h)
        self.fc = nn.Linear(d_in, d_out)
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_in)  
        self.fc1 = nn.Linear(d_out, d_out)
        self.init_weights()

        self.noise_scale = nn.Parameter(torch.ones(1) * 0.1)

    def init_weights(self):
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.constant_(self.fc.bias, 0.0)

    def forward(self, out, x, pad_mask=None):
        
        residual, attn = self.attention(x, pad_mask)
        residual = self.sigmoid(self.fc(residual))
        residual = self.dropout(residual)  
        
        # Step 2: Layer normalization before residual connection
        
        #out = out.unsqueeze(1) + residual  
        out = torch.einsum('bfd,bd->bfd', [residual, out])

        out = self.layer_norm(out)
        '''
        if self.num_embeds > 1:
            
            noise = torch.randn(out.size(0), self.num_embeds, out.size(2), dtype=out.dtype, device=out.device)
            #out = out.unsqueeze(1).repeat(1, self.num_embeds, 1)
            out = out + self.noise_scale * noise  
        '''
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
          
          self.mean_head = FCN_head(mid_dim=[256, 256])
          self.var_head = FCN_head(mid_dim=[64])
          
          
      def probabilistic_video(self, video_pooled, videos):
          output = {} 
          
          out, attn, residual = self.pie_net_video(video_pooled, videos)        # (B 512) (B 12 512)   multiheadatt + fc + sigmoid + (residual) + laynorm 
        
          #uncertain_out = self.uncertain_net_video(video_pooled, videos)        # (B 512) (B 12 512)   multiheadatt + fc + (residual)         
          #out1 = uncertain_out['logsigma']#logsigma = uncertain_out['logsigma']
          
          feat_mean = self.mean_head(out)
          feat_var = torch.log1p(torch.exp(self.var_head(out)))
          #logsigma = feat_var#feat_mean / feat_var
          output['logsigma'] = feat_var       # B 512    
          output['mu'] = feat_mean
          
          out = l2_normalize(feat_mean / feat_var)     # B 512     
          output['embedding'] = out

           
          #output['embedding'] = sample_gaussian_tensors(l2_normalize(feat_mean), l2_normalize(logsigma), 7)      # B 7 512      

          return output


      def probabilistic_text(self, text_pooled, text_token):
          output = {}
          
          out, attn, residual = self.pie_net_text(text_pooled, text_token)     # (B 512) (B 32 512)   multiheadatt + fc + sigmoid + (residual) + laynorm
         
          #uncertain_out = self.uncertain_net_text(text_pooled, text_token)     # (B 512) (B 32 512)   multiheadatt + fc + (residual)   
          #out1 = uncertain_out['logsigma']#logsigma = uncertain_out['logsigma']
          
          feat_mean = self.mean_head(out)
          feat_var = torch.log1p(torch.exp(self.var_head(out)))
          #logsigma = feat_var#feat_mean / feat_var
          output['logsigma'] = feat_var
          output['mu'] = feat_mean

          out = l2_normalize(feat_mean / feat_var)
          output['embedding'] = out

          
          #output['embedding'] = sample_gaussian_tensors(l2_normalize(feat_mean), l2_normalize(logsigma), 7)
          
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
        
def js_div(p_output, q_output, get_softmax=True):
    """
    Function that measures JS divergence between target and output logits:
    """
    KLDivLoss = nn.KLDivLoss(reduction='batchmean')
    if get_softmax:
        p_output = F.softmax(p_output)
        q_output = F.softmax(q_output)
    log_mean_output = ((p_output + q_output )/2).log()
    return (KLDivLoss(log_mean_output, p_output) + KLDivLoss(log_mean_output, q_output))/2

class CossAtt(nn.Module):
    def __init__(self):
        super(CossAtt, self).__init__()
        self.temperature = 0.01

    def forward(self, video_embeds, text_embeds):
    
        # num_vids x 7 x 7
        sims = torch.bmm(video_embeds, text_embeds.permute(0, 2, 1))
        attention_weights = F.softmax(sims/self.temperature, dim=-1)
        

        # num_vids x 7 x embed_dim
        video_embeds_pooled = F.softmax(torch.bmm(attention_weights, video_embeds), dim=-1) * video_embeds
        return video_embeds_pooled
        
class KLdivergence(nn.Module):
    def __init__(self):
        super(KLdivergence, self).__init__()
        
        self.distill_criterion = nn.MSELoss()
        self.sl_loss = nn.SmoothL1Loss()
        self.ct = CossAtt()
    
    def kl_divergence(self, mu, logsigma):
        #mu = l2_normalize(mu)
        #logsigma = l2_normalize(logsigma)
        return -0.5 * (1 + logsigma - mu.pow(2) - logsigma.exp())
       
    def calculate_kl(self, mu_q, sig_q, mu_p, sig_p):
        kl = 0.5 * (2 * torch.log(sig_p / sig_q) - 1 + (sig_q / sig_p).pow(2) + ((mu_p - mu_q) / sig_p).pow(2)).sum() / mu_q.shape[0]
        return kl

    def forward(self, sampled_video_features, video_logsigma, sampled_text_features, text_logsigma):
        #vib_loss = self.kl_divergence(sampled_video_features, video_logsigma) + self.kl_divergence(sampled_text_features, text_logsigma)
        #vib_loss = self.distill_criterion(sampled_video_features, sampled_text_features)
        #vib_loss1 = (self.kl_divergence(sampled_video_features.mean(1), video_logsigma)).mean() #/ (video_logsigma + 1e-10) + torch.log(video_logsigma + 1e-10)).mean()
        #vib_loss2 = (self.kl_divergence(sampled_text_features.mean(1), text_logsigma)).mean() #/ (text_logsigma + 1e-10) + torch.log(text_logsigma + 1e-10)).mean()
        #video = sampled_video_features / video_logsigma
        #text = sampled_text_features / text_logsigma
        #vib_loss = self.sl_loss(video, text)
        vib_loss1 = self.calculate_kl(sampled_video_features, video_logsigma, sampled_text_features, text_logsigma)
        #vib_loss2 = self.calculate_kl(sampled_text_features, text_logsigma, sampled_video_features, video_logsigma)
        return vib_loss1#(vib_loss1 + vib_loss2) / 2 #vib_loss

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
        mm_mask = np.kron(mm_mask, la)     
        mm_mask = torch.tensor(mm_mask).float().bool()
        mm_mask = mm_mask.to(sim_matrix.device)
        
        sim_loss = - (F.log_softmax(sim_matrix, dim=1) * mm_mask).sum(1) / mm_mask.sum(1)
        #sim_loss = sim_loss.mean()
        return sim_loss

class CrossEn(nn.Module):
    def __init__(self,):
        super(CrossEn, self).__init__()

    def forward(self, sim_matrix):
        logpt = F.log_softmax(sim_matrix, dim=-1)
        logpt = torch.diag(logpt)
        nce_loss = -logpt
        sim_loss = nce_loss.mean()
        return sim_loss

def calcul_loss(scores, margin=0.1, loss_type="mse",max_violation=False, text_sim_matrix=None, param = "0.8 | 5"):

    diagonal = scores.diag().view(scores.shape[0], 1)
       
    d1 = diagonal.expand_as(scores)
      
    # compare every diagonal score to scores in its column
    # caption retrieval img--text
    cost_s = (margin + scores - d1).clamp(min=0)

    mask = (torch.eye(scores.size(0)) > .5).to(scores.device)

    cost_s = cost_s.masked_fill_(mask, 0)  

    if max_violation:
        cost_s = cost_s.max(1)[0]
    
    return cost_s.mean()

def cosine(x):
    x = l2_normalize(x, axis=1)
    distmat = torch.matmul(x, x.t())
    return distmat
        
class preLoss(nn.Module):
      def __init__(self):
          super(preLoss,self).__init__()
          self.kl = KLdivergence()
          self.p = pre()
          self.mil = MILNCELoss_BoF()
          self.miln = CrossEn()

      def forward(self, video_emb, video_token, text_emb, text_token):
          prob_video = self.p.probabilistic_video(video_emb, video_token.contiguous())
          #prob_video_embedding = prob_video['embedding']       
          prob_video_logsigma = prob_video['logsigma']         
          prob_video_mu = prob_video['mu']
          
          prob_text = self.p.probabilistic_text(text_emb, text_token.contiguous())
          #prob_text_embedding = prob_text['embedding']   # b n 512
          prob_text_logsigma = prob_text['logsigma']     # bs 512
          prob_text_mu = prob_text['mu']
          
          loss = self.kl(prob_video_mu, prob_video_logsigma, prob_text_mu, prob_text_logsigma)
          
          MIL_loss_v = torch.zeros(1).to(video_emb.device)
          MIL_loss_t = torch.zeros(1).to(text_emb.device)
          B = text_emb.shape[0]
          num_tokens = 7
          for i in range(B):
            
              MIL_loss_v = MIL_loss_v + torch.sum(torch.abs(cosine(prob_video["embedding"][i]) - torch.eye(num_tokens).to(video_emb.device))) / B / num_tokens / num_tokens
              MIL_loss_t = MIL_loss_t + torch.sum(torch.abs(cosine(prob_text["embedding"][i]) - torch.eye(num_tokens).to(text_emb.device))) / B / num_tokens / num_tokens
            
          MIL_loss = (MIL_loss_v + MIL_loss_t) / 2
          
          '''
          dim = 512
          bs = text_emb.shape[0]
          prob_sim_matrix_from_v = torch.einsum('ad,bd->ab', [prob_video["embedding"].view(-1, dim), prob_text["embedding"].view(-1, dim)])
          MIL_loss_v = self.mil(prob_sim_matrix_from_v, bs, 7, 7).mean()
          prob_sim_matrix_from_t = torch.einsum('ad,bd->ab', [prob_text["embedding"].view(-1, dim), prob_video["embedding"].view(-1, dim)])    
          MIL_loss_t = self.mil(prob_sim_matrix_from_t, bs, 7, 7).mean()
          MIL_loss = (MIL_loss_v + MIL_loss_t) / 2
          '''
          '''          
          MIL_loss_v = torch.zeros(1).to(video_emb.device)
          MIL_loss_t = torch.zeros(1).to(text_emb.device)
          B = text_emb.shape[0]
          num_tokens = 7
          prob_sim_matrix_from_v = torch.einsum('bvd,btd->bvt', [prob_video["embedding"], prob_text["embedding"]])
          prob_sim_matrix_from_t = torch.einsum('btd,bvd->btv', [prob_text["embedding"], prob_video["embedding"]])
          for i in range(B):
            
              MIL_loss_v = MIL_loss_v + self.miln(prob_sim_matrix_from_v[i]) / B
              MIL_loss_t = MIL_loss_t + self.miln(prob_sim_matrix_from_t[i]) / B
            
          MIL_loss = (MIL_loss_v + MIL_loss_t) / 2 
          '''          
          
          return loss, MIL_loss, prob_video_logsigma, prob_text_logsigma
'''
a=torch.rand(32,512)
b=torch.rand(32,512)
a1=torch.rand(32,12,512)
b1=torch.rand(32,20,512)
print(preLoss()(a,a1,b,b1))
'''
