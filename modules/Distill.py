import torch
import torch.nn as nn
import numpy as np


from sklearn import cluster, metrics
import numpy as np
from sklearn.preprocessing import normalize
import time

def normalize(x, axis=None):
    """Normalizing to unit length along the specified dimension.
    Args:
      x: pytorch Variable
    Returns:
      x: pytorch Variable, same shape as input
    """
    y = 1. * x / (torch.norm(x, 2, axis, keepdim=True).expand_as(x) + 1e-12)
    return y

def sim_matrix_training(text_embeds, vid_embeds_pooled, pooling_type):
    """
    Computes the similarity matrix using pooled video frames
    
    Output
        sims: num_texts x num_vids
    """
    text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)
    vid_embeds_pooled = vid_embeds_pooled / vid_embeds_pooled.norm(dim=-1, keepdim=True)
    
    if pooling_type == 'avg':
        sims = torch.mm(text_embeds, vid_embeds_pooled.t())
        
    else:
        # num_texts x embed_dim x num_vids
        vid_embeds_pooled = vid_embeds_pooled.permute(1,2,0)
        # num_texts x 1 x embed_dim
        text_embeds = text_embeds.unsqueeze(1)
        
        sims = torch.bmm(text_embeds, vid_embeds_pooled).squeeze(1)

    return sims
    
def cosine_similarity(x1, x2, dim=-1, eps=1e8):
    w12 = torch.sum(x1*x2, dim)
    w1 = torch.norm(x1, 2, dim)
    w2 = torch.norm(x2, 2, dim)
    return (w12 / (w1 * w2).clamp(min=eps)).squeeze()
    
class Distill(nn.Module):
    def __init__(self, device):
        super(Distill, self).__init__()
        
        self.pa1 = torch.FloatTensor([1]).to(device)
        self.pa2 = torch.FloatTensor([1]).to(device)
        self.pa1.requires_grad = True
        self.pa2.requires_grad = True
        self.mask = self.pa1 * torch.eye(60).to(device) + self.pa2 * (torch.ones(60).to(device) - torch.eye(60).to(device))
        self.huberloss = nn.SmoothL1Loss(reduce=False)
        self.distill_criterion = nn.MSELoss(reduce=True, size_average=True)
        self.similarity_loss = nn.SmoothL1Loss(reduce=True, size_average=True)
        
        self.similarity_type = 'diag'
       
        
    def forward_adaptive_similarity(self, s1, s2):
        batchsize = self.mask.size(0)
        weight = F.softmax(self.mask,dim=0)
        reweight = torch.ones(batchsize,batchsize).to(device)/(torch.abs(s1).detach()+torch.ones(batchsize,batchsize).to(device)*1e-6)
        weight = reweight * weight
        loss = torch.sum(weight * self.huberloss(s1,s2)) * batchsize
        return loss
        
    def forward_loss_distill_similarity(self, s1, s2, *args, **kwargs):
        if self.similarity_type=='svd':
            a,b,c = torch.svd(s1)
            s1 = torch.matmul(a,torch.matmul(torch.diag(torch.log(b)),c))
            a,b,c = torch.svd(s2)
            s2 = torch.matmul(a,torch.matmul(torch.diag(torch.log(b)),c))
            loss = self.similarity_loss(s1, s2)
        elif self.similarity_type == 'eig':
            a,b = torch.eig(s1,eigenvectors=True)
            s1 = torch.matmul(b,torch.matmul(torch.diag(a[:,0]),torch.inverse(b)))
            a,b = torch.eig(s2,eigenvectors=True)
            s2 = torch.matmul(b,torch.matmul(torch.diag(a[:,0]),torch.inverse(b)))
            loss = self.similarity_loss(s1, s2)
        elif self.similarity_type == 'diag':
            loss = torch.sum(torch.diagonal(self.huberloss(s1,s2)))
        elif self.similarity_type == 'adapt':
            with torch.no_grad():
                batchsize = self.mask.size(0)
                weight = F.softmax(self.mask,dim=0)
                
            loss = torch.sum(weight.detach() * self.huberloss(s1,s2)) * batchsize
        elif self.similarity_type == 'maxdiag':
            loss = -torch.sum(torch.diagonal(s2))
        else:
            loss = self.similarity_loss(s1, s2)
        
        return loss
        
    def forward(self, s1, s2, flag=True):
        
        '''
        if flag==True:
           
           loss2 = self.forward_loss_distill_similarity(s1, s2)
           return loss2
           
        else:
           batch_v = mvsa_feature.shape[0]
           batch_t = aud_feature.shape[0]
           stu_img1 = mvsa_feature.unsqueeze(dim=1).expand(-1, batch_t, -1)
           stu_aud1 = aud_feature.unsqueeze(dim=0).expand(batch_v, -1, -1)
           loss2 = self.forward_adaptive_similarity(mvsa_feature1.detach(), Ft.detach(), stu_img1.detach(), stu_aud1.detach())
           return loss2
        '''
        s1 = normalize(s1)
        s2 = normalize(s2)
        loss2 = self.forward_loss_distill_similarity(s1, s2)
        return loss2
