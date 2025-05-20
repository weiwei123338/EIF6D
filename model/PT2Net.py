import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.rotation_utils import Ortho6d2Mat
from utils.transformer import Transformer

from modules import ModifiedResnet_prior, PointNet2MSG
from losses import SmoothL1Dis, ChamferDis, PoseDis,PoseDis2,PoseDis1


class PT2Net(nn.Module):
    def __init__(self, nclass=6):
        super(PT2Net, self).__init__()
        self.nclass = nclass
        self.rgb_extractor = ModifiedResnet_prior()
        self.pts_extractor = PointNet2MSG(radii_list=[[0.01, 0.02], [0.02, 0.04], [0.04, 0.08], [0.08, 0.16]])
        self.prior_extractor = PointNet2MSG(radii_list=[[0.05, 0.10], [0.10, 0.20], [0.20, 0.30], [0.30, 0.40]])  # new
        self.prior_transform = PriorTransformation(nclass)
        self.Estimator = Estimator()
        self.Supervisor = Supervisor()
        self.RotEstimator = RotEstimator()
    def forward(self, inputs):
        end_points = {}

        # assert False
        rgb = inputs['rgb']
        pts = inputs['pts']
        choose = inputs['choose']

        prior = inputs['prior']  # new

        cls = inputs['category_label'].reshape(-1)

        c = torch.mean(pts, 1, keepdim=True)
        pts = pts - c

        b = pts.size(0)
        index = cls + torch.arange(b, dtype=torch.long).cuda() * self.nclass

        # rgb feat
        rgb_local, rgb_local_c64 = self.rgb_extractor(rgb)  # extract image level feature
        d0 = rgb_local.size(1)
        rgb_local = rgb_local.view(b, d0, -1)
        choose0 = choose.unsqueeze(1).repeat(1, d0, 1)
        rgb_local = torch.gather(rgb_local, 2, choose0).contiguous()

        d1 = rgb_local_c64.size(1)
        rgb_local_c64 = rgb_local_c64.view(b, d1, -1)
        choose1 = choose.unsqueeze(1).repeat(1, d1, 1)
        rgb_local_c64 = torch.gather(rgb_local_c64, 2, choose1).contiguous()
        # prior feat


        if self.training:
            
            delta_t1 = torch.rand(b, 1, 3).cuda()
            delta_t1 = delta_t1.uniform_(-0.02, 0.02)
            delta_r1 = torch.rand(b, 6).cuda()
            delta_r1 = Ortho6d2Mat(delta_r1[:, :3].contiguous(), delta_r1[:, 3:].contiguous()).view(-1, 3, 3)
            delta_s1 = torch.rand(b, 1).cuda()
            delta_s1 = delta_s1.uniform_(0.8, 1.2)
            pts1 = (pts - delta_t1) / delta_s1.unsqueeze(2) @ delta_r1

            
            pts1_local = self.pts_extractor(pts1)
            r_sup1, t_sup1, s_sup1 = self.Supervisor(pts1, rgb_local, pts1_local)
            pts_w1, pts_w_local_prior1, prior_trans1 = self.prior_transform(rgb_local, rgb_local_c64, pts1_local, pts1,
                                                                            prior, index)
            r1 = self.RotEstimator(pts1, pts_w1, rgb_local, pts1_local, pts_w_local_prior1)
            t1, s1 = self.Estimator(pts1, pts_w1, rgb_local, pts1_local, pts_w_local_prior1)

            end_points["pred_qo1"] = pts_w1
            end_points["prior_trans1"] = prior_trans1
            end_points['pred_rotation1'] = delta_r1 @ r1
            end_points['pred_translation1'] = delta_t1.squeeze(1) + delta_s1 * torch.bmm(delta_r1,
                                                                                         t1.unsqueeze(2)).squeeze(
                2) + c.squeeze(1)
            end_points['pred_size1'] = s1 * delta_s1
            end_points['pred_rotation_sup1'] = r_sup1
            end_points['pred_translation_sup1'] = delta_t1.squeeze(1) + delta_s1 * t_sup1 + c.squeeze(1)
            end_points['pred_size_sup1'] = s_sup1

          
            delta_t2 = torch.rand(b, 1, 3).cuda()
            delta_t2 = delta_t2.uniform_(-0.02, 0.02)
            delta_r2 = torch.rand(b, 6).cuda()
            delta_r2 = Ortho6d2Mat(delta_r2[:, :3].contiguous(), delta_r2[:, 3:].contiguous()).view(-1, 3, 3)
            delta_s2 = torch.rand(b, 1).cuda()
            delta_s2 = delta_s2.uniform_(0.8, 1.2)
            pts2 = (pts - delta_t2) / delta_s2.unsqueeze(2) @ delta_r2

            
            pts2_local = self.pts_extractor(pts2)
            r_sup2, t_sup2, s_sup2 = self.Supervisor(pts2, rgb_local, pts2_local)
            pts_w2, pts_w_local_prior2, prior_trans2 = self.prior_transform(rgb_local, rgb_local_c64, pts2_local, pts2,
                                                                            prior, index)
            r2 = self.RotEstimator(pts2, pts_w2, rgb_local, pts2_local, pts_w_local_prior2)
            t2, s2 = self.Estimator(pts2, pts_w2, rgb_local, pts2_local, pts_w_local_prior2)

            end_points["pred_qo2"] = pts_w2
            end_points["prior_trans2"] = prior_trans2
            end_points['pred_rotation2'] = delta_r2 @ r2
            end_points['pred_translation2'] = delta_t2.squeeze(1) + delta_s2 * torch.bmm(delta_r2,
                                                                                         t2.unsqueeze(2)).squeeze(
                2) + c.squeeze(1)
            end_points['pred_size2'] = s2 * delta_s2
            end_points['pred_rotation_sup2'] = r_sup2
            end_points['pred_translation_sup2'] = delta_t2.squeeze(1) + delta_s2 * t_sup2 + c.squeeze(1)
            end_points['pred_size_sup2'] = s_sup2

        else:
            pts_local = self.pts_extractor(pts)
            pts_w, pts_w_local_prior, prior_trans = self.prior_transform(rgb_local, rgb_local_c64, pts_local, pts, prior, index)
            r = self.RotEstimator(pts, pts_w, rgb_local, pts_local, pts_w_local_prior)
            t, s = self.Estimator(pts, pts_w, rgb_local, pts_local, pts_w_local_prior)
            end_points["pred_qo"] = pts_w
            end_points["prior_trans"] = prior_trans
            end_points['pred_rotation'] = r
            end_points['pred_translation'] = t + c.squeeze(1)
            end_points['pred_size'] = s

        return end_points


class SupervisedLoss(nn.Module):
    def __init__(self, cfg):
        super(SupervisedLoss, self).__init__()
        self.cfg = cfg.loss

    def forward(self, end_points):
        qo1 = end_points['pred_qo1']
        t1 = end_points['pred_translation1']
        r1 = end_points['pred_rotation1']
        s1 = end_points['pred_size1']
        loss1 = self._get_loss1(r1, t1, s1, qo1, end_points)

        qo2 = end_points['pred_qo2']
        t2 = end_points['pred_translation2']
        r2 = end_points['pred_rotation2']
        s2 = end_points['pred_size2']
        loss2 = self._get_loss2(r2, t2, s2, qo2, end_points)

        return loss1 + loss2

    def _get_loss1(self, r, t, s, qo, end_points):
        model = end_points['model']  
        r_sup = end_points['pred_rotation_sup1']
        t_sup = end_points['pred_translation_sup1']
        s_sup = end_points['pred_size_sup1']
        loss_qo = SmoothL1Dis(qo, end_points['qo'])
        loss_reconstrtucted = ChamferDis(end_points["prior_trans1"], model)  
        loss_pose1 = PoseDis1(t, s, end_points['translation_label'], end_points['size_label'])
        loss_pose2 = PoseDis2(r, end_points['rotation_label'])
        loss_pose_sup = PoseDis(r_sup, t_sup, s_sup, end_points['rotation_label'], end_points['translation_label'],
                                end_points['size_label'])
        cfg = self.cfg
        loss = loss_pose1 + loss_pose2 + loss_pose_sup + cfg.gamma1 * loss_qo + cfg.beta * loss_reconstrtucted  
        return loss

    def _get_loss2(self, r, t, s, qo, end_points):
        model = end_points['model']  
        r_sup = end_points['pred_rotation_sup2']
        t_sup = end_points['pred_translation_sup2']
        s_sup = end_points['pred_size_sup2']
        loss_qo = SmoothL1Dis(qo, end_points['qo'])
        loss_reconstrtucted = ChamferDis(end_points["prior_trans2"], model) 
        loss_pose1 = PoseDis1(t, s, end_points['translation_label'], end_points['size_label'])
        loss_pose2 = PoseDis2(r, end_points['rotation_label'])
        loss_pose_sup = PoseDis(r_sup, t_sup, s_sup, end_points['rotation_label'], end_points['translation_label'],
                                end_points['size_label'])
        cfg = self.cfg
        loss = loss_pose1 + loss_pose2 + loss_pose_sup + cfg.gamma1 * loss_qo + cfg.beta * loss_reconstrtucted  
        return loss

class UnSupervisedLoss(nn.Module):
    def __init__(self, cfg):
        super(UnSupervisedLoss, self).__init__()
        self.cfg = cfg.loss

    def forward(self, end_points):
        pts = end_points['pts']

        qo1 = end_points['pred_qo1']
        t1 = end_points['pred_translation1']
        r1 = end_points['pred_rotation1']
        s1 = end_points['pred_size1']
        prior_trans1 = end_points['prior_trans1']
        loss1 = self._get_loss1(pts, qo1, r1, t1, s1)

        qo2 = end_points['pred_qo2']
        t2 = end_points['pred_translation2']
        r2 = end_points['pred_rotation2']
        s2 = end_points['pred_size2']
        prior_trans2 = ['prior_trans2']
        loss2 = self._get_loss1(pts, qo2, r2, t2, s2)
        loss3 = self._get_loss2(qo1,r1,t1,s1,prior_trans1,qo2,r2,t2,s2,prior_trans2)

        return loss1 + loss2 + loss3

    def _get_loss1(self, pts, qo, r, t, s):


        scale = torch.norm(s, dim=1).reshape(-1, 1, 1) + 1e-8
        qo_ = ((pts - t.unsqueeze(1)) / scale) @ r
        return SmoothL1Dis(qo_, qo)


    def _get_loss2(self,qo1,r1,t1,s1,prior_trans1,qo2,r2,t2,s2,prior_trans2):
        loss_qo = torch.norm(qo1-qo2, dim=2).mean()
        loss_qv = ChamferDis(prior_trans1, prior_trans2)
        loss_pose = PoseDis(r1,t1,s1,r2,t2,s2)
        cfg = self.cfg
        loss = loss_pose + cfg.beta1 * loss_qv + cfg.beta2 * loss_qo
        return loss


class PriorTransformation(nn.Module):
    def __init__(self, nclass=6):
        super(PriorTransformation, self).__init__()
        self.nclass = nclass
        self.transformation = Transformation(nclass)

    def forward(self, rgb_local, rgb_local_c64, pts_local, pts, prior, index):
        pts_w_local_prior, pts_w, prior_trans = self.transformation(pts, rgb_local, rgb_local_c64, pts_local, prior, index)
        return pts_w, pts_w_local_prior, prior_trans


class Transformation(nn.Module):
    def __init__(self, nclass=6):
        super(Transformation, self).__init__()
        self.nclass = nclass
        self.prior_extractor = PointNet2MSG(radii_list=[[0.05, 0.10], [0.10, 0.20], [0.20, 0.30], [0.30, 0.40]])  
        self.transformer128 = Transformer(emb_dims=128, N=1)    

        self.pts_mlp1 = nn.Sequential(
            nn.Conv1d(3, 32, 1),
            nn.ReLU(),
            nn.Conv1d(32, 64, 1),
            nn.ReLU(),
        )

        self.deform_mlp1 = nn.Sequential(
            nn.Conv1d(64 + 256, 384, 1),
            nn.ReLU(),
            nn.Conv1d(384, 256, 1),
            nn.ReLU(),
        )

        self.deform_mlp1_prior = nn.Sequential(
            nn.Conv1d(384, 384, 1),
            nn.BatchNorm1d(384),
            nn.ReLU(),
            nn.Conv1d(384, 256, 1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Conv1d(256, 128, 1),
            nn.BatchNorm1d(128),
        )   #1，2
        # self.deform_mlp1_prior = nn.Sequential(
        #     nn.Conv1d(384, 384, 1),
        #     nn.ReLU(),
        #     nn.Conv1d(384, 256, 1),
        #     nn.ReLU(),
        #     nn.Conv1d(256, 128, 1),
        # )   #3

        self.deform_mlp2 = nn.Sequential(
            nn.Conv1d(512, 384, 1),
            nn.ReLU(),
            nn.Conv1d(384, 256, 1),
            nn.ReLU(),
            nn.Conv1d(256, 128, 1),
            nn.ReLU(),
        )

        self.deform_mlp3 = nn.Sequential(
            nn.Conv1d(384 + 256, 384, 1),
            nn.ReLU(),
            nn.Conv1d(384, 256, 1),
            nn.ReLU(),
            nn.Conv1d(256, 128, 1),
            nn.ReLU(),
        )   #!,3
        # self.deform_mlp3 = nn.Sequential(
        #     nn.Conv1d(384, 384, 1),
        #     nn.ReLU(),
        #     nn.Conv1d(384, 256, 1),
        #     nn.ReLU(),
        #     nn.Conv1d(256, 128, 1),
        #     nn.ReLU(),
        # )   #2

        self.category = nn.Sequential(
            nn.Conv1d(3, 64, 1),
            nn.ReLU(),
            nn.Conv1d(64, 64, 1),
            nn.ReLU(),
            nn.Conv1d(64, 128, 1),
            nn.ReLU(),
        )

        self.deformation = nn.Sequential(
            nn.Conv1d(128, 512, 1),
            nn.ReLU(),
            nn.Conv1d(512, 256, 1),
            nn.ReLU(),
            nn.Conv1d(256, nclass * 3, 1),
        )

        self.pred_nocs = nn.Sequential(
            nn.Conv1d(128, 256, 1),
            nn.ReLU(),
            nn.Conv1d(256, 128, 1),
            nn.ReLU(),
            nn.Conv1d(128, nclass * 3, 1),
        )

    def forward(self, pts, rgb_local, rgb_local_c64, pts_local, prior, index):
        npoint = pts_local.size(2)
        pts_pose_feat = self.pts_mlp1(pts.transpose(1, 2))

        deform_feat = torch.cat([
            pts_pose_feat,
            pts_local,
            rgb_local
        ], dim=1)

        pts_w_local = self.deform_mlp1(deform_feat)
        pts_w_global = torch.mean(pts_w_local, 2, keepdim=True)
        pts_w_local = torch.cat([pts_w_local, pts_w_global.expand_as(pts_w_local)], 1)
        pts_w_local = self.deform_mlp2(pts_w_local)
        
        inst_feature = torch.cat((pts_pose_feat, rgb_local_c64), dim=1)    

        cat_prior = prior.permute(0, 2, 1)    
        prior_feature = self.category(cat_prior)  
        inst_feature_p, prior_feature_p = self.transformer128(inst_feature, prior_feature)  
        deform_featp = prior_feature + prior_feature_p  
        deltas = self.deformation(deform_featp)  
        deltas = deltas.view(-1, 3, npoint).contiguous()
        deltas = torch.index_select(deltas, 0, index)
        deltas = deltas.permute(0, 2, 1).contiguous()
        prior_trans = prior + deltas 
        
        prior_local_trans = self.prior_extractor(prior_trans)

        rgb_global = torch.mean(rgb_local, 2, keepdim=True)
        pts_global = torch.mean(pts_local, 2, keepdim=True)

        deform_feat_prior = torch.cat([
            prior_local_trans,
            pts_global.repeat(1, 1, npoint),
            rgb_global.repeat(1, 1, npoint)
        ], dim=1)
        deform_feat_prior = self.deform_mlp1_prior(deform_feat_prior)

        prior_local_trans = F.relu(prior_local_trans + deform_feat_prior)  
        prior_global = torch.mean(prior_local_trans, 2, keepdim=True)
        pts_w_local_prior = torch.cat([pts_w_local, prior_local_trans, prior_global.expand_as(prior_local_trans), pts_local, rgb_local], 1)
        #pts_w_local_prior = torch.cat([pts_w_local, prior_local_trans, prior_global.expand_as(prior_local_trans)], 1)  
        pts_w_local_prior = self.deform_mlp3(pts_w_local_prior)
        #print(f"pts_w_local_prior shape: {pts_w_local_prior.shape}")
        # ==
        pts_w = self.pred_nocs(pts_w_local_prior)
        pts_w = pts_w.view(-1, 3, npoint).contiguous()
        pts_w = torch.index_select(pts_w, 0, index)
        pts_w = pts_w.permute(0, 2, 1).contiguous()

        return pts_w_local_prior, pts_w, prior_trans   


class Supervisor(nn.Module):
    def __init__(self):
        super(Supervisor, self).__init__()

        self.pts_mlp = nn.Sequential(
            nn.Conv1d(3, 32, 1),
            nn.ReLU(),
            nn.Conv1d(32, 64, 1),
            nn.ReLU(),
        )

        self.pose_mlp1 = nn.Sequential(
            nn.Conv1d(128 + 64 + 128, 256, 1),
            nn.ReLU(),
            nn.Conv1d(256, 256, 1),
            nn.ReLU(),
        )

        self.pose_mlp2 = nn.Sequential(
            nn.Conv1d(512, 512, 1),
            nn.ReLU(),
            nn.Conv1d(512, 512, 1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )

        self.rotation_estimator = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 6),
        )
        self.translation_estimator = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 3),
        )
        self.size_estimator = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 3),
        )

    def forward(self, pts, rgb_local, pts_local):
        pts = self.pts_mlp(pts.transpose(1, 2)) 
        pose_feat = torch.cat([rgb_local, pts, pts_local], dim=1) 

        pose_feat = self.pose_mlp1(pose_feat)   
        pose_global = torch.mean(pose_feat, 2, keepdim=True)    
        pose_feat = torch.cat([pose_feat, pose_global.expand_as(pose_feat)], 1) 
        pose_feat = self.pose_mlp2(pose_feat).squeeze(2)   

        r = self.rotation_estimator(pose_feat)  
        r = Ortho6d2Mat(r[:, :3].contiguous(), r[:, 3:].contiguous()).view(-1, 3, 3)  
        t = self.translation_estimator(pose_feat)   
        s = self.size_estimator(pose_feat)  
        return r, t, s


class Estimator(nn.Module):
    def __init__(self):
        super(Estimator, self).__init__()

        self.pts_mlp1 = nn.Sequential(
            nn.Conv1d(3, 32, 1),
            nn.ReLU(),
            nn.Conv1d(32, 64, 1),
            nn.ReLU(),
        )
        self.pts_mlp2 = nn.Sequential(
            nn.Conv1d(3, 32, 1),
            nn.ReLU(),
            nn.Conv1d(32, 64, 1),
            nn.ReLU(),
        )
        self.pose_mlp1 = nn.Sequential(
            nn.Conv1d(64 + 64 + 384, 256, 1),
            nn.ReLU(),
            nn.Conv1d(256, 256, 1),
            nn.ReLU(),
        )
        self.pose_mlp2 = nn.Sequential(
            nn.Conv1d(512, 512, 1),
            nn.ReLU(),
            nn.Conv1d(512, 512, 1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )

        self.translation_estimator = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 3),
        )
        self.size_estimator = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 3),
        )

    def forward(self, pts, pts_w, rgb_local, pts_local, pts_w_local_prior):
        pts = self.pts_mlp1(pts.transpose(1, 2))
        pts_w = self.pts_mlp2(pts_w.transpose(1, 2))

        pose_feat = torch.cat([rgb_local, pts, pts_local, pts_w, pts_w_local_prior], dim=1)
        pose_feat = self.pose_mlp1(pose_feat)
        pose_global = torch.mean(pose_feat, 2, keepdim=True)
        pose_feat = torch.cat([pose_feat, pose_global.expand_as(pose_feat)], 1)     
        pose_feat = self.pose_mlp2(pose_feat).squeeze(2)

          
        t = self.translation_estimator(pose_feat)   
        s = self.size_estimator(pose_feat)    
        return  t, s

class RotEstimator(nn.Module):
    def __init__(self):
        super(RotEstimator, self).__init__()

        self.pts_mlp1 = nn.Sequential(
            nn.Conv1d(3, 32, 1),
            nn.ReLU(),
            nn.Conv1d(32, 64, 1),
            nn.ReLU(),
        )
        self.pts_mlp2 = nn.Sequential(
            nn.Conv1d(3, 32, 1),
            nn.ReLU(),
            nn.Conv1d(32, 64, 1),
            nn.ReLU(),
        )
        self.pose_mlp1 = nn.Sequential(
            nn.Conv1d(64 + 64 + 384, 256, 1),
            nn.ReLU(),
            nn.Conv1d(256, 256, 1),
            nn.ReLU(),
        )
        self.pose_mlp2 = nn.Sequential(
            nn.Conv1d(512, 512, 1),
            nn.ReLU(),
            nn.Conv1d(512, 512, 1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.rotation_estimator = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 6),
        )


    def forward(self, pts, pts_w, rgb_local, pts_local, pts_w_local_prior): 
        pts = self.pts_mlp1(pts.transpose(1, 2))
        pts_w = self.pts_mlp2(pts_w.transpose(1, 2))

        pose_feat = torch.cat([rgb_local, pts, pts_local, pts_w, pts_w_local_prior], dim=1)
        pose_feat = self.pose_mlp1(pose_feat)
        pose_global = torch.mean(pose_feat, 2, keepdim=True)
        pose_feat = torch.cat([pose_feat, pose_global.expand_as(pose_feat)], 1)     
        pose_feat = self.pose_mlp2(pose_feat).squeeze(2)

        r = self.rotation_estimator(pose_feat)
        r = Ortho6d2Mat(r[:, :3].contiguous(), r[:, 3:].contiguous()).view(-1, 3, 3)    
        return r

