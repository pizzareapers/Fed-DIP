import torch
import torch.nn as nn
import torch.nn.functional as F


class Distiller(nn.Module):
    def __init__(self, student, teacher):
        super(Distiller, self).__init__()
        self.student = student
        self.teacher = teacher

    def train(self, mode=True):
        # teacher as eval mode by default
        if not isinstance(mode, bool):
            raise ValueError("training mode is expected to be boolean")
        self.training = mode
        for module in self.children():
            module.train(mode)
        self.teacher.eval()
        return self

    def get_learnable_parameters(self):
        # if the method introduces extra parameters, re-impl this function
        return [v for k, v in self.student.named_parameters()]

    def get_extra_parameters(self):
        # calculate the extra parameters introduced by the distiller
        return 0

    def forward_train(self, **kwargs):
        # training function for the distillation method
        raise NotImplementedError()

    def forward_test(self, image):
        return self.student(image)[0]

    def forward(self, **kwargs):
        if self.training:
            return self.forward_train(**kwargs)
        return self.forward_test(kwargs["image"])


def dkd_loss_origin(logits_student, logits_teacher, target, alpha, beta, temperature):
    gt_mask = _get_gt_mask(logits_student, target)
    other_mask = _get_other_mask(logits_student, target)
    pred_student = F.softmax(logits_student / temperature, dim=1)
    pred_teacher = F.softmax(logits_teacher / temperature, dim=1)
    pred_student = cat_mask(pred_student, gt_mask, other_mask)
    pred_teacher = cat_mask(pred_teacher, gt_mask, other_mask)
    log_pred_student = torch.log(pred_student)
    tckd_loss = (
            F.kl_div(log_pred_student, pred_teacher, size_average=False)
            * (temperature ** 2)
            / target.shape[0]
    )
    pred_teacher_part2 = F.softmax(
        logits_teacher / temperature - 1000.0 * gt_mask, dim=1
    )
    log_pred_student_part2 = F.log_softmax(
        logits_student / temperature - 1000.0 * gt_mask, dim=1
    )
    nckd_loss = (
            F.kl_div(log_pred_student_part2, pred_teacher_part2, size_average=False)
            * (temperature ** 2)
            / target.shape[0]
    )
    return alpha * tckd_loss + beta * nckd_loss


def dkd_loss(logits_student, logits_teacher, target, alpha, beta, temperature):
    gt_mask = _get_gt_mask(logits_student, target)
    other_mask = _get_other_mask(logits_student, target)
    pred_student_probs = F.softmax(logits_student / temperature, dim=1)
    pred_teacher_probs = F.softmax(logits_teacher / temperature, dim=1)

    pred_student_cat = cat_mask(pred_student_probs, gt_mask, other_mask)
    pred_teacher_cat = cat_mask(pred_teacher_probs, gt_mask, other_mask)

    # Add a small epsilon to prevent log(0)
    log_pred_student_cat = torch.log(pred_student_cat + 1e-7)

    tckd_loss = (
            F.kl_div(log_pred_student_cat, pred_teacher_cat, reduction='none')
            * (temperature ** 2)
            / target.shape[0]
    )
    pred_teacher_part2 = F.softmax(
        logits_teacher / temperature - 1000.0 * gt_mask, dim=1
    )
    log_pred_student_part2 = F.log_softmax(
        logits_student / temperature - 1000.0 * gt_mask, dim=1
    )
    nckd_loss = (
            F.kl_div(log_pred_student_part2, pred_teacher_part2, reduction='none')
            * (temperature ** 2)
            / target.shape[0]
    )

    tckd_loss = torch.sum(tckd_loss, dim=1)
    nckd_loss = torch.sum(nckd_loss, dim=1)
    return alpha * tckd_loss + beta * nckd_loss


def multi_dkd(out_s_multi, out_t_multi, target, alpha, beta, temperature):
    """
    Multi-scale decoupling distillation loss

    Args:
        out_s_multi: Student patch predictions [B, C, N]
        out_t_multi: Teacher patch predictions [B, C, N]
        target: Target labels [B]
        alpha: Target-class distillation strength
        beta: Non-target-class distillation strength
        temperature: Temperature parameter

    Returns:
        Weighted distillation loss
    """
    # Handle case where batch size is 1 - expand dimensions
    if out_s_multi.dim() == 2:
        out_s_multi = out_s_multi.unsqueeze(2)
    if out_t_multi.dim() == 2:
        out_t_multi = out_t_multi.unsqueeze(2)

    # From B X C X N to N*B X C
    out_s_multi_t = out_s_multi.permute(2, 0, 1)
    out_t_multi_t = out_t_multi.permute(2, 0, 1)

    out_t = torch.reshape(out_t_multi_t, (out_t_multi_t.shape[0] * out_t_multi_t.shape[1], out_t_multi_t.shape[2]))
    out_s = torch.reshape(out_s_multi_t, (out_s_multi_t.shape[0] * out_s_multi_t.shape[1], out_s_multi_t.shape[2]))

    # Repeat targets for each patch
    target_r = target.repeat(out_t_multi.shape[2])

    # Calculate distillation loss
    loss = dkd_loss(out_s, out_t, target_r, alpha, beta, temperature)

    # Get teacher predictions for each patch
    out_t_predict = torch.argmax(out_t, dim=1)

    mask_true = out_t_predict == target_r
    mask_false = out_t_predict != target_r

    # Get global prediction (first patch)
    global_prediction = out_t_predict[0:len(target)]
    global_prediction_true_mask = global_prediction == target
    global_prediction_false_mask = global_prediction != target

    # Repeat masks for all patches
    global_prediction_true_mask_repeat = global_prediction_true_mask.clone().detach().repeat(out_t_multi.shape[2])
    global_prediction_false_mask_repeat = global_prediction_false_mask.clone().detach().repeat(out_t_multi.shape[2])

    # Global true local wrong
    mask_false[global_prediction_false_mask_repeat] = False
    mask_false[0:len(target)] = False
    gt_lw = mask_false

    # Global wrong local true
    mask_true[global_prediction_true_mask_repeat] = False
    mask_true[0:len(target)] = False
    gw_lt = mask_true

    # Reset masks
    mask_false = out_t_predict != target_r
    mask_true = out_t_predict == target_r

    # Global wrong local wrong
    mask_false[global_prediction_true_mask_repeat] = False
    gw_lw = mask_false

    # Global true local true
    mask_true[global_prediction_false_mask_repeat] = False
    gt_lt = mask_true

    # Create index tensor for weighted loss
    index = torch.zeros_like(loss).float()

    # Weight different cases differently
    index[gw_lw] = 1.0  # Global wrong, local wrong
    index[gt_lt] = 1.0  # Global true, local true
    index[gw_lt] = 2.0  # Global wrong, local true (complementary information)
    index[gt_lw] = 2.0  # Global true, local wrong (complementary information)

    # Weight and sum the loss
    loss = torch.sum(loss * index)

    # Handle potential numerical issues
    if torch.isnan(loss) or torch.isinf(loss):
        print("Warning: Loss is nan or inf, setting to zero")
        loss = torch.zeros(1).float().to(out_s_multi.device)

    return loss


def _get_gt_mask(logits, target):
    target = target.reshape(-1)
    mask = torch.zeros_like(logits).scatter_(1, target.unsqueeze(1), 1).bool()
    return mask


def _get_other_mask(logits, target):
    target = target.reshape(-1)
    mask = torch.ones_like(logits).scatter_(1, target.unsqueeze(1), 0).bool()
    return mask


def cat_mask(t, mask1, mask2):
    t1 = (t * mask1).sum(dim=1, keepdims=True)
    t2 = (t * mask2).sum(1, keepdims=True)
    rt = torch.cat([t1, t2], dim=1)
    return rt


class SDD_DKD(Distiller):
    def __init__(self, student, teacher, cfg):

        super(SDD_DKD, self).__init__(student, teacher)
        self.ce_loss_weight = cfg.ce_weight
        self.alpha = cfg.dkd_alpha
        self.beta = cfg.dkd_beta
        self.temperature = cfg.dkd_T
        self.warmup = cfg.sdd_dkd_warmup
        self.local_epochs = cfg.local_epochs

    def forward_train(self, logits_student, patch_s, logits_teacher, patch_t, labels, **kwargs):

        loss_sdd_dkd = min(self.local_epochs / self.warmup, 1.0) * multi_dkd(
            patch_s,
            patch_t,
            labels,
            self.alpha,
            self.beta,
            self.temperature,
        )
        return loss_sdd_dkd



def Distillation(logit_scale_student, logit_scale_teacher, text_proj_student, text_proj_teacher, cfg,
                 feature_map_student, student_global_logits, feature_map_teacher, teacher_global_logits,
                 student_text_encoder_output, teacher_text_encoder_output, sdd_dkd, true_labels): # Added true_labels

    feature_map_student = feature_map_student[:, 1:, :]  # [B, N, D_vis]
    feature_map_teacher = feature_map_teacher[:, 1:, :]  # [B, N, D_vis]

    text_embeddings_student = text_proj_student(student_text_encoder_output)  # [B, D_text] → [B, D_vis]
    text_embeddings_teacher = text_proj_teacher(teacher_text_encoder_output)  # [B, D_text] → [B, D_vis]

    feature_map_student = feature_map_student.permute(0, 2, 1)  # [B, D_vis, N]
    feature_map_teacher = feature_map_teacher.permute(0, 2, 1)  # [B, D_vis, N]

    text_embeddings_student = F.normalize(text_embeddings_student, dim=-1)  # [B, D_vis]
    text_embeddings_teacher = F.normalize(text_embeddings_teacher, dim=-1)  # [B, D_vis]
    feature_map_student = F.normalize(feature_map_student, dim=1)  # [B, D_vis, N]
    feature_map_teacher = F.normalize(feature_map_teacher, dim=1)  # [B, D_vis, N]

    # feature_map_student: [num_image, N, D_vis], num_image is batch size
    # text_embeddings_student: [num_text, D_vis], num_text is class_num (not batch_size)
    patch_logits_student = torch.einsum('b d n, t d -> b t n', feature_map_student,
                                        text_embeddings_student)  # [num_image, num_text, N], num_image is batch size , num_text is class_num
    patch_logits_teacher = torch.einsum('b d n, t d -> b t n', feature_map_teacher,
                                        text_embeddings_teacher)  # [num_image, num_text, N], num_image is batch size , num_text is class_num
    patch_logits_student = patch_logits_student * logit_scale_student
    patch_logits_teacher = patch_logits_teacher * logit_scale_teacher

    # batch_size = patch_logits_student.shape[0] # Not needed for arange anymore
    # labels = torch.arange(batch_size).cuda() # REMOVED THIS LINE

    # Use the provided true_labels
    loss_sdd_dkd = sdd_dkd.forward_train(student_global_logits, patch_logits_student, teacher_global_logits,
                                         patch_logits_teacher, true_labels)

    return loss_sdd_dkd