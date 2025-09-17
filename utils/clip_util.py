import clip
import torch
from nets.sdd_dkd import multi_dkd


def get_similarity(image_features, text_features):
    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)
    similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
    return similarity


def get_image_features(image, model, cpreprocess, device='cuda', need_preprocess=False):
    if need_preprocess:
        image = cpreprocess(image).unsqueeze(0).to(device)
    with torch.no_grad():
        image_features = model.encode_image(image)
    return image_features


def freeze_param(model):
    for name, param in model.named_parameters():
        param.requires_grad = False


def get_text_features_list(texts, model, device='cuda', train=False):
    if train:
        text_inputs = torch.cat([clip.tokenize(c)
                                for c in texts]).to(device)
        text_features = model.encode_text(text_inputs)
    else:
        with torch.no_grad():
            text_inputs = torch.cat([clip.tokenize(c)
                                     for c in texts]).to(device)
            text_features = model.encode_text(text_inputs)

    return text_features


def distillation_loss(student_features, teacher_features, temperature=1.0):
    """
    Compute the knowledge distillation loss between student and teacher features

    Args:
        student_features: Features from the student adapter
        teacher_features: Features from the teacher adapter
        temperature: Temperature parameter to soften the logits

    Returns:
        Distillation loss (MSE loss between normalized features)
    """
    # Normalize features
    student_features = student_features / student_features.norm(dim=-1, keepdim=True)
    teacher_features = teacher_features / teacher_features.norm(dim=-1, keepdim=True)

    # Use MSE loss for feature-based distillation
    return torch.nn.functional.mse_loss(student_features, teacher_features)


def sdd_dkd_distillation_loss(student_features, teacher_features, temperature=1.0, alpha=1.0, beta=8.0):
    """
    Compute knowledge distillation loss using SDD-DKD approach

    Args:
        student_features: Features from the student adapter
        teacher_features: Features from the teacher adapter
        temperature: Temperature parameter for distillation
        alpha: Weight for target-class distillation
        beta: Weight for non-target-class distillation

    Returns:
        SDD-DKD distillation loss
    """
    # Convert feature similarity to pseudo-logits
    batch_size = student_features.size(0)

    # Normalize features
    student_features = student_features / student_features.norm(dim=-1, keepdim=True)
    teacher_features = teacher_features / teacher_features.norm(dim=-1, keepdim=True)

    # Create pseudo-logits from cosine similarity between samples
    student_logits = student_features @ student_features.t() * temperature
    teacher_logits = teacher_features @ teacher_features.t() * temperature

    # Create pseudo targets (each sample is its own class)
    pseudo_targets = torch.arange(batch_size, device=student_features.device)

    # Generate patch-based features by splitting each feature vector
    feature_dim = student_features.size(1)
    num_patches = 3  # Split features into 3 parts to simulate patch-based features
    patch_size = feature_dim // num_patches

    student_patches = []
    teacher_patches = []

    for i in range(num_patches):
        start_idx = i * patch_size
        end_idx = start_idx + patch_size if i < num_patches - 1 else feature_dim

        s_patch = student_features[:, start_idx:end_idx]
        t_patch = teacher_features[:, start_idx:end_idx]

        # Create patch-level logits
        s_patch_logits = s_patch @ s_patch.t() * temperature
        t_patch_logits = t_patch @ t_patch.t() * temperature

        student_patches.append(s_patch_logits.unsqueeze(2))
        teacher_patches.append(t_patch_logits.unsqueeze(2))

    # Stack patches along new dimension [B, C, N] where N is number of patches
    student_patches = torch.cat(student_patches, dim=2)  # [B, B, N]
    teacher_patches = torch.cat(teacher_patches, dim=2)  # [B, B, N]

    # Apply multi-scale decoupled distillation
    loss = multi_dkd(
        student_patches,
        teacher_patches,
        pseudo_targets,
        alpha,
        beta,
        temperature
    )

    return loss
