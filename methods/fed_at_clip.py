import sys
import os
from config import ROOT_DIR, CHECKPOINT_DIR, DEFAULT_RESUME_PATH
from utils.config import img_param_init, set_random_seed
import utils.clip_util as clu
from utils.prepare_data_dg_clip import *
import copy
import argparse
from nets.models import ClipModelat
import torch.optim as optim
import torch.nn as nn
import torch
import numpy as np
from datetime import datetime


def toeval(model):
    model.model.eval()  # Changed clip_model to model
    model.invarient_adapter.eval()
    model.aware_adapter.eval()


def totrain(model):
    model.model.train()  # Changed clip_model to model
    model.invarient_adapter.train()
    model.aware_adapter.train()


def train_with_distillation(args, model, data_loader, optimizer, device, distillation_temp=1.0):
    """
    Training function with bidirectional distillation
    """
    # Phase 1: Freeze invariant adapter, train aware adapter
    model.freeze_invarient_adapter()
    model.unfreeze_aware_adapter()

    train_phase(args, model, data_loader, optimizer, device,
                is_invarient_teacher=True, distillation_temp=distillation_temp)

    # Phase 2: Freeze aware adapter, train invariant adapter
    model.freeze_aware_adapter()
    model.unfreeze_invarient_adapter()

    train_phase(args, model, data_loader, optimizer, device,
                is_invarient_teacher=False, distillation_temp=distillation_temp)


def train_phase(args, model, data_loader, optimizer, device, is_invarient_teacher=True, distillation_temp=1.0):
    """
    Single phase of training with distillation

    Args:
        is_invarient_teacher: If True, invarient adapter is the teacher, otherwise aware adapter is the teacher
    """
    model.model.train()  # Changed clip_model to model

    # Set the teacher adapter to eval mode
    if is_invarient_teacher:
        model.invarient_adapter.eval()
        model.aware_adapter.train()
    else:
        model.aware_adapter.eval()
        model.invarient_adapter.train()

    loss_img = nn.CrossEntropyLoss()
    # loss_txt = nn.CrossEntropyLoss() # Not needed for classification

    for batch in data_loader:
        image, _, label = batch # text is now a placeholder

        if len(image) > 1:
            image = image.to(device)
            # text = text.to(device) # Not needed
            label = label.to(device)  # Ensure label is on the correct device

            # Get text features from prompt learner
            prompts = model.prompt_learner()
            tokenized_prompts = model.tokenized_prompts
            text_features = model.encode_text(prompts, tokenized_prompts).float()

            # Get base features
            image_features = model.model.encode_image(image).float()  # Changed clip_model to model
            # text_features = model.model.encode_text(text).float()  # Changed clip_model to model # Replaced

            # Get teacher features (with detach to prevent gradient flow)
            if is_invarient_teacher:
                with torch.no_grad():
                    teacher_features = model.apply_invarient_adapter(image_features)
                student_features = model.apply_aware_adapter(image_features)
            else:
                with torch.no_grad():
                    teacher_features = model.apply_aware_adapter(image_features)
                student_features = model.apply_invarient_adapter(image_features)

            # Combined features for main task loss
            combined_features = (teacher_features.detach() + student_features) / 2

            # Normalize features
            combined_features = combined_features / combined_features.norm(dim=1, keepdim=True)
            text_features = text_features / text_features.norm(dim=1, keepdim=True)

            # Calculate main task loss
            logit_scale = model.model.logit_scale.exp()  # Changed clip_model to model
            logits_per_image = logit_scale * combined_features @ text_features.t()
            # logits_per_text = logits_per_image.t() # Not needed for classification

            # ground_truth = torch.arange(len(image), dtype=torch.long, device=device) # Incorrect for classification
            # The ground truth is the class label
            task_loss = loss_img(logits_per_image, label)

            # Calculate distillation loss using SDD-DKD
            if args.use_sdd_dkd:
                distill_loss = clu.sdd_dkd_distillation_loss(
                    student_features,
                    teacher_features.detach(),
                    temperature=distillation_temp,
                    alpha=args.sdd_dkd_alpha,
                    beta=args.sdd_dkd_beta
                )
            else:
                # Fallback to classic distillation if not using SDD-DKD
                distill_loss = clu.distillation_loss(
                    student_features,
                    teacher_features.detach(),
                    temperature=distillation_temp
                )

            # Combined loss (weighted sum of task loss and distillation loss)
            # The weight can be adjusted as a hyperparameter
            alpha = args.distill_alpha  # Changed fixed value to use argument
            loss = task_loss + alpha * distill_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


def train(args, model, data_loader, optimizer, device):
    # Replace the original train function with the bidirectional distillation version
    train_with_distillation(args, model, data_loader, optimizer, device,
                           distillation_temp=args.distill_temp)


def test(args, model, data_loader, device):
    toeval(model)
    model.prompt_learner.eval()
    total = 0
    correct = 0

    prompts = model.prompt_learner()
    tokenized_prompts = model.tokenized_prompts
    text_features = model.encode_text(prompts, tokenized_prompts).float()

    with torch.no_grad():
        for batch in data_loader:
            image, _, label = batch
            image = image.to(device)
            label = label.to(device)

            image_features = clu.get_image_features(image, model.model, model.preprocess).float()
            image_features = model.apply_dual_adapters(image_features).detach()

            similarity = clu.get_similarity(image_features, text_features)
            _, indices = similarity.topk(1)
            pred = torch.squeeze(indices, dim=1)

            # 计算当前 batch 的正确预测数
            correct += torch.sum(pred == label).item()
            total += len(label)

        return correct / total


def communication(args, server_model, models, client_weights):
    client_num = len(models)
    device = server_model.device
    with torch.no_grad():
        # Aggregate prompt_learner parameters
        for key in server_model.prompt_learner.state_dict().keys():
            temp = torch.zeros_like(server_model.prompt_learner.state_dict()[key], dtype=torch.float32, device=device)
            for client_idx in range(client_num):
                temp += client_weights[client_idx] * models[client_idx].prompt_learner.state_dict()[key]
            server_model.prompt_learner.state_dict()[key].data.copy_(temp)
            for client_idx in range(client_num):
                models[client_idx].prompt_learner.state_dict()[key].data.copy_(server_model.prompt_learner.state_dict()[key])

        # Handle invarient_adapter - aggregate across clients
        for key in server_model.invarient_adapter.state_dict().keys():
            if 'num_batches_tracked' in key or 'bert' in key:
                server_model.invarient_adapter.state_dict()[key].data.copy_(
                    models[0].invarient_adapter.state_dict()[key])
            else:
                temp = torch.zeros_like(server_model.invarient_adapter.state_dict()[
                                        key], dtype=torch.float32, device=device)
                for client_idx in range(client_num):
                    temp += client_weights[client_idx] * \
                        models[client_idx].invarient_adapter.state_dict()[key]
                server_model.invarient_adapter.state_dict()[key].data.copy_(temp)
                for client_idx in range(client_num):
                    models[client_idx].invarient_adapter.state_dict()[key].data.copy_(
                        server_model.invarient_adapter.state_dict()[key])

        # No aggregation for aware_adapter - keep parameters local to each client
        # The aware_adapter will continue to be trained locally without federation
    return server_model, models


if __name__ == '__main__':
    device = torch.device('cuda')
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='pacs',
                        choices=['pacs', 'office_home', 'vlcs', 'domain_net'])
    parser.add_argument('--lr', type=float, default=5e-5, help='learning rate')
    parser.add_argument('--prompt_lr', type=float, default=1e-4, help='learning rate for prompt learner')
    parser.add_argument('--datapercent', type=float,
                        default=6e-1, help='data percent to use')
    parser.add_argument('--batch', type=int, default=1024, help='batch size')
    parser.add_argument('--root_dir', type=str, default=ROOT_DIR)
    parser.add_argument('--iters', type=int, default=300,
                        help='iterations for communication')
    parser.add_argument('--wk_iters', type=int, default=2,
                        help='optimization iters in local worker between communication')
    parser.add_argument('--net', type=str, default='ViT-B/16',
                        help='ViT-B/16')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument('--beta1', type=float, default=0.9)
    parser.add_argument('--beta2', type=float, default=0.98)
    parser.add_argument('--eps', type=float, default=1e-6)
    parser.add_argument('--weight_decay', type=float, default=0.2)
    # New arguments for prompt tuning
    parser.add_argument('--n_ctx', type=int, default=16, help='number of context tokens')
    parser.add_argument('--ctx_init', type=str, default="a photo of a", help='init context')
    # New arguments for distillation
    parser.add_argument('--use_sdd_dkd', type=bool, default=True,
                        help='use SDD-DKD distillation instead of classic distillation')
    parser.add_argument('--sdd_dkd_alpha', type=float, default=1.0,
                        help='alpha parameter for SDD-DKD (target class weight)')
    parser.add_argument('--sdd_dkd_beta', type=float, default=8.0,
                        help='beta parameter for SDD-DKD (non-target class weight)')
    parser.add_argument('--distill_temp', type=float, default=1.0,
                        help='temperature for distillation')
    parser.add_argument('--distill_alpha', type=float, default=0.1,
                        help='weight for distillation loss')
    # New arguments for checkpointing and resuming
    parser.add_argument('--checkpoint_dir', type=str, default=CHECKPOINT_DIR,
                        help='directory to save checkpoints')
    parser.add_argument('--resume', type=str, default=DEFAULT_RESUME_PATH,
                        help='path to checkpoint to resume from')
    parser.add_argument('--save_freq', type=int, default=10,
                        help='frequency of saving checkpoints (in iterations)')
    args = parser.parse_args()
    args.random_state = np.random.RandomState(1)
    set_random_seed(args.seed)


    # Adjust n_clients based on the number of domains in each dataset
    if args.dataset == 'pacs' or args.dataset == 'office_home' or args.dataset == 'vlcs':
        args.n_clients = 4
    elif args.dataset == 'domain_net':
        args.n_clients = 6

    args = img_param_init(args)
    os.makedirs('../data/', exist_ok=True)

    # Create a unique directory for this run's checkpoints
    run_name = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{args.dataset}_lr{args.lr}_batch{args.batch}wk{args.wk_iters}prompt{args.prompt_lr}_distill{args.distill_alpha}"
    checkpoint_path = os.path.join(args.checkpoint_dir, run_name)
    os.makedirs(checkpoint_path, exist_ok=True)
    print(f"Checkpoints will be saved in: {checkpoint_path}")

    server_model = ClipModelat(
        args.net, imgadpy=True, freezepy=True)

    train_loaders, val_loaders, test_loaders = get_data(
        args.dataset)(args, server_model)

    server_model.initdgatal(train_loaders[0], args)

    client_num = len(test_loaders)
    client_weights = [1/client_num for i in range(client_num)]
    models = [copy.deepcopy(server_model)for idx in range(client_num)]
    for i in range(client_num):
        models[i].model.to(device)  # Changed clip_model to model
        models[i].invarient_adapter.to(device)
        models[i].aware_adapter.to(device)
        models[i].prompt_learner.to(device)
    best_changed = False

    best_acc = [0. for j in range(client_num)]
    best_test_accs = [0. for _ in range(client_num)]
    best_test_sources = ['' for _ in range(client_num)]
    finalrecord = ''
    logrecord = ''
    start_iter = 0
    best_epoch = 0

    optimizers = [optim.Adam(params=[{'params': models[idx].invarient_adapter.parameters()},
                                     {'params': models[idx].aware_adapter.parameters()},
                                     {'params': models[idx].prompt_learner.parameters(), 'lr': args.prompt_lr}],
                             lr=args.lr, betas=(args.beta1, args.beta2),
                             eps=args.eps, weight_decay=args.weight_decay) for idx in range(client_num)]

    if args.resume:
        if os.path.isfile(args.resume):
            print(f"=> loading checkpoint '{args.resume}'")
            checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
            start_iter = checkpoint['iter'] + 1

            # Load server model state
            server_model.prompt_learner.load_state_dict(checkpoint['server_model_prompt_learner'])
            server_model.invarient_adapter.load_state_dict(checkpoint['server_model_invarient_adapter'])

            for i in range(client_num):
                client_model_state = {
                    'prompt_learner': checkpoint[f'client_{i}_model_prompt_learner'],
                    'invarient_adapter': checkpoint[f'client_{i}_model_invarient_adapter'],
                    'aware_adapter': checkpoint[f'client_{i}_model_aware_adapter']
                }
                models[i].load_state_dict(client_model_state)
                optimizers[i].load_state_dict(checkpoint[f'optimizer_{i}'])

            best_acc = checkpoint['best_acc']
            best_test_accs = checkpoint['best_test_accs']
            best_test_sources = checkpoint['best_test_sources']
            best_epoch = checkpoint['best_epoch']
            logrecord = checkpoint['logrecord']
            finalrecord = checkpoint['finalrecord']
            print(f"=> loaded checkpoint '{args.resume}' (resuming from iter {start_iter})")
        else:
            print(f"=> no checkpoint found at '{args.resume}'")

    for a_iter in range(start_iter, args.iters):
        if a_iter == start_iter and not args.resume: # ensure optimizers are fresh if not resuming
            optimizers = [optim.Adam(params=[{'params': models[idx].invarient_adapter.parameters()},
                                             {'params': models[idx].aware_adapter.parameters()},
                                             {'params': models[idx].prompt_learner.parameters(), 'lr': args.prompt_lr}],
                                     lr=args.lr, betas=(args.beta1, args.beta2),
                                     eps=args.eps, weight_decay=args.weight_decay) for idx in range(client_num)]

        for wi in range(args.wk_iters):
            print("============ Train epoch {} ============".format(
                wi + a_iter * args.wk_iters))
            logrecord += 'Train epoch:%d\n' % (wi + a_iter * args.wk_iters)
            for client_idx, model in enumerate(models):
                models[client_idx].prompt_learner.train() # Set prompt learner to train
                train(
                    args, model, train_loaders[client_idx], optimizers[client_idx], device)
        with torch.no_grad():
            server_model, models = communication(
                args, server_model, models, client_weights)

            val_acc_list = [0. for j in range(client_num)]
            for client_idx, model in enumerate(models):
                train_acc = test(
                    args, model, train_loaders[client_idx], device)
                print(' Domain-{:s}| Train Acc: {:.4f}'.format(
                    args.domains[client_idx], train_acc))
                logrecord += ' Domain-{:s}| Train Acc: {:.4f}\n'.format(
                    args.domains[client_idx], train_acc)

                val_acc = test(
                    args, model, val_loaders[client_idx], device)
                val_acc_list[client_idx] = val_acc
                print(' Domain-{:s}| Val  Acc: {:.4f}'.format(
                    args.domains[client_idx], val_acc), flush=True)
                logrecord += ' Domain-{:s}| Val  Acc: {:.4f}\n'.format(
                    args.domains[client_idx], val_acc)

            print("============ Test Results ============")
            logrecord += "============ Test Results ============\n"
            current_test_accs = [0. for _ in range(client_num)]
            for test_domain_idx in range(client_num):
                max_acc = 0.0
                best_source_domain = ''
                for source_model_idx in range(client_num):
                    test_acc = test(
                        args, models[source_model_idx], test_loaders[test_domain_idx], device)
                    if test_acc > max_acc:
                        max_acc = test_acc
                        best_source_domain = args.domains[source_model_idx]

                current_test_accs[test_domain_idx] = max_acc
                if max_acc > best_test_accs[test_domain_idx]:
                    best_test_accs[test_domain_idx] = max_acc
                    best_test_sources[test_domain_idx] = best_source_domain

                print(' Test on {:s} | Best Acc: {:.4f} (from {:s})'.format(
                    args.domains[test_domain_idx], max_acc, best_source_domain))
                logrecord += ' Test on {:s} | Best Acc: {:.4f} (from {:s})\n'.format(
                    args.domains[test_domain_idx], max_acc, best_source_domain)

            if np.mean(val_acc_list) > np.mean(best_acc):
                for client_idx in range(client_num):
                    best_acc[client_idx] = val_acc_list[client_idx]
                best_epoch = a_iter
                best_changed = True

            if best_changed:
                finalrecord = finalrecord+str(a_iter)+','
                for item in current_test_accs:
                    finalrecord = finalrecord+str(item)+','
                best_changed = False

        if (a_iter + 1) % args.save_freq == 0 or a_iter == args.iters - 1:
            checkpoint_data = {
                'iter': a_iter,
                'server_model_prompt_learner': server_model.prompt_learner.state_dict(),
                'server_model_invarient_adapter': server_model.invarient_adapter.state_dict(),
                'best_acc': best_acc,
                'best_test_accs': best_test_accs,
                'best_test_sources': best_test_sources,
                'best_epoch': best_epoch,
                'logrecord': logrecord,
                'finalrecord': finalrecord,
                'args': args
            }
            for i in range(client_num):
                checkpoint_data[f'client_{i}_model_prompt_learner'] = models[i].prompt_learner.state_dict()
                checkpoint_data[f'client_{i}_model_invarient_adapter'] = models[i].invarient_adapter.state_dict()
                checkpoint_data[f'client_{i}_model_aware_adapter'] = models[i].aware_adapter.state_dict()
                checkpoint_data[f'optimizer_{i}'] = optimizers[i].state_dict()

            save_path = os.path.join(checkpoint_path, f'checkpoint_iter_{a_iter+1}.pth')
            torch.save(checkpoint_data, save_path)
            print(f"Checkpoint saved to {save_path}")

    print('best epoch:%d\n' % (best_epoch))
    logrecord += '\n best epoch:%d\n' % (best_epoch)

    print("============ Final Optimal Test Accuracies ============")
    logrecord += "============ Final Optimal Test Accuracies ============\n"
    ts = ''
    for i in range(client_num):
        domain_info = ' Test on {:s} | Best Acc: {:.4f} (from {:s})\n'.format(
            args.domains[i], best_test_accs[i], best_test_sources[i])
        print(domain_info, end='')
        logrecord += domain_info
        ts += '%.4f ' % best_test_accs[i]

    print('best test acc avg: {:.4f}'.format(np.mean(best_test_accs)))
    logrecord += 'best test acc avg: {:.4f}\n'.format(np.mean(best_test_accs))
    logrecord += 'best test accs: '+ts

