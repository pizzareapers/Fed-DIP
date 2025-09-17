import torch
import torch.nn as nn
import clip
from utils.clip_util import freeze_param, get_image_features
from classnames import *


class PromptLearner(nn.Module):
    def __init__(self, classnames, clip_model, n_ctx=16, ctx_init="a photo of a"):
        super().__init__()
        n_cls = len(classnames)
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]
        device = clip_model.token_embedding.weight.device

        if ctx_init:
            # use given words to initialize context vectors
            ctx_init = ctx_init.replace("_", " ")
            n_ctx = len(ctx_init.split(" "))
            prompt = clip.tokenize(ctx_init).to(device)
            with torch.no_grad():
                embedding = clip_model.token_embedding(prompt).type(dtype)
            ctx_vectors = embedding[0, 1 : 1 + n_ctx, :]
            prompt_prefix = ctx_init
        else:
            # random initialization
            ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype, device=device)
            nn.init.normal_(ctx_vectors, std=0.02)
            prompt_prefix = " ".join(["X"] * n_ctx)

        print(f'Initial context: "{prompt_prefix}"')
        print(f"Number of context words (tokens): {n_ctx}")

        self.ctx = nn.Parameter(ctx_vectors)  # to be optimized

        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(clip.tokenize(name)) for name in classnames]
        prompts = [prompt_prefix + " " + name + "." for name in classnames]

        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts]).to(device)
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)

        # These token vectors will be saved when in save_model(),
        # but they should be ignored in load_model() as we want to use
        # those computed using the current class names
        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx :, :])  # CLS, EOS

        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor

    def forward(self):
        ctx = self.ctx
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)

        prefix = self.token_prefix
        suffix = self.token_suffix

        prompts = torch.cat(
            [
                prefix,  # (n_cls, 1, dim)
                ctx,     # (n_cls, n_ctx, dim)
                suffix,  # (n_cls, *, dim)
            ],
            dim=1,
        )

        return prompts


class ClipModelat(object):

    CLIP_MODELS = [
        'ViT-B/16'
    ]

    def __init__(self, model_name='Vit-B/16', device='cuda', logger=None, imgadpy=True, freezepy=True):
        self.logger = logger
        if type(model_name) is int:
            model_name = self.index_to_model(model_name)
        self.model, self.preprocess = clip.load(
            model_name, device=device, jit=False)
        self.model.eval()
        self.model.to(device)
        self.model_name = model_name
        self.imgadpy = imgadpy
        self.freezepy = freezepy
        self.device = device

    def load_state_dict(self, state_dict):
        self.prompt_learner.load_state_dict(state_dict['prompt_learner'])
        self.invarient_adapter.load_state_dict(state_dict['invarient_adapter'])
        self.aware_adapter.load_state_dict(state_dict['aware_adapter'])
        # The base model is frozen and shared, so we don't typically need to load its state
        # unless it was also being trained. If it is, you would load it here.
        # self.model.load_state_dict(state_dict['model'])

    def encode_text(self, prompts, tokenized_prompts):
        """
        Custom text encoding to handle prompts from PromptLearner.
        """
        x = prompts + self.model.positional_embedding.type(self.model.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.model.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.model.ln_final(x).type(self.model.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.model.text_projection
        return x

    def initdgatal(self, dataloader, args):
        dataset_name = args.dataset
        if dataset_name == 'pacs':
            classnames = PACS_CLASSES
        elif dataset_name == 'office_home':
            classnames = OFFICEHOME_CLASSES
        elif dataset_name == 'vlcs':
            classnames = VLCS_CLASSES
        elif dataset_name == 'domain_net':
            classnames = DOMAINNET_CLASSES
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")

        self.prompt_learner = PromptLearner(classnames, self.model, n_ctx=args.n_ctx, ctx_init=args.ctx_init).to(self.device)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts

        for batch in dataloader:
            image, _, label = batch
            image = image.to(self.device)
            label = label.to(self.device)
            image_features = get_image_features(
                image, self.model, self.preprocess)
            break
        if self.freezepy:
            freeze_param(self.model)

        if self.imgadpy:
            # Replace single adapter with dual parallel adapters
            self.invarient_adapter = nn.Sequential(
                nn.Linear(image_features.shape[1], image_features.shape[1]),
                nn.Tanh(),
                nn.Linear(image_features.shape[1], image_features.shape[1]),
                nn.Softmax(dim=1)
            ).to(self.device)

            self.aware_adapter = nn.Sequential(
                nn.Linear(image_features.shape[1], image_features.shape[1]),
                nn.ReLU(),
                nn.Linear(image_features.shape[1], image_features.shape[1]),
                nn.Softmax(dim=1)
            ).to(self.device)

    def apply_dual_adapters(self, image_features):
        """Apply both adapters and combine their outputs"""
        features_attn1 = self.invarient_adapter(image_features)
        features_attn2 = self.aware_adapter(image_features)

        # Combine the outputs (element-wise average)
        combined_features = (features_attn1 + features_attn2) / 2
        return torch.mul(combined_features, image_features)

    # New methods for bidirectional distillation
    def apply_invarient_adapter(self, image_features):
        """Apply only the invarient adapter for distillation"""
        features_attn = self.invarient_adapter(image_features)
        return torch.mul(features_attn, image_features)

    def apply_aware_adapter(self, image_features):
        """Apply only the aware adapter for distillation"""
        features_attn = self.aware_adapter(image_features)
        return torch.mul(features_attn, image_features)

    def freeze_invarient_adapter(self):
        """Freeze parameters of the invarient adapter"""
        for param in self.invarient_adapter.parameters():
            param.requires_grad = False

    def unfreeze_invarient_adapter(self):
        """Unfreeze parameters of the invarient adapter"""
        for param in self.invarient_adapter.parameters():
            param.requires_grad = True

    def freeze_aware_adapter(self):
        """Freeze parameters of the aware adapter"""
        for param in self.aware_adapter.parameters():
            param.requires_grad = False

    def unfreeze_aware_adapter(self):
        """Unfreeze parameters of the aware adapter"""
        for param in self.aware_adapter.parameters():
            param.requires_grad = True

    def index_to_model(self, index):
        return self.CLIP_MODELS[index]

    @staticmethod
    def get_model_name_by_index(index):
        name = ClipModelat.CLIP_MODELS[index]
        name = name.replace('/', '_')
        return name

    def setselflabel(self, labels):
        self.labels = labels
