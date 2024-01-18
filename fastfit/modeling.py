import copy

from typing import Optional
from torch.autograd import Variable
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import (
    AutoModel,
    AutoConfig,
    AutoTokenizer,
    PreTrainedModel,
    AutoModelForMaskedLM,
)

from transformers.configuration_utils import PretrainedConfig
from transformers.utils import logging

logger = logging.get_logger(__name__)


class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""

    def __init__(self, temperature=0.07, contrast_mode="all", base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None, sim=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf
        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        device = torch.device("cuda") if features.is_cuda else torch.device("cpu")

        if len(features.shape) < 3:
            raise ValueError(
                "`features` needs to be [bsz, n_views, ...],"
                "at least 3 dimensions are required"
            )
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError("Cannot define both `labels` and `mask`")
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError("Num of labels does not match num of features")
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == "one":
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == "all":
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError("Unknown mode: {}".format(self.contrast_mode))

        # compute logits
        if sim is None:
            anchor_dot_contrast = torch.div(
                torch.matmul(anchor_feature, contrast_feature.T), self.temperature
            )
        else:
            anchor_dot_contrast = sim

        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0,
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        # log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-6)

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + 1e-6)

        # loss
        loss = -(self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss


def set_dropout(model, dropout):
    def _set_dropout(module, dropout):
        if isinstance(module, torch.nn.modules.dropout.Dropout):
            module.p = dropout
        else:
            if hasattr(module, "dropout") and isinstance(module.dropout, float):
                module.dropout = dropout
            if hasattr(module, "activation_dropout") and isinstance(
                module.activation_dropout, float
            ):
                module.activation_dropout = dropout

    _set_dropout(model, dropout)
    for module in model.modules():
        _set_dropout(module, dropout)


@dataclass
class ConfigArguments:
    num_repeats: Optional[int] = field(
        default=1,
        metadata={
            "help": "The number of times to repeat the queries and docs in every batch."
        },
    )
    proj_dim: Optional[int] = field(
        default=128, metadata={"help": "The dimension of the projection layer."}
    )
    clf_loss_factor: Optional[float] = field(
        default=0.1, metadata={"help": "The factor to scale the classification loss."}
    )
    pretrain_mode: Optional[bool] = field(
        default=False, metadata={"help": "Whether to do pre-training."}
    )
    inference_type: Optional[str] = field(
        default="sim", metadata={"help": "The inference type to be used."}
    )
    sim_rep: Optional[str] = field(
        default="reg",
        metadata={"help": "The similarity representation for training and inference."},
    )
    length_norm: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether to normalize by length while considering pad"},
    )
    sim_func_version: Optional[int] = field(
        default=2,
        metadata={"help": "Whether to normalize by length while considering pad"},
    )
    mlm_factor: Optional[float] = field(
        default=0.0, metadata={"help": "The factor to scale the MLM loss."}
    )
    mask_prob: Optional[float] = field(
        default=0.0, metadata={"help": "The probability of masking a token."}
    )


class FastFitConfig(PretrainedConfig):
    model_type = "FastFit"
    is_composition = True

    def __init__(
        self,
        all_docs=None,
        num_repeats=1,
        sim_rep="reg",
        inference_type="sim",
        clf_level="cls",
        init_freeze="reg",
        clf_dim=77,
        proj_dim=128,
        similarity_metric="cosine",
        encoder_dropout=None,
        encoding_type="dual",
        clf_query=True,
        clf_doc=False,
        mask_query=False,
        mask_doc=False,
        sim_factor=1.0,
        clf_factor=0.1,
        mlm_factor=0.0,
        mask_prob=0.15,
        pretrain_mode=False,
        length_norm=False,  # wether to noramlize scores by length
        scores_temp=0.07,
        inference_direction="doc",
        sim_func_version=2,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.all_docs = all_docs
        self.num_repeats = num_repeats
        self.sim_rep = sim_rep
        self.clf_level = clf_level
        self.init_freeze = init_freeze
        self.clf_dim = clf_dim
        self.proj_dim = proj_dim
        self.similarity_metric = similarity_metric
        self.encoder_dropout = encoder_dropout
        self.encoding_type = encoding_type
        self.clf_query = clf_query if not pretrain_mode else False
        self.clf_doc = clf_doc
        self.mask_query = mask_query
        self.mask_doc = mask_doc
        self.sim_factor = sim_factor
        self.clf_factor = clf_factor
        self.mlm_factor = mlm_factor
        self.inference_type = inference_type
        self.pretrain_mode = pretrain_mode
        self.length_norm = length_norm
        self.scores_temp = scores_temp
        self.inference_direction = inference_direction
        self.sim_func_version = sim_func_version
        self.mlm_prob = mask_prob
        self.mask_prob = mask_prob

        assert inference_direction in ["query", "doc", "both"]
        assert "encoder" in kwargs, "Config has to be initialized with encoder config"
        encoder_config = kwargs.pop("encoder")
        encoder_model_type = encoder_config.pop("model_type")
        self.encoder = AutoConfig.for_model(encoder_model_type, **encoder_config)
        self.hidden_size = self.encoder.hidden_size

    @classmethod
    def from_encoder_config(
        cls, encoder_config: PretrainedConfig, **kwargs
    ) -> PretrainedConfig:
        return cls(encoder=encoder_config.to_dict(), **kwargs)

    def to_dict(self):
        output = copy.deepcopy(self.__dict__)
        output["encoder"] = self.encoder.to_dict()
        output["model_type"] = self.__class__.model_type
        return output


class FastFit(PreTrainedModel):
    config_class = FastFitConfig
    base_model_prefix = "FastFit"

    def __init__(
        self,
        config: Optional[PretrainedConfig] = None,
        encoder: Optional[PreTrainedModel] = None,
    ):
        if config is None and encoder is None:
            raise ValueError(
                "Either a configuration or an encoder and a decoder has to be provided."
            )
        if config is None:
            config = FastFitConfig.from_encoder_config(encoder.config)
        else:
            if not isinstance(config, self.config_class):
                raise ValueError(
                    f"Config: {config} has to be of type {self.config_class}"
                )

        super().__init__(config)

        if encoder is None:
            encoder = AutoModel.from_config(config.encoder)

        self.encoder = encoder

        if self.encoder.config.to_dict() != self.config.encoder.to_dict():
            logger.warning(
                f"Config of the encoder: {self.encoder.__class__} is overwritten by shared encoder config:"
                f" {self.config.encoder}"
            )

        self.projection = nn.Linear(config.hidden_size, config.proj_dim, bias=False)
        self.inner_dim = config.hidden_size

        if config.clf_query:
            if config.clf_level == "cls":
                self.clf = nn.Linear(config.hidden_size, config.clf_dim)

            elif config.clf_level == "token":
                self.clf = nn.Linear(
                    self.inner_dim * config.head_token_size, config.clf_dim
                )

        self.dropout = nn.Dropout(0.1)  # follow the default in bert model
        self.batch_norm = nn.BatchNorm1d(num_features=self.inner_dim)

        if config.encoder_dropout is not None:
            set_dropout(self.encoder, config.encoder_dropout)

        self.skiplist = {}

        if config.mask_prob > 0.0:
            self.tokenizer = AutoTokenizer.from_pretrained(config.encoder._name_or_path)

        if config.mlm_factor > 0.0:
            self.mlm_criterion = nn.CrossEntropyLoss()
            self.lm_head = AutoModelForMaskedLM.from_pretrained(
                config.encoder._name_or_path
            ).lm_head

        if config.clf_factor > 0.0:
            self.clf_criterion = nn.CrossEntropyLoss()
        else:
            self.config.clf_doc = False
            self.config.clf_query = False

        if config.sim_factor > 0.0:
            self.sim_criterion = SupConLoss(temperature=0.07)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        from_tf = kwargs.pop("from_tf", False)
        if from_tf:
            raise ValueError("Loading a TensorFlow model in PyTorch is not supported.")

        # At the moment fast initialization is not supported for composite models
        if kwargs.get("_fast_init", False):
            logger.warning(
                "Fast initialization is currently not supported for EncoderDecoderModel. "
                "Falling back to slow initialization..."
            )
        kwargs["_fast_init"] = False

        return super().from_pretrained(
            pretrained_model_name_or_path, *model_args, **kwargs
        )

    @classmethod
    def from_encoder_pretrained(
        cls, encoder_pretrained_model_name_or_path: str = None, *model_args, **kwargs
    ) -> PreTrainedModel:
        kwargs_encoder = {
            argument[len("encoder_") :]: value
            for argument, value in kwargs.items()
            if argument.startswith("encoder_")
        }

        # remove encoder, decoder kwargs from kwargs
        for key in kwargs_encoder.keys():
            del kwargs["encoder_" + key]

        # Load and initialize the encoder and decoder
        # The distinction between encoder and decoder at the model level is made
        # by the value of the flag `is_decoder` that we need to set correctly.
        encoder = kwargs_encoder.pop("model", None)
        if encoder is None:
            if encoder_pretrained_model_name_or_path is None:
                raise ValueError(
                    "If `encoder_model` is not defined as an argument, a `encoder_pretrained_model_name_or_path` has "
                    "to be defined."
                )

            if "config" not in kwargs_encoder:
                encoder_config, kwargs_encoder = AutoConfig.from_pretrained(
                    encoder_pretrained_model_name_or_path,
                    **kwargs_encoder,
                    return_unused_kwargs=True,
                )

                if (
                    encoder_config.is_decoder is True
                    or encoder_config.add_cross_attention is True
                ):
                    logger.info(
                        f"Initializing {encoder_pretrained_model_name_or_path} as a encoder model "
                        "from a decoder model. Cross-attention and casual mask are disabled."
                    )
                    encoder_config.is_decoder = False
                    encoder_config.add_cross_attention = False

                kwargs_encoder["config"] = encoder_config

            encoder = AutoModel.from_pretrained(
                encoder_pretrained_model_name_or_path, *model_args, **kwargs_encoder
            )

        # instantiate config with corresponding kwargs
        if "config" not in kwargs:
            config = FastFitConfig.from_encoder_config(encoder.config, **kwargs)
        else:
            config = kwargs.pop("config")

        return cls(encoder=encoder, config=config)

    def get_encoder(self):
        return self.encoder

    def get_input_embeddings(self):
        return self.encoder.get_input_embeddings()

    def resize_token_embeddings(self, *args, **kwargs):
        embeddings = self.encoder.resize_token_embeddings(*args, **kwargs)
        self.config.encoder.vocab_size = embeddings.num_embeddings
        return embeddings

    def set_documetns(self, all_docs):
        self.all_docs = all_docs

    def inference_forward(self, query_input_ids, query_attention_mask):
        with torch.no_grad():
            doc_input_ids, doc_attention_mask = self.all_docs

            query_encodings, doc_encodings = self.encode(
                query_input_ids, query_attention_mask, doc_input_ids, doc_attention_mask
            )

            if self.config.inference_type == "sim":
                query_projections = self.project(
                    query_encodings, keep_dims=self.training
                )
                doc_projections = self.project(doc_encodings, keep_dims=self.training)

                if self.config.inference_direction == "query":
                    scores = self.tokens_similarity(
                        query_projections,
                        query_attention_mask,
                        doc_projections,
                        doc_attention_mask,
                    ).T
                elif self.config.inference_direction == "doc":
                    scores = self.tokens_similarity(
                        doc_projections,
                        doc_attention_mask,
                        query_projections,
                        query_attention_mask,
                    )
                elif self.config.inference_direction == "both":
                    first = self.tokens_similarity(
                        query_projections,
                        query_attention_mask,
                        doc_projections,
                        doc_attention_mask,
                    ).T
                    second = self.tokens_similarity(
                        doc_projections,
                        doc_attention_mask,
                        query_projections,
                        query_attention_mask,
                    )
                    scores = (first + second) / 2
                if self.config.sim_func_version == 1:
                    scores = (
                        (
                            doc_projections
                            @ query_projections.permute(0, 2, 1).unsqueeze(1)
                        )
                        .max(2)[0]
                        .sum(2)
                    )

            elif self.config.inference_type == "clf":
                _, scores = self.clf_loss(
                    query_encodings,
                    doc_encodings,
                    query_attention_mask,
                    doc_attention_mask,
                    return_scores=True,
                )

        return scores

    def prepare_inputs(
        self,
        query_input_ids,
        query_attention_mask,
        doc_input_ids,
        doc_attention_mask,
        labels=None,
    ):
        if self.config.num_repeats > 1:
            n = self.config.num_repeats
            query_input_ids = query_input_ids.repeat(n, 1)
            query_attention_mask = query_attention_mask.repeat(n, 1)
            doc_input_ids = doc_input_ids.repeat(n, 1)
            doc_attention_mask = doc_attention_mask.repeat(n, 1)
            labels = labels.repeat(n)

        mlm_labels = None
        if self.config.mask_prob > 0.0:
            query_input_ids, mlm_labels = self.mask_tokens(query_input_ids)

        return (
            query_input_ids,
            query_attention_mask,
            doc_input_ids,
            doc_attention_mask,
            labels,
            mlm_labels,
        )

    def forward(
        self,
        query_input_ids,
        query_attention_mask,
        doc_input_ids,
        doc_attention_mask,
        labels=None,
    ):
        scores = None
        if not self.training:
            scores = self.inference_forward(query_input_ids, query_attention_mask)

        (
            query_input_ids,
            query_attention_mask,
            doc_input_ids,
            doc_attention_mask,
            labels,
            mlm_labels,
        ) = self.prepare_inputs(
            query_input_ids,
            query_attention_mask,
            doc_input_ids,
            doc_attention_mask,
            labels,
        )

        query_encodings, doc_encodings = self.encode(
            query_input_ids, query_attention_mask, doc_input_ids, doc_attention_mask
        )

        total_loss = 0.0

        if self.config.sim_factor > 0.0:
            query_projections = self.project(query_encodings, keep_dims=self.training)
            doc_projections = self.project(doc_encodings, keep_dims=self.training)
            sim_loss = self.sim_loss(
                query_projections,
                doc_projections,
                query_attention_mask,
                doc_attention_mask,
                labels,
            )
            total_loss += sim_loss * self.config.sim_factor

        if self.config.mlm_factor > 0.0:
            mlm_loss = self.mlm_loss(query_encodings, mlm_labels)
            total_loss += mlm_loss * self.config.mlm_factor

        if self.config.clf_factor > 0.0 and not self.config.pretrain_mode:
            clf_loss = self.clf_loss(query_encodings, labels)
            total_loss += clf_loss * self.config.clf_factor

        return total_loss, scores

    def sim_loss(
        self,
        query_projections,
        doc_projections,
        query_attention_mask,
        doc_attention_mask,
        labels,
    ):
        sim_mat = None
        lbls = None
        bs = query_projections.size(0) // self.config.num_repeats  # original batch size
        Q, D = query_projections, doc_projections
        if self.config.pretrain_mode:
            if self.config.sim_rep == "reg":
                sim_mat = self.token_sim(Q[:bs, :, :], Q[bs:, :, :], bs)
                a, b = Q[:bs, :1, :], Q[bs:, :1, :]
            elif self.config.sim_rep == "cls":
                a, b = Q[:bs, :, :], Q[bs:, :, :]

        else:
            lbls = Variable(labels.type(torch.DoubleTensor), requires_grad=True)
            if self.config.sim_rep == "reg":
                if self.config.sim_func_version == 1:
                    sim_mat = self.token_sim1(Q, D, self.config.num_repeats * bs)
                if self.config.sim_func_version == 2:
                    sim_mat = self.token_sim2(
                        Q, D, query_attention_mask, doc_attention_mask
                    )
                a, b = Q[:, :1, :], D[:, :1, :]
            elif self.config.sim_rep == "cls":
                a, b = Q, D

        features = torch.cat((a, b), 1)
        sim_loss = self.sim_criterion(features=features, labels=lbls, sim=sim_mat)
        return sim_loss

    def clf_loss(self, encodings, labels, return_scores=False):
        if self.config.clf_level == "token":
            x = (encodings.last_hidden_state).permute(0, 2, 1)
            logits = (
                (
                    self.clf.weight.view(
                        self.config.head_token_size, self.inner_dim, -1
                    ).permute(2, 0, 1)
                    @ x.unsqueeze(1)
                )
                .max(2)
                .values.sum(2)
            )

        elif self.config.clf_level == "cls":
            if "deberta" in self.config.encoder.model_type:
                cls = encodings[0][:, 0, :]
            else:
                cls = encodings.pooler_output
            logits = self.clf(cls)
        else:
            raise ValueError("Unknown clf level: {}".format(self.config.clf_level))

        clf_scores = logits / self.config.clf_dim  # normalizing for colbert
        clf_loss = self.clf_criterion(clf_scores, labels)

        if return_scores:
            return clf_loss, clf_scores

        return clf_loss

    def mlm_loss(self, encodings, mlm_lables):
        sequence_output = encodings[0]
        prediction_scores = self.lm_head(sequence_output)
        vocab_size = (
            self.tokenizer.vocab_size
            if self.encoder.config.model_type != "deberta-v2"
            else self.tokenizer.vocab_size + 100
        )
        mlm_loss = self.mlm_criterion(
            prediction_scores.view(-1, vocab_size), mlm_lables.view(-1)
        )
        return mlm_loss

    def encode(
        self,
        query_input_ids,
        query_attention_mask,
        doc_input_ids,
        doc_attention_mask,
        do_zero_mask=False,
    ):
        if self.config.encoding_type == "dual":
            query_encodings = self.encode_one(query_input_ids, query_attention_mask)
            doc_input_ids, doc_attention_mask = (
                doc_input_ids.to(query_input_ids.device),
                doc_attention_mask.to(query_attention_mask.device),
            )
            doc_encodings = self.encode_one(doc_input_ids, doc_attention_mask)

        elif self.config.encoding_type == "cross":
            raise NotImplementedError("Cross encoder not implemented yet")
        else:
            raise ValueError(
                "Unknown encoding type: {}".format(self.config.encoding_type)
            )

        if do_zero_mask:
            query_encodings[0][query_attention_mask] = 0.0
            doc_encodings[0][doc_attention_mask] = 0.0

        return query_encodings, doc_encodings

    def encode_one(self, input_ids, attention_mask):
        encodings = self.encoder(input_ids, attention_mask=attention_mask)
        return encodings

    def project(self, encodings, keep_dims=True):
        if self.config.sim_rep == "reg":
            encoded = encodings[0]
            if keep_dims:
                encoded = self.dropout(encoded)
        elif self.config.sim_rep == "cls":
            cls = encodings  # TODO: fix this
            if keep_dims:
                cls = self.batch_norm(cls)
                cls = self.dropout(cls)
            encoded = cls.unsqueeze(1)

        encoded = self.projection(encoded)
        encoded = torch.nn.functional.normalize(encoded, p=2, dim=2)
        return encoded

    def score(self, Q, D):
        if self.config.similarity_metric == "cosine":
            return (Q @ D.permute(0, 2, 1)).max(2).values.sum(1)

        assert self.config.similarity_metric == "l2"
        return (
            (-1.0 * ((Q.unsqueeze(2) - D.unsqueeze(1)) ** 2).sum(-1))
            .max(-1)
            .values.sum(-1)
        )

    def mask(self, input_ids):
        mask = [
            [(x not in self.skiplist) and (x != 0) for x in d]
            for d in input_ids.cpu().tolist()
        ]
        return mask

    def token_sim1(self, Q_n, D_n, half_bsz):  # if token level in colbert forworad
        q_tokens = Q_n.shape[1]
        d_tokens = D_n.shape[1]
        Q_bsz = torch.cat((half_bsz * [Q_n]))
        D_bsz = torch.repeat_interleave(D_n, half_bsz, dim=0)
        colbert_score_bsz_QD = (
            (Q_bsz @ D_bsz.permute(0, 2, 1)).max(2).values.sum(1)
        ).view(half_bsz, -1).permute(1, 0) / q_tokens

        Q_bsz_2 = torch.repeat_interleave(Q_n, half_bsz, dim=0)
        colbert_score_bsz_QQ = (
            (Q_bsz @ Q_bsz_2.permute(0, 2, 1)).max(2).values.sum(1)
        ).view(half_bsz, -1).permute(1, 0) / q_tokens

        D_bsz_1 = torch.cat((half_bsz * [D_n]))
        colbert_score_bsz_DD = (
            (D_bsz_1 @ D_bsz.permute(0, 2, 1)).max(2).values.sum(1)
        ).view(half_bsz, -1).permute(1, 0) / d_tokens

        return (
            torch.cat(
                (
                    torch.cat((colbert_score_bsz_QQ, colbert_score_bsz_QD.T)),
                    torch.cat((colbert_score_bsz_QD, colbert_score_bsz_DD)),
                ),
                1,
            )
            / 0.07
        )  # div temp

    def tokens_similarity(self, B1, B1_mask, B2, B2_mask):
        tokens_sim = B1 @ B2.permute(0, 2, 1).unsqueeze(1)
        lens = B2.shape[1]

        if self.config.length_norm:
            B1_mask = B1_mask.type(B1.dtype).to(B1.device)
            B2_mask = B2_mask.type(B2.dtype).to(B2.device)
            tokens_mask = B1_mask.unsqueeze(-1) @ B2_mask.unsqueeze(-1).permute(
                0, 2, 1
            ).unsqueeze(1)  # [n_d, n_q, l_d, l_q]
            lens = tokens_mask.max(2)[0].sum(2)
            tokens_sim = tokens_sim * tokens_mask

        scores = tokens_sim.max(2)[0].sum(2)

        normalized_scores = scores / lens

        return normalized_scores / self.config.scores_temp

    def token_sim3(self, Q, D, Q_mask, D_mask, symetric_mode=True):
        # pad to longer sequence then concat
        max_length = max(Q.shape[1], D.shape[1])
        Q = F.pad(Q, (0, 0, 0, max_length - Q.shape[1]), "constant", 0)
        D = F.pad(D, (0, 0, 0, max_length - D.shape[1]), "constant", 0)
        Q_mask = F.pad(Q_mask, (0, max_length - Q_mask.shape[1]), "constant", 0)
        D_mask = F.pad(D_mask, (0, max_length - D_mask.shape[1]), "constant", 0)
        QD = torch.cat([Q, D], dim=0)
        QD_mask = torch.cat([Q_mask, D_mask], dim=0)
        sim = self.tokens_similarity(QD, QD_mask, QD, QD_mask)

        if symetric_mode:
            sim[Q.shape[0] :, : D.shape[0]] = sim[Q.shape[0] :, : D.shape[0]].T

        return sim

    def token_sim2(self, Q, D, Q_mask, D_mask, symetric_mode=True):
        # pad to longer sequence then concat

        QQ = self.tokens_similarity(Q, Q_mask, Q, Q_mask)
        QD = self.tokens_similarity(Q, Q_mask, D, D_mask)
        DD = self.tokens_similarity(D, D_mask, D, D_mask)
        DQ = QD.T if symetric_mode else self.tokens_similarity(D, D_mask, Q, Q_mask)

        sim = torch.cat((torch.cat((QQ, DQ)), torch.cat((QD, DD))), 1)
        return sim

    def mask_tokens(self, inputs, special_tokens_mask=None):
        """
        Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original.
        """
        labels = inputs.clone()

        probability_matrix = torch.full(labels.shape, self.config.mlm_prob)
        if special_tokens_mask is None:
            special_tokens_mask = [
                self.tokenizer.get_special_tokens_mask(
                    val, already_has_special_tokens=True
                )
                for val in labels.tolist()
            ]
            special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool)
        else:
            special_tokens_mask = special_tokens_mask.bool()

        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
        probability_matrix[torch.where(inputs == 0)] = 0.0
        masked_indices = torch.bernoulli(probability_matrix).bool()
        labels[~masked_indices] = -100  # We only compute loss on masked tokens

        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = (
            torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        )
        inputs[indices_replaced] = self.tokenizer.mask_token_id

        # 10% of the time, we replace masked input tokens with random word
        indices_random = (
            torch.bernoulli(torch.full(labels.shape, 0.5)).bool()
            & masked_indices
            & ~indices_replaced
        )
        indices_random = indices_random.to(inputs.device)
        random_words = torch.randint(
            len(self.tokenizer), labels.shape, dtype=torch.long
        ).to(inputs.device)
        inputs[indices_random] = random_words[indices_random]

        # The rest of the time (10% of the time) we keep the masked input tokens unchanged
        return inputs, labels


# main tests:
if __name__ == "__main__":
    model = FastFit.from_pretrined_encoder("roberta-base")
