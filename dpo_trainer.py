import copy
from typing import Callable, Dict, List, Optional

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from datasets import IterableDataset
from transformers import Seq2SeqTrainer



# Helper: entropy from logits

def entropy_from_logits(logits: torch.Tensor) -> torch.Tensor:
    """
    Compute token-level entropy.

    Args:
        logits: [batch, seq_len, vocab_size]

    Returns:
        entropy: [batch, seq_len]
    """
    log_probs = F.log_softmax(logits, dim=-1)
    probs     = torch.exp(log_probs)
    entropy   = -(probs * log_probs).sum(dim=-1)
    return entropy



# Custom Whisper DPO Trainer

class WhisperDPOTrainer(Seq2SeqTrainer):
    """
    Custom DPO Trainer for Whisper encoder-decoder models.

    DPO objective:

        L_DPO = -log sigmoid(
            beta * [
                log πθ(y_chosen | audio) - log πref(y_chosen | audio)
                -
                log πθ(y_rejected | audio) + log πref(y_rejected | audio)
            ]
        )

    Important:
        Use remove_unused_columns=False in Seq2SeqTrainingArguments.
    """

    def __init__(
        self,
        *args,
        ref_model=None,
        processor=None,
        beta: float = 0.1,
        precompute_ref_logps: bool = False,
        label_pad_token_id: int = -100,
        loss_types: Optional[List[str]] = None,
        loss_weights: Optional[List[float]] = None,
        label_smoothing: float = 0.0,
        length_normalize_logps: bool = True,
        log_dpo_metrics: bool = True,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.beta                   = beta
        self.precompute_ref_logps   = precompute_ref_logps
        self.label_pad_token_id     = label_pad_token_id
        self.label_smoothing        = label_smoothing
        self.length_normalize_logps = length_normalize_logps
        self.log_dpo_metrics        = log_dpo_metrics

        if self.label_smoothing >= 0.5:
            raise ValueError("label_smoothing must be < 0.5 for robust DPO.")

        # Processor / tokenizer
        if processor is not None:
            self.processor = processor
        elif hasattr(self, "processing_class") and self.processing_class is not None:
            self.processor = self.processing_class
        else:
            raise ValueError(
                "Please pass processor=WhisperProcessor(...) to WhisperDPOTrainer."
            )

        # Loss types
        if loss_types is None:
            loss_types = ["sigmoid"]
        if isinstance(loss_types, str):
            loss_types = [loss_types]
        self.loss_types = loss_types

        # Loss weights
        if loss_weights is None:
            loss_weights = [1.0] * len(self.loss_types)
        if len(loss_weights) != len(self.loss_types):
            raise ValueError(
                f"loss_weights length must match loss_types. "
                f"Got {len(loss_weights)} weights for {len(self.loss_types)} losses."
            )
        self.loss_weights = loss_weights

        # Reference model
        if ref_model is None:
            self.ref_model = copy.deepcopy(self.model)
        else:
            self.ref_model = ref_model

        if self.ref_model is not None:
            self.ref_model.eval()
            for param in self.ref_model.parameters():
                param.requires_grad = False
            self.ref_model.to(self.args.device)

        # Disable cache
        if hasattr(self.model.config, "use_cache"):
            self.model.config.use_cache = False
        if self.ref_model is not None and hasattr(self.ref_model.config, "use_cache"):
            self.ref_model.config.use_cache = False




    def _get_decoder_pad_token_id(self, model) -> int:
        pad_token_id = None
        if hasattr(self.processor, "tokenizer"):
            pad_token_id = self.processor.tokenizer.pad_token_id
        if pad_token_id is None:
            pad_token_id = getattr(model.config, "pad_token_id", None)
        if pad_token_id is None:
            pad_token_id = getattr(model.config, "eos_token_id", None)
        if pad_token_id is None:
            raise ValueError("Could not find pad_token_id for decoder labels.")
        return pad_token_id

    def _get_decoder_start_token_id(self, model) -> int:
        dst = getattr(model.config, "decoder_start_token_id", None)
        if dst is None:
            raise ValueError("Could not find decoder_start_token_id in model.config.")
        return dst

    def _shift_tokens_right(
        self,
        labels: torch.Tensor,
        pad_token_id: int,
        decoder_start_token_id: int,
    ) -> torch.Tensor:
        """
        labels:            [A, B, C, D]
        decoder_input_ids: [decoder_start, A, B, C]
        """
        shifted = labels.new_zeros(labels.shape)
        shifted[:, 1:] = labels[:, :-1].clone()
        shifted[:, 0]  = decoder_start_token_id
        shifted = shifted.masked_fill(shifted.eq(self.label_pad_token_id), pad_token_id)
        return shifted


    

    def _compute_log_probs(
        self,
        model,
        input_features: torch.Tensor,
        labels: torch.Tensor,
        return_logits: bool = False,
    ):
        """
        Returns sequence-level log π(y | audio).

        outputs.loss is always None here because we pass decoder_input_ids
        manually without labels to the model forward.
        SFT loss is computed explicitly via F.cross_entropy in compute_loss.
        """
        pad_token_id           = self._get_decoder_pad_token_id(model)
        decoder_start_token_id = self._get_decoder_start_token_id(model)

        decoder_input_ids = self._shift_tokens_right(
            labels, pad_token_id, decoder_start_token_id
        )

        outputs = model(
            input_features=input_features,
            decoder_input_ids=decoder_input_ids,
            use_cache=False,
        )

        logits = outputs.logits  # [B, L, vocab_size]

        target_labels = labels
        target_mask   = target_labels.ne(self.label_pad_token_id)
        safe_labels   = target_labels.masked_fill(
            target_labels.eq(self.label_pad_token_id), 0
        )

        log_probs       = F.log_softmax(logits, dim=-1)
        per_token_logps = log_probs.gather(
            dim=-1, index=safe_labels.unsqueeze(-1)
        ).squeeze(-1)
        per_token_logps = per_token_logps * target_mask

        token_lengths = target_mask.sum(dim=1).clamp(min=1)

        if self.length_normalize_logps:
            sequence_logps = per_token_logps.sum(dim=1) / token_lengths
        else:
            sequence_logps = per_token_logps.sum(dim=1)

        if return_logits:
            return sequence_logps, logits, target_labels, target_mask

        return sequence_logps


 

    def compute_ref_log_probs(self, inputs: Dict[str, torch.Tensor]):
        """
        Live computation against the frozen ref model.
        Only called when precomputed logps are absent from the batch.
        """
        if self.ref_model is None:
            raise ValueError(
                "ref_model is None but 'ref_chosen_logps' / 'ref_rejected_logps' "
                "are missing from this batch.\n"
                "Make sure:\n"
                "  1. _precompute_ref_logps() was called before training.\n"
                "  2. The data collator passes those columns through to the batch dict.\n"
                "  3. trainer.precompute_ref_logps = True is set after precomputation."
            )

        self.ref_model.eval()
        with torch.no_grad():
            ref_chosen_logps = self._compute_log_probs(
                self.ref_model, inputs["input_features"], inputs["chosen_labels"]
            )
            ref_rejected_logps = self._compute_log_probs(
                self.ref_model, inputs["input_features"], inputs["rejected_labels"]
            )

        return ref_chosen_logps, ref_rejected_logps




    def _precompute_ref_logps(
        self,
        dataset,
        batch_size: Optional[int] = None,
        desc: str = "Precomputing reference log probabilities",
        normalize_fn: Optional[Callable[[str], str]] = None,
    ):
        """
        Runs the frozen ref model once over the dataset and stores:
            ref_chosen_logps, ref_rejected_logps  (as new columns)

        Also pre-filters rows that the DPO collator would silently drop so
        len(dataset) == number of logps produced, preventing the add_column
        row-count mismatch error.

        Must be called BEFORE training/evaluation when precompute_ref_logps=True.
        The returned (filtered + annotated) dataset must be assigned back to
        trainer.train_dataset / trainer.eval_dataset.
        """

        if isinstance(dataset, IterableDataset):
            raise ValueError(
                "_precompute_ref_logps() does not support IterableDataset."
            )
        if self.ref_model is None:
            raise ValueError(
                "ref_model is None. Cannot precompute reference log probabilities."
            )

        if batch_size is None:
            batch_size = self.args.per_device_train_batch_size



        def _norm(text: str) -> str:
            return normalize_fn(text or "") if normalize_fn else (text or "").strip().lower()

        def _is_valid(example):
            chosen   = _norm(example.get("text",     ""))
            rejected = _norm(example.get("rejected", ""))
            return chosen != "" and rejected != "" and chosen != rejected

        n_before = len(dataset)
        dataset  = dataset.filter(_is_valid, desc="Pre-filtering before precompute")
        n_after  = len(dataset)

        if n_after < n_before:
            print(f"[precompute] Removed {n_before - n_after} invalid rows "
                  f"({n_before} → {n_after}).")

        # Drop stale columns
        stale = [
            c for c in ["ref_chosen_logps", "ref_rejected_logps"]
            if hasattr(dataset, "column_names") and c in dataset.column_names
        ]
        if stale:
            dataset = dataset.remove_columns(stale)

     
        # Wraps self.data_collator so each sample is processed individually.
        # This prevents one bad audio sample from silently shrinking the batch
        # and causing a row-count mismatch.

        def _safe_collate(features):
            results = []
            for f in features:
                out = self.data_collator([f])   # batch-of-1; raises on error
                results.append(out)

            keys  = results[0].keys()
            batch = {}
            for k in keys:
                tensors = [r[k] for r in results]   # each shape [1, ...]

                if k in ("chosen_labels", "rejected_labels"):
                    # Pad to the longest sequence in this mini-batch
                    max_len = max(t.size(1) for t in tensors)
                    padded  = []
                    for t in tensors:
                        gap = max_len - t.size(1)
                        if gap > 0:
                            t = torch.cat(
                                [t, t.new_full((1, gap), self.label_pad_token_id)],
                                dim=1,
                            )
                        padded.append(t)
                    batch[k] = torch.cat(padded, dim=0)   # [B, max_len]
                else:
                    batch[k] = torch.cat(tensors, dim=0)  # [B, ...]

            return batch

        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            collate_fn=_safe_collate,
            shuffle=False,
            num_workers=0,          # closures don't pickle cleanly with >0
            pin_memory=self.args.dataloader_pin_memory,
        )

  

        ref_chosen_logps_all   = []
        ref_rejected_logps_all = []

        self.ref_model.eval()

        for batch in tqdm(dataloader, desc=desc):
            batch = self._prepare_inputs(batch)
            with torch.no_grad():
                rc, rr = self.compute_ref_log_probs(batch)
            ref_chosen_logps_all.append(rc.detach().cpu())
            ref_rejected_logps_all.append(rr.detach().cpu())

        ref_chosen_logps_all   = torch.cat(ref_chosen_logps_all,   dim=0).float().tolist()
        ref_rejected_logps_all = torch.cat(ref_rejected_logps_all, dim=0).float().tolist()

    

        if len(ref_chosen_logps_all) != len(dataset):
            raise RuntimeError(
                f"Row-count mismatch after precompute: dataset has {len(dataset)} rows "
                f"but produced {len(ref_chosen_logps_all)} logps. "
                f"_safe_collate must have silently dropped a sample."
            )

        dataset = dataset.add_column("ref_chosen_logps",   ref_chosen_logps_all)
        dataset = dataset.add_column("ref_rejected_logps", ref_rejected_logps_all)

        return dataset



    def compute_loss(
        self,
        model,
        inputs,
        return_outputs: bool = False,
        num_items_in_batch=None,
    ):
        has_preference_batch = (
            "chosen_labels" in inputs and "rejected_labels" in inputs
        )

        # Fallback: plain CE (non-preference eval batches)
        if not has_preference_batch:
            labels  = inputs.get("labels", None)
            outputs = model(
                input_features=inputs["input_features"],
                labels=labels,
                use_cache=False,
            )
            loss = outputs.loss
            return (loss, outputs) if return_outputs else loss

        chosen_logps, chosen_logits, chosen_target_labels, chosen_mask = (
            self._compute_log_probs(
                model, inputs["input_features"], inputs["chosen_labels"],
                return_logits=True,
            )
        )
        rejected_logps, rejected_logits, rejected_target_labels, rejected_mask = (
            self._compute_log_probs(
                model, inputs["input_features"], inputs["rejected_labels"],
                return_logits=True,
            )
        )

        # Use precomputed columns when available; otherwise run live.
        has_precomputed = (
            "ref_chosen_logps"   in inputs
            and "ref_rejected_logps" in inputs
        )

        if has_precomputed:
            ref_chosen_logps = inputs["ref_chosen_logps"].to(
                device=chosen_logps.device, dtype=chosen_logps.dtype
            ).view(-1)
            ref_rejected_logps = inputs["ref_rejected_logps"].to(
                device=rejected_logps.device, dtype=rejected_logps.dtype
            ).view(-1)
        else:
            ref_chosen_logps, ref_rejected_logps = self.compute_ref_log_probs(inputs)

        chosen_logratios   = chosen_logps   - ref_chosen_logps
        rejected_logratios = rejected_logps - ref_rejected_logps
        delta_score        = chosen_logratios - rejected_logratios

     
        loss = chosen_logps.new_tensor(0.0)

        for loss_type, loss_weight in zip(self.loss_types, self.loss_weights):

            if loss_type == "sigmoid":
                label_smoothing = 0.3

                per_sequence_loss = (
                    -(1 - label_smoothing) * F.logsigmoid(self.beta * delta_score)
                    - label_smoothing * F.logsigmoid(-self.beta * delta_score))
            elif loss_type == "hinge":
                per_sequence_loss = torch.relu(1.0 - self.beta * delta_score)

            elif loss_type == "ipo":
                per_sequence_loss = (delta_score - 1.0 / (2.0 * self.beta)) ** 2

            elif loss_type == "robust":
                clean_loss   = -(1.0 - self.label_smoothing) * F.logsigmoid( self.beta * delta_score)
                flipped_loss = -self.label_smoothing           * F.logsigmoid(-self.beta * delta_score)
                per_sequence_loss = (clean_loss - flipped_loss) / (1.0 - 2.0 * self.label_smoothing)

            elif loss_type == "sft":
                # outputs.loss is None in the manual-shift path; compute CE
                # explicitly — identical to HuggingFace's internal pattern:
                #   loss_fct(lm_logits.view(-1, vocab_size), labels.reshape(-1))
                sft_loss = F.cross_entropy(
                    chosen_logits.view(-1, chosen_logits.size(-1)),  # [B*L, vocab]
                    chosen_target_labels.reshape(-1),                # [B*L]
                    ignore_index=self.label_pad_token_id,
                )
                per_sequence_loss = sft_loss.expand(chosen_logps.size(0))

            else:
                raise ValueError(
                    f"Unknown loss_type: '{loss_type}'. "
                    f"Supported: ['sigmoid', 'hinge', 'ipo', 'robust', 'sft']"
                )

            loss = loss + per_sequence_loss.mean() * loss_weight

        
        with torch.no_grad():
            chosen_rewards   = self.beta * chosen_logratios
            rejected_rewards = self.beta * rejected_logratios

            reward_accuracy = (chosen_rewards > rejected_rewards).float().mean()
            reward_margin   = (chosen_rewards - rejected_rewards).mean()

            chosen_entropy   = entropy_from_logits(chosen_logits.detach())
            rejected_entropy = entropy_from_logits(rejected_logits.detach())
            entropy_sum      = (
                (chosen_entropy  * chosen_mask).sum()
                + (rejected_entropy * rejected_mask).sum()
            )
            entropy = entropy_sum / (chosen_mask.sum() + rejected_mask.sum()).clamp(min=1)

            chosen_preds        = chosen_logits.detach().argmax(dim=-1)
            correct_tokens      = ((chosen_preds == chosen_target_labels) & chosen_mask.bool()).sum()
            mean_token_accuracy = correct_tokens.float() / chosen_mask.sum().clamp(min=1).float()

        if return_outputs:
            return loss, {
                "chosen_logps":        chosen_logps,
                "rejected_logps":      rejected_logps,
                "ref_chosen_logps":    ref_chosen_logps,
                "ref_rejected_logps":  ref_rejected_logps,
                "chosen_rewards":      chosen_rewards,
                "rejected_rewards":    rejected_rewards,
                "reward_accuracy":     reward_accuracy,
                "reward_margin":       reward_margin,
                "entropy":             entropy,
                "mean_token_accuracy": mean_token_accuracy,
            }

        return loss


    # Prediction step

    def prediction_step(
        self,
        model,
        inputs,
        prediction_loss_only,
        ignore_keys=None,
    ):
        """
        During evaluation:
          - Compute DPO loss when chosen/rejected labels are present.
          - Fall back to parent for plain CE otherwise.
          - Use chosen_labels for generation / WER.
        """
        has_preference_batch = (
            "chosen_labels" in inputs and "rejected_labels" in inputs
        )

        if not has_preference_batch:
            return super().prediction_step(
                model, inputs, prediction_loss_only, ignore_keys=ignore_keys
            )

        prepared = self._prepare_inputs(inputs)

        with torch.no_grad():
            loss, _ = self.compute_loss(model, prepared, return_outputs=True)

        loss = loss.detach()

        if prediction_loss_only:
            return loss, None, None

        # Generation / WER against chosen labels
        _, generated_tokens, labels = super().prediction_step(
            model,
            {"input_features": inputs["input_features"], "labels": inputs["chosen_labels"]},
            prediction_loss_only=False,
            ignore_keys=ignore_keys,
        )

        return loss, generated_tokens, labels