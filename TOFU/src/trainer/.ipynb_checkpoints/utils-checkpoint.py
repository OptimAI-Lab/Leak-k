import torch
import random
import numpy as np
from torch import nn
import torch.nn.functional as F
import json
import os


def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def compute_kl_divergence(model, target_model, inputs):
    with torch.no_grad():
        ref_outputs = target_model(**inputs)

    ref_probs = F.log_softmax(ref_outputs.logits, dim=-1)
    ref_probs = ref_probs.view(-1, ref_outputs.logits.shape[-1])

    outputs = model(**inputs)
    current_probs = F.log_softmax(outputs.logits, dim=-1)
    current_probs = current_probs.view(-1, outputs.logits.shape[-1])

    # minimum KL divergence
    return nn.functional.kl_div(
        current_probs, ref_probs, reduction="batchmean", log_target=True
    ), outputs


def compute_batch_nll(model, inputs):
    # get the sum loss for each sequence in a batch
    # NOTE: not same as model(**inputs).loss but has sum loss for each seq in a batch
    outputs = model(**inputs)
    logits = outputs.logits
    labels = inputs["labels"]
    shifted_labels = labels[..., 1:].contiguous()
    logits = logits[..., :-1, :].contiguous()
    loss_function = nn.CrossEntropyLoss(ignore_index=-100, reduction="none")
    loss = loss_function(logits.transpose(-1, -2), shifted_labels).sum(dim=-1)
    return loss, outputs

from torch.distributions import Categorical

def compute_entropy_loss(model, inputs):
    """Compute the entropy of the model's predictions for given inputs, with masking."""
    outputs = model(**inputs)
    logits = outputs.logits  # [batch_size, seq_len, vocab_size]
    labels = inputs["labels"]
    
    # Align logits and labels by excluding the last token from logits and first from labels
    shifted_labels = labels[..., 1:].contiguous()  # (bs, seq_len - 1)
    logits = logits[..., :-1, :].contiguous()  # (bs, seq_len - 1, vocab_size)
    
    # Compute probabilities
    probs = torch.softmax(logits, dim=-1)
    
    # Calculate entropy per token
    entropy = Categorical(probs=probs).entropy()  # [batch_size, seq_len-1]
    
    # Create a mask for valid labels (non -100)
    # This mask will have the same shape as shifted_labels and entropy
    loss_mask = (shifted_labels != -100).float() # Convert to float for multiplication
    
    # Apply the mask to the entropy
    masked_entropy = entropy * loss_mask
    
    # Calculate the sum of masked entropy and the sum of the mask
    # This allows for averaging only over the valid (non-masked) tokens
    sum_masked_entropy = masked_entropy.sum()
    sum_loss_mask = loss_mask.sum()
    
    # Avoid division by zero if all tokens are masked
    if sum_loss_mask == 0:
        return torch.tensor(0.0, device=logits.device)
    
    # Compute the average entropy over the non-masked tokens
    loss = sum_masked_entropy / sum_loss_mask
    
    return loss

def compute_dpo_loss(model, ref_model, win_inputs=None, lose_inputs=None, beta=1.0):
    """
    Compute DPO loss with optional entropy minimization for lose_inputs (forget set).
    """
    if win_inputs is None and lose_inputs is None:
        raise ValueError("Both win_inputs and lose_inputs can't be None")

    win_log_ratio, lose_log_ratio = 0.0, 0.0
    win_outputs, lose_outputs = None, None
    entropy_loss = 0.0

    # Compute win terms (retain set)
    if win_inputs is not None:
        win_loss, win_outputs = compute_batch_nll(model, win_inputs)
        with torch.no_grad():
            win_ref_loss, _ = compute_batch_nll(ref_model, win_inputs)
        win_log_ratio = -(win_loss - win_ref_loss)

    # Compute lose terms (forget set)
    if lose_inputs is not None:
        lose_loss, lose_outputs = compute_batch_nll(model, lose_inputs)
        with torch.no_grad():
            lose_ref_loss, _ = compute_batch_nll(ref_model, lose_inputs)
        lose_log_ratio = -(lose_loss - lose_ref_loss)
        
        # Add entropy minimization for forget set
        entropy_loss = compute_entropy_loss(model, lose_inputs)

    # Original DPO loss
    dpo_loss = -2 / beta * F.logsigmoid(beta * (win_log_ratio - lose_log_ratio)).mean()
    
    # Combined loss: DPO + entropy minimization for forget set
    entropy_weight = 1  # Adjust this weight as needed
    total_loss = dpo_loss + entropy_weight * entropy_loss

    print(f"Total Loss: {total_loss.item()}, DPO Loss: {dpo_loss.item()}, Entropy Loss: {entropy_loss.item()}")

    result = {
        "total_loss": float(total_loss.item()),
        "dpo_loss": float(dpo_loss.item()),
        "entropy_loss": float(entropy_loss.item())
    }
    json_path = "loss_log.json"
    if os.path.exists(json_path):
        with open(json_path, "r") as f:
            data = json.load(f)
    else:
        data = []
    data.append(result)
    with open(json_path, "w") as f:
        json.dump(data, f, indent=2)

    return total_loss, (win_outputs, lose_outputs)

# def compute_dpo_loss(model, ref_model, win_inputs=None, lose_inputs=None, beta=1.0):
#     if win_inputs is None and lose_inputs is None:
#         raise ValueError("Both win_inputs and lose_inputs can't be None")

#     win_log_ratio, lose_log_ratio = 0.0, 0.0
#     win_outputs, lose_outputs = None, None

#     if win_inputs is not None:
#         win_loss, win_outputs = compute_batch_nll(model, win_inputs)
#         with torch.no_grad():
#             win_ref_loss, _ = compute_batch_nll(ref_model, win_inputs)
#         win_log_ratio = -(win_loss - win_ref_loss)

#     if lose_inputs is not None:
#         lose_loss, lose_outputs = compute_batch_nll(model, lose_inputs)
#         with torch.no_grad():
#             lose_ref_loss, _ = compute_batch_nll(ref_model, lose_inputs)
#         lose_log_ratio = -(lose_loss - lose_ref_loss)

#     loss = -2 / beta * F.logsigmoid(beta * (win_log_ratio - lose_log_ratio)).mean()
#     return loss, (win_outputs, lose_outputs)


def compute_undial_loss(model, ref_model, inputs, beta):
    # Forward pass on the student (trainable) model
    outputs = model(**inputs)
    logits = outputs.logits
    labels = inputs["labels"]

    shift_labels = labels[..., 1:].contiguous()
    shift_logits = logits[..., :-1, :].contiguous()

    # Forward pass on the teacher model (no grad)
    with torch.no_grad():
        teacher_logits = ref_model(**inputs).logits
    shift_teacher_logits = teacher_logits[..., :-1, :].contiguous()

    # Build the mask that identifies the tokens need to be unlearned
    mask = torch.zeros_like(shift_teacher_logits)
    batch_idx = torch.arange(mask.shape[0]).view(-1, 1, 1)
    seq_idx = torch.arange(mask.shape[1]).view(1, -1, 1)
    mask[batch_idx, seq_idx, shift_labels.unsqueeze(-1)] = 1.0

    # Adjust teacher logits: subtract di_strength on the correct token
    pre_softmax = shift_teacher_logits - mask * beta
    soft_label = F.softmax(pre_softmax, dim=-1)

    loss_fct = nn.CrossEntropyLoss(reduction="none")
    loss = loss_fct(
        shift_logits.view(-1, shift_logits.size(-1)),
        soft_label.view(-1, soft_label.size(-1)),
    )
    return loss.mean(), outputs


def compute_wga_loss(model, inputs, beta):
    outputs = model(**inputs)
    labels = inputs["labels"]
    labels = labels.to(outputs.logits.device)

    shift_logits = outputs.logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()

    lm_loss = nn.CrossEntropyLoss(ignore_index=-100, reduction="none")(
        shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
    )
    weight_ce = ((-lm_loss).exp().detach()) ** beta
    forget_loss = -(weight_ce * lm_loss)[shift_labels.view(-1) != -100].mean()
    return forget_loss, outputs


def compute_satimp_loss(model, inputs, beta1, beta2):
    outputs = model(**inputs)
    labels = inputs["labels"]
    labels = labels.to(outputs.logits.device)

    shift_logits = outputs.logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()

    lm_loss = nn.CrossEntropyLoss(ignore_index=-100, reduction="none")(
        shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
    )
    weight_sat = ((-lm_loss).exp().detach()) ** beta1
    weight_imp = (1 - (-lm_loss).exp().detach()) ** beta2
    forget_loss = -((weight_sat * weight_imp) * lm_loss)[
        shift_labels.view(-1) != -100
    ].mean()
    return forget_loss, outputs
