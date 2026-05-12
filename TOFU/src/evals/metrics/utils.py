from typing import List
from tqdm import tqdm
from rouge_score import rouge_scorer
from collections import defaultdict
from omegaconf import OmegaConf
import numpy as np
import scipy as sc
from torch import nn
import torch
from transformers import StoppingCriteria, StoppingCriteriaList, PreTrainedTokenizer
from data.utils import IGNORE_INDEX
import warnings
import os
import json
import torch.nn.functional as F


def dict_transpose(evals):
    """Transpose a nested dictionary structure to group statistics by item indices."""
    # evals looks like {iidx0: {idx453: {prob: 0.1, loss: 1}},
    #                   iidx1: {idx453: {prob: 0.2, loss: 2}}}
    # multiple answers indexed by intra_item_idx, then item_idx
    # invert the dict, put outermost iidx deepest inside
    # after dict transpose looks like {idx453: {prob: [0.1, 0.2], loss: [1, 2]}
    all_iidxs = list(evals.keys())
    all_idxs = list(evals[all_iidxs[0]].keys())
    all_stat_names = list(evals[all_iidxs[0]][all_idxs[0]].keys())
    evals = {
        idx: {
            stat: [evals[iidx][idx][stat] for iidx in all_iidxs]
            for stat in all_stat_names
        }
        for idx in all_idxs
    }
    return evals


def aggregate_to_1D(x):
    return np.mean(x, axis=tuple(range(1, x.ndim)))


def get_forget_quality(model_tr, reference_tr):
    test_res = sc.stats.ks_2samp(1 / (model_tr + 1e-10), 1 / (reference_tr + 1e-10))
    return {"agg_value": test_res.pvalue}


def run_batchwise_evals(model, dataloader, batch_eval_fn, batch_eval_fn_args, eval_msg):
    """Run batch-wise evaluations on a dataset using a specified evaluation function. Handles
    multi-answer datasets by organizing evaluations by answer indices and aggregating results."""
    evals = defaultdict(dict)
    for batch in tqdm(dataloader, desc=eval_msg, total=len(dataloader)):
        # if data arrives in normal format we convert the batch to multiple answer-style
        # like in tofu_perturbed by adding a fake intra_item_index
        if "input_ids" in batch:
            batch = {"0": batch}
        # Assume batch like {"0": {"input_ids": [[]]..., "index": [453, 454..]},
        #                    "1": {"input_ids": [[]]..., "index": [453, 454..]}..}
        assert isinstance(next(iter(batch.values())), dict) and "input_ids" in next(
            iter(batch.values())
        )
        for intra_item_idx, mini_batch in batch.items():
            data_indices = (
                mini_batch.pop("index").cpu().numpy().tolist()
            )  # data item indices
            batch_evals = batch_eval_fn(
                model=model, batch=mini_batch, **batch_eval_fn_args
            )
            indexwise_batch_evals = dict(zip(data_indices, batch_evals))
            assert not (
                evals[intra_item_idx].keys() & indexwise_batch_evals.keys()
            ), "Data indices repeated while iterating dataloader"
            evals[intra_item_idx] |= indexwise_batch_evals
    # evals looks like {iidx0: {idx453: {prob: 0.1, loss: 1}},
    #                   iidx1: {idx453: {prob: 0.2, loss: 2}}}
    if len(evals) == 1:  # normal single answer dataset, no need for list
        evals = next(iter(evals.values()))
    else:
        # for each index return a dict with all intra_item_idx values in list
        # after dict transpose looks like {idx453: {prob: [0.1, 0.2], loss: [1, 2]}}
        evals = dict_transpose(evals)
    print("Evaluated", len(evals), "examples")
    return evals


def evaluate_probability(model, batch, temperature=1.0, top_p=None):
    """Evaluate model probabilities and average token-level loss for a given batch.
    
    Args:
        model: The model to evaluate.
        batch: Input batch data.
        temperature: Temperature parameter for softmax (default=1.0).
        top_p: Top-p (nucleus) sampling threshold (default=None).
    """
    batch = {k: v.to(model.device) for k, v in batch.items()}
    with torch.no_grad():
        output = model(**batch)
    logits = output.logits
    labels = batch["labels"]
    # print("Evaluating probabilities with temperature:", temperature)
    # print("top_p:", top_p)
    # Apply temperature scaling
    if temperature != 1.0:
        logits = logits / temperature
    
    # Apply top-p sampling if specified
    if top_p is not None and top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
        
        # Remove tokens with cumulative probability above top_p
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift the indices to the right to keep the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        
        # Create a mask to zero out logits that should be removed
        indices_to_remove = sorted_indices_to_remove.scatter(-1, sorted_indices, sorted_indices_to_remove)
        logits = logits.masked_fill(indices_to_remove, float('-inf'))
    
    shifted_labels = labels[..., 1:].contiguous()
    logits = logits[..., :-1, :].contiguous()
    loss_function = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX, reduction="none")
    
    # Calculate losses
    losses = loss_function(logits.transpose(-1, -2), shifted_labels).sum(dim=-1)
    num_token_gt = (batch["labels"] != IGNORE_INDEX).sum(-1)
    avg_losses = losses / num_token_gt
    normalized_probs = torch.exp(-avg_losses)

    avg_losses = avg_losses.cpu().numpy().tolist()
    normalized_probs = normalized_probs.cpu().numpy().tolist()
    return [
        {"prob": prob, "avg_loss": avg_loss}
        for prob, avg_loss in zip(normalized_probs, avg_losses)
    ]


def tokenwise_logprobs(model, batch, grad=False, return_labels=False):
    """
    Compute token-wise next token prediction logprobs for all labeled tokens for each sample in a batch.
    `grad` decides whether gradients are turned on
    Returns
    log_probs_batch (List[Tensor]): Tensors of size seq_len where seq_len is length of labeled tokens
    labels_batch (List[Tensor]): List of tensors of length N. Returned only if return_labels is True
    """
    batch = {k: v.to(model.device) for k, v in batch.items()}
    with torch.set_grad_enabled(grad):
        output = model(**batch)

    logits = output.logits
    bsz, seq_len, V = logits.shape
    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)[:, :-1, :]
    # ^ we don't predict next token for last token, bsz x seq_len-1 x V
    next_tokens = batch["input_ids"][:, 1:].unsqueeze(-1)  # bsz x seq_len-1 x 1
    target_log_probs = torch.gather(log_probs, dim=2, index=next_tokens).squeeze(-1)
    log_probs_batch = []
    labels_batch = []
    for i in range(bsz):
        labels = batch["labels"][i]
        # only focus on tokens which have loss on them (i.e. used in labels)
        actual_indices = (labels != IGNORE_INDEX).nonzero(as_tuple=True)[0][
            :-1
        ]  # -1 to ignore eos prediction
        num_actual_tokens = actual_indices.numel()
        if num_actual_tokens == 0:
            labels_batch.append(torch.tensor([], device=labels.device))
            log_probs_batch.append(torch.tensor([], device=labels.device))
            continue
        start_idx, end_idx = actual_indices[0].item(), actual_indices[-1].item()
        if start_idx == 0:
            warnings.warn(
                "Index 0 in a datapoint's input_ids must not have loss (unignored labels) on it",
                UserWarning,
            )
        log_probs_batch.append(target_log_probs[i, start_idx - 1 : end_idx])
        labels_batch.append(labels[actual_indices])

    return (log_probs_batch, labels_batch) if return_labels else log_probs_batch


def tokenwise_vocab_logprobs(model, batch, grad=False, return_labels=False):
    """Get vocabulary-wise log probabilities for each token in the sequence.

    Returns:
        log_probs_batch (List[Tensor]): List of tensors of shape (N, V) containing log probabilities
        for each sequence, where N is the length of labeled tokens and V is vocab size.
        labels_batch (List[Tensor]): List of tensors of length N. Returned only if return_labels is True
    """
    batch = {k: v.to(model.device) for k, v in batch.items()}
    with torch.set_grad_enabled(grad):
        output = model(**batch)

    logits = output.logits
    bsz, seq_len, V = logits.shape
    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)[
        :, :-1, :
    ]  # Don't predict for last token

    # Process each sequence in batch separately
    log_probs_batch = []
    labels_batch = []
    for i in range(bsz):
        labels = batch["labels"][i]
        # Only include positions that have labels
        actual_indices = (labels != IGNORE_INDEX).nonzero(as_tuple=True)[0][
            :-1
        ]  # -1 to ignore eos prediction
        if len(actual_indices) == 0:
            labels_batch.append(torch.tensor([], device=labels.device))
            log_probs_batch.append(torch.zeros(0, V, device=labels.device))
            continue
        start_idx, end_idx = actual_indices[0].item(), actual_indices[-1].item()
        if start_idx == 0:
            warnings.warn(
                "Index 0 in a datapoint's input_ids must not have loss (unignored labels) on it",
                UserWarning,
            )
        # Return full distribution for each position: shape (N, V)
        log_probs_batch.append(log_probs[i, start_idx - 1 : end_idx])
        labels_batch.append(labels[actual_indices])

    return (log_probs_batch, labels_batch) if return_labels else log_probs_batch


class MultiTokenEOSCriteria(StoppingCriteria):
    """Criteria to stop on the specified multi-token sequence. Stopping Criteria forked
    and modified from [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness/blob/27924d77953491f66a038a09892807065e469358/lm_eval/models/utils.py#L208)"""

    def __init__(
        self,
        sequence: str,
        tokenizer: PreTrainedTokenizer,
        initial_decoder_input_length: int,
        batch_size: int,
    ) -> None:
        self.initial_decoder_input_length = initial_decoder_input_length
        self.done_tracker = [False] * batch_size
        self.sequence = sequence
        self.sequence_ids = tokenizer.encode(sequence, add_special_tokens=False)
        # we look back for 2 more tokens than it takes to encode our stop sequence
        # because tokenizers suck, and a model might generate `['\n', '\n']` but our `sequence` is `['\n\n']`
        # and we don't want to mistakenly not stop a generation because our
        # (string) stop sequence was output in a different tokenization

        # NOTE: there is a minor danger that this will end up looking back 2 tokens into the past, into the inputs to the model,
        # and stopping generation immediately as a result. With only 2 extra tokens of lookback, this risk is minimized
        # Additionally, in lookback_ids_batch we should prevent ever looking back into the inputs as described.
        self.sequence_id_len = len(self.sequence_ids) + 2
        self.tokenizer = tokenizer

    def __call__(self, input_ids, scores, **kwargs) -> bool:
        # For efficiency, we compare the last n tokens where n is the number of tokens in the stop_sequence
        lookback_ids_batch = input_ids[:, self.initial_decoder_input_length :]

        lookback_ids_batch = lookback_ids_batch[:, -self.sequence_id_len :]

        lookback_tokens_batch = self.tokenizer.batch_decode(lookback_ids_batch)

        for i, done in enumerate(self.done_tracker):
            if not done:
                self.done_tracker[i] = self.sequence in lookback_tokens_batch[i]
        return False not in self.done_tracker


def stop_sequences_criteria(
    tokenizer: PreTrainedTokenizer,
    stop_sequences: List[str],
    initial_decoder_input_length: int,
    batch_size: int,
) -> StoppingCriteriaList:
    return StoppingCriteriaList(
        [
            *[
                MultiTokenEOSCriteria(
                    sequence, tokenizer, initial_decoder_input_length, batch_size
                )
                for sequence in stop_sequences
            ],
        ]
    )


def pass_k(model, tokenizer, dataloader, generation_args):
    # model_name = getattr(model, "name_or_path", "model")
    output_dir = generation_args.get("output_dir")
    result_dir = os.path.join(output_dir, generation_args.get("set"), "temperature={}".format(generation_args.get("temperature"))+"top_p={}".format(generation_args.get("top_p")))
    print(f"Saving results to {result_dir}")
    os.makedirs(result_dir, exist_ok=True)
    print("generation_args",generation_args)
    # n_list = [1, 2, 4, 8, 16, 32, 64, 128]
    n_list = [200]
    for n in n_list:
        with open(os.path.join(result_dir, f"generations_n{n}.json"), "w") as f:
            pass  

    for batch in tqdm(dataloader, desc="Evaluating batches", total=len(dataloader)):
        pass_k_per_query(model, tokenizer, batch, generation_args, result_dir)

    summary = {}
    for n in n_list:
        file_path = os.path.join(result_dir, f"generations_n{n}.json")
        scores = []
        with open(file_path, "r") as f:
            for line in f:
                data = json.loads(line)
                scores.append(data["best_answer"]["rougeL_recall"])
        mean_score = sum(scores) / len(scores) if scores else 0
        summary[f"generations_n{n}"] = mean_score

    with open(os.path.join(result_dir, "rougeL_summary.json"), "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    return summary

def generate_with_adaptive_temperature(
    model, 
    tokenizer, 
    prompt_ids, 
    generation_args,
    cT=0.9 # Confidence threshold for adaptive temperature
):
    """
    Generates a single text sequence token-by-token using adaptive temperature scaling.
    """
    # Get parameters from generation_args
    max_new_tokens = generation_args.get("max_new_tokens", 256)
    normal_temp = generation_args.get("temperature", 0.1)
    top_p = generation_args.get("top_p", 0.9)
    eos_token_id = tokenizer.eos_token_id

    generated_ids = prompt_ids
    confidence_history = []
    
    with torch.no_grad():
        for _ in range(max_new_tokens):
            # 1. Get model's probability distribution (logits) for the next token
            outputs = model(input_ids=generated_ids)
            next_token_logits = outputs.logits[:, -1, :]

            # 2. Calculate the current confidence
            #    First, get the probability of the most likely token
            next_token_probs = F.softmax(next_token_logits, dim=-1)
            most_likely_token_prob = torch.max(next_token_probs).item()
            confidence_history.append(most_likely_token_prob)
            current_confidence = sum(confidence_history) / len(confidence_history)

            # 3. Adaptively adjust the temperature
            if current_confidence > cT:
                temperature = 0.0  # Force greedy decoding
            else:
                temperature = normal_temp

            # 4. Sample the next token based on the temperature
            if temperature == 0.0:
                # If temperature is 0, choose the token with the highest probability
                next_token_id = torch.argmax(next_token_logits, dim=-1).unsqueeze(0)
            else:
                # Apply top-p and temperature for sampling
                # a. Apply temperature scaling
                scaled_logits = next_token_logits / temperature
                
                # b. Apply Top-p (Nucleus Sampling)
                sorted_logits, sorted_indices = torch.sort(scaled_logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                
                indices_to_remove = sorted_indices[sorted_indices_to_remove]
                scaled_logits[:, indices_to_remove] = -float('Inf')
                
                # c. Sample from the modified distribution
                probs = F.softmax(scaled_logits, dim=-1)
                next_token_id = torch.multinomial(probs, num_samples=1)

            # 5. Append the newly generated token to the sequence
            generated_ids = torch.cat([generated_ids, next_token_id], dim=-1)

            # Check if the end-of-sequence token was generated
            if next_token_id.item() == eos_token_id:
                break
                
    return generated_ids

def pass_k_per_query(model, tokenizer, batch, generation_args, result_dir):
#     seed = generation_args.get("seed", 42)
#     torch.manual_seed(seed)
#     if torch.cuda.is_available():
#         torch.cuda.manual_seed_all(seed)
#     print("*"*50)
#     print(" I have seed ")
#     print("seed:",seed)
    batch = {k: v.to(model.device) for k, v in batch.items()}
    input_ids = batch["input_ids"]
    labels = batch["labels"]
    input_texts = tokenizer.batch_decode(
        input_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
    )
    tokens = [label[label != IGNORE_INDEX] for label in labels]
    full_texts = tokenizer.batch_decode(
        tokens, skip_special_tokens=True, clean_up_tokenization_spaces=True
    )
    ground_truths = [
        full_text.replace(input_text, "").strip()
        for input_text, full_text in zip(input_texts, full_texts)
    ]
    prompt = input_texts[0]
    ground_truth = ground_truths[0]

    def contains_exact_phrase(response, answer):
        response_clean = response.lower()
        reference_clean = answer.lower()
        return reference_clean in response_clean

    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    adaptive_tmp = False
#     for n in [1, 2, 4, 8, 16, 32, 64, 128]:
    for n in [200]:
        if adaptive_tmp:
            # Adaptive temperature logic unchanged
            prompt_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)
            output_ids_list = []
            for _ in range(n):
                output_ids = generate_with_adaptive_temperature(
                    model=model,
                    tokenizer=tokenizer,
                    prompt_ids=prompt_ids,
                    generation_args=generation_args,
                    cT=generation_args.get("cT", 0.9)
                )
                output_ids_list.append(output_ids[0])
            decoded_outputs = tokenizer.batch_decode(
                output_ids_list, skip_special_tokens=True, clean_up_tokenization_spaces=True
            )
            del prompt_ids, output_ids_list
        else:
            decoded_outputs = []
            # Use batch_size = 32 for n >= 64
            if n >= 64:
                batch_size = 32
                for start in range(0, n, batch_size):
                    current_batch = min(batch_size, n - start)
                    prompts = [prompt] * current_batch
                    inputs = tokenizer(prompts, return_tensors="pt", padding=True, add_special_tokens=True).to(model.device)
                    with torch.no_grad():
                        if generation_args.get("temperature")==0:
                            output_ids = model.generate(
                                input_ids=inputs["input_ids"],
                                attention_mask=inputs["attention_mask"],
                                max_new_tokens=generation_args.get("max_new_tokens", 256),
                                do_sample=False,  # Greedy decoding
                                num_return_sequences=1,  # FIXED: 1 per prompt
                                pad_token_id=tokenizer.eos_token_id,
                            )
                        else:
                            output_ids = model.generate(
                                input_ids=inputs["input_ids"],
                                attention_mask=inputs["attention_mask"],
                                max_new_tokens=generation_args.get("max_new_tokens", 256),
                                do_sample=True,
                                num_return_sequences=1,  # FIXED: 1 per prompt
                                pad_token_id=tokenizer.eos_token_id,
                                top_p=generation_args.get("top_p", 0.9),
                                temperature=generation_args.get("temperature", 0.1)
                            )
                    decoded_outputs.extend(tokenizer.batch_decode(
                        output_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
                    ))
                    # Cleanup after each mini-batch
                    del inputs, output_ids
                    torch.cuda.empty_cache()
            else:
                # For n < 64, process in a single batch
                prompts = [prompt] * n
                inputs = tokenizer(prompts, return_tensors="pt", padding=True, add_special_tokens=True).to(model.device)
                with torch.no_grad():
                    if generation_args.get("temperature")==0:
                        output_ids = model.generate(
                            input_ids=inputs["input_ids"],
                            attention_mask=inputs["attention_mask"],
                            max_new_tokens=generation_args.get("max_new_tokens", 256),
                            do_sample=False,  # Greedy decoding
                            num_return_sequences=1,  # FIXED: 1 per prompt
                            pad_token_id=tokenizer.eos_token_id,
                        )
                    else:
                        output_ids = model.generate(
                            input_ids=inputs["input_ids"],
                            attention_mask=inputs["attention_mask"],
                            max_new_tokens=generation_args.get("max_new_tokens", 256),
                            do_sample=True,
                            num_return_sequences=1,  # FIXED: 1 per prompt
                            pad_token_id=tokenizer.eos_token_id,
                            top_p=generation_args.get("top_p", 0.9),
                            temperature=generation_args.get("temperature", 0.1)
                        )
                decoded_outputs = tokenizer.batch_decode(
                    output_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
                )
                del inputs, output_ids
                torch.cuda.empty_cache()

        # --- Evaluation ---
        responses = []
        best_rouge = -1
        best_idx = -1

        for idx, gen_text in enumerate(decoded_outputs):
            response_clean = gen_text.split(prompt)[-1].strip() if prompt in gen_text else gen_text.strip()
            rouge_recall = scorer.score(ground_truth, response_clean)['rougeL'].recall
            exact_score = 1 if contains_exact_phrase(response_clean, ground_truth) else 0
            responses.append({
                "response": response_clean,
                "rougeL_recall": rouge_recall,
                "exact_match": exact_score
            })
            if generation_args.get("set") == "forget":
                if rouge_recall > best_rouge:
                    best_rouge = rouge_recall
                    best_idx = idx
            else:
                if rouge_recall < best_rouge:
                    best_rouge = rouge_recall
                    best_idx = idx

        if best_idx == -1 and responses:
            best_idx = 0

        best_answer = {
            "response": responses[best_idx]["response"],
            "rougeL_recall": responses[best_idx]["rougeL_recall"],
            "exact_match": responses[best_idx]["exact_match"]
        } if best_idx != -1 else {"response": "", "rougeL_recall": 0, "exact_match": 0}

        result = {
            "prompt": prompt,
            "ground_truth": ground_truth,
            "responses": responses,
            "best_answer": best_answer
        }

        with open(os.path.join(result_dir, f"generations_n{n}.json"), "a") as f:
            json.dump(result, f, ensure_ascii=False)
            f.write("\n")

        # Final cleanup for each `n` loop
        del decoded_outputs, responses
        torch.cuda.empty_cache()


import torch
import torch.nn.functional as F


# def generate_with_adaptive_temperature_accelerated(
#     model, 
#     tokenizer, 
#     prompt_ids, 
#     generation_args,
#     cT=0.9 # Confidence threshold for adaptive temperature
# ):
#     """
#     Generates text sequences token-by-token using adaptive temperature scaling,
#     supporting batch generation.
    
#     prompt_ids: torch.Tensor of shape (batch_size, sequence_length)
#     """
#     # Get parameters from generation_args
#     max_new_tokens = generation_args.get("max_new_tokens", 256)
#     normal_temp = generation_args.get("temperature", 0.1)
#     top_p = generation_args.get("top_p", 0.9)
#     eos_token_id = tokenizer.eos_token_id

#     batch_size = prompt_ids.shape[0]
#     generated_ids = prompt_ids.clone()
    
#     confidence_history_list = [[0.0] for _ in range(batch_size)]

#     active_sequences = torch.ones(batch_size, dtype=torch.bool, device=model.device)
#     num_active_sequences = batch_size

#     with torch.no_grad():
#         for _ in range(max_new_tokens):
#             if num_active_sequences == 0:
#                 break

#             current_input_ids = generated_ids[active_sequences]
#             if current_input_ids.shape[0] == 0:
#                 break

#             outputs = model(input_ids=current_input_ids)
#             next_token_logits = outputs.logits[:, -1, :]

#             next_token_probs = F.softmax(next_token_logits, dim=-1)
#             most_likely_token_prob_active = torch.max(next_token_probs, dim=-1).values

#             current_confidence_active = torch.zeros_like(most_likely_token_prob_active)
#             active_indices = torch.where(active_sequences)[0]

#             for i, original_idx in enumerate(active_indices):
#                 confidence_history_list[original_idx].append(most_likely_token_prob_active[i].item())
#                 current_confidence_active[i] = sum(confidence_history_list[original_idx]) / len(confidence_history_list[original_idx])

#             temperature_active = torch.where(current_confidence_active > cT, 
#                                              torch.tensor(0.0, device=model.device), 
#                                              torch.tensor(normal_temp, device=model.device))

#             next_token_id_active = torch.zeros(num_active_sequences, dtype=torch.long, device=model.device)

#             greedy_mask = (temperature_active == 0.0)
#             sampled_mask = ~greedy_mask

#             if torch.any(greedy_mask):
#                 next_token_id_active[greedy_mask] = torch.argmax(next_token_logits[greedy_mask], dim=-1)

#             if torch.any(sampled_mask):
#                 # Apply temperature scaling for sampled sequences
#                 scaled_logits_sampled = next_token_logits[sampled_mask] / temperature_active[sampled_mask].unsqueeze(-1)
                
#                 # Ensure temperature is not effectively zero for sampled sequences
#                 # A very small non-zero temperature can cause inf/nan if input logits are also large.
#                 # If temperature_active[sampled_mask] contains any zeros, this division will produce inf.
#                 # This should ideally be caught earlier by the greedy_mask, but double-checking here.
#                 if (temperature_active[sampled_mask] == 0).any():
#                     # This implies a logic error if sampled_mask is true but temperature is 0.
#                     # As a safety, you could force a small non-zero value or log a warning.
#                     # For now, let's assume this branch is only for non-zero temperatures.
#                     pass # The `greedy_mask` should prevent this.

#                 # --- Optimized Top-p Implementation ---
#                 sorted_logits, sorted_indices = torch.sort(scaled_logits_sampled, descending=True)
                
#                 # Apply softmax to sorted logits to get sorted probabilities
#                 sorted_probs = F.softmax(sorted_logits, dim=-1)
                
#                 # Calculate cumulative probabilities
#                 cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
                
#                 # Create a mask for tokens to keep: cumulative_probs <= top_p
#                 # Also, ensure at least one token is kept (the highest prob token)
#                 # The first token is always kept by default in this setup because cumulative_probs starts >= 0.
#                 indices_to_remove = cumulative_probs > top_p
                
#                 # Shift the mask to the right to retain the first token that crosses the top_p threshold
#                 indices_to_remove[..., 1:] = indices_to_remove[..., :-1].clone()
#                 indices_to_remove[..., 0] = False # Ensure the highest probability token is always kept
                
#                 # Create a mask to set logits to -inf
#                 # This scatter operation creates a mask in the original unsorted order
#                 logits_to_remove_mask = torch.zeros_like(scaled_logits_sampled, dtype=torch.bool)
#                 logits_to_remove_mask.scatter_(dim=1, index=sorted_indices, src=indices_to_remove)

#                 # Set logits to -inf for tokens to remove
#                 scaled_logits_sampled[logits_to_remove_mask] = -float('Inf')

#                 # After masking, re-normalize probabilities
#                 probs_sampled = F.softmax(scaled_logits_sampled, dim=-1)

#                 # Robustness check for all-zero probabilities after softmax (can happen if all logits are -inf)
#                 # This line was causing the IndexError.
#                 # The issue was that probs_sampled.sum(dim=-1, keepdim=True) == 0 creates a mask of shape (N, 1)
#                 # and you were trying to use it to index a tensor of shape (N, V) for element-wise assignment.
#                 # Instead, we want to set *entire rows* to uniform.
                
#                 # Identify rows where the sum of probabilities is effectively zero (or very close)
#                 # Using a small epsilon instead of exactly 0 can be more robust against floating point inaccuracies
#                 zero_sum_mask = (probs_sampled.sum(dim=-1) < 1e-6) # Check if sum is very close to zero
                
#                 if torch.any(zero_sum_mask):
#                     num_vocab = probs_sampled.shape[-1]
#                     uniform_prob_value = 1.0 / num_vocab
                    
#                     # Create a tensor of uniform probabilities for the problematic rows
#                     uniform_dist_for_zero_rows = torch.full_like(probs_sampled[zero_sum_mask], uniform_prob_value)
                    
#                     # Assign this uniform distribution to the problematic rows
#                     probs_sampled[zero_sum_mask] = uniform_dist_for_zero_rows
                
#                 # Now probs_sampled should be valid for torch.multinomial
#                 next_token_id_active[sampled_mask] = torch.multinomial(probs_sampled, num_samples=1).squeeze(1)

#             # ... (rest of the function, which was largely fine)
#             current_max_len = generated_ids.shape[1]
#             new_max_len = current_max_len + 1

#             padded_generated_ids = torch.full((batch_size, new_max_len), 
#                                                 tokenizer.pad_token_id if tokenizer.pad_token_id is not None else eos_token_id, 
#                                                 dtype=torch.long, device=model.device)
#             padded_generated_ids[:, :current_max_len] = generated_ids

#             # Ensure next_token_id_active has correct shape (num_active_sequences,) before indexing
#             # It already should be from the .squeeze(1) on multinomial output
#             padded_generated_ids[active_sequences, new_max_len - 1] = next_token_id_active
#             generated_ids = padded_generated_ids

#             eos_generated = (next_token_id_active == eos_token_id)
            
#             # Update active_sequences mask
#             # Ensure active_indices is consistent.
#             # active_sequences[active_indices[eos_generated]] = False
#             # Better:
#             active_sequences[active_indices[eos_generated]] = False
            
#             num_active_sequences = active_sequences.sum().item()

#     return generated_ids

# def pass_k_per_query_accelerated(model, tokenizer, batch, generation_args, result_dir):
#     batch = {k: v.to(model.device) for k, v in batch.items()}
#     input_ids = batch["input_ids"]
#     labels = batch["labels"]
#     input_texts = tokenizer.batch_decode(
#         input_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
#     )
#     tokens = [label[label != IGNORE_INDEX] for label in labels]
#     full_texts = tokenizer.batch_decode(
#         tokens, skip_special_tokens=True, clean_up_tokenization_spaces=True
#     )
#     ground_truths = [
#         full_text.replace(input_text, "").strip()
#         for input_text, full_text in zip(input_texts, full_texts)
#     ]
#     prompt = input_texts[0]
#     ground_truth = ground_truths[0]

#     def contains_exact_phrase(response, answer):
#         response_clean = response.lower()
#         reference_clean = answer.lower()
#         return reference_clean in response_clean

#     scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
#     adaptive_tmp = True # This needs to be controlled by your overall logic

#     for n in [1, 2, 4, 8, 16, 32, 64, 128]:
#         if adaptive_tmp:
#             # Prepare prompts for batch generation
#             # Repeat the prompt 'n' times for batching
#             prompts_for_batch = [prompt] * n
#             prompt_ids_batch = tokenizer(prompts_for_batch, return_tensors="pt", 
#                                          padding=True, add_special_tokens=True).input_ids.to(model.device)
            
#             # Call the accelerated adaptive temperature generation function
#             output_ids_batch = generate_with_adaptive_temperature_accelerated(
#                 model=model,
#                 tokenizer=tokenizer,
#                 prompt_ids=prompt_ids_batch,
#                 generation_args=generation_args,
#                 cT=generation_args.get("cT", 0.9)
#             )
            
#             # Decode the entire batch of generated sequences
#             decoded_outputs = tokenizer.batch_decode(
#                 output_ids_batch, skip_special_tokens=True, clean_up_tokenization_spaces=True
#             )
#             del prompt_ids_batch, output_ids_batch
#             torch.cuda.empty_cache() # Clear cache after large operations
#         else:
#             decoded_outputs = []
#             # Use batch_size = 32 for n >= 64, or any suitable batch size
#             batch_size = 32 if n >= 64 else n # Use n as batch size if smaller than 32
            
#             for start in range(0, n, batch_size):
#                 current_batch_size = min(batch_size, n - start)
#                 prompts_chunk = [prompt] * current_batch_size
#                 inputs = tokenizer(prompts_chunk, return_tensors="pt", padding=True, add_special_tokens=True).to(model.device)
#                 with torch.no_grad():
#                     output_ids = model.generate(
#                         input_ids=inputs["input_ids"],
#                         attention_mask=inputs["attention_mask"],
#                         max_new_tokens=generation_args.get("max_new_tokens", 256),
#                         do_sample=True,
#                         num_return_sequences=1,
#                         pad_token_id=tokenizer.eos_token_id,
#                         top_p=generation_args.get("top_p", 0.9),
#                         temperature=generation_args.get("temperature", 0.1)
#                     )
#                 decoded_outputs.extend(tokenizer.batch_decode(
#                     output_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
#                 ))
#                 del inputs, output_ids
#                 torch.cuda.empty_cache()

#         # --- Evaluation --- (Remains largely the same)
#         responses = []
#         best_rouge = -1
#         best_idx = -1

#         for idx, gen_text in enumerate(decoded_outputs):
#             response_clean = gen_text.split(prompt)[-1].strip() if prompt in gen_text else gen_text.strip()
#             rouge_recall = scorer.score(ground_truth, response_clean)['rougeL'].recall
#             exact_score = 1 if contains_exact_phrase(response_clean, ground_truth) else 0
#             responses.append({
#                 "response": response_clean,
#                 "rougeL_recall": rouge_recall,
#                 "exact_match": exact_score
#             })
#             if generation_args.get("set") == "forget":
#                 if rouge_recall > best_rouge:
#                     best_rouge = rouge_recall
#                     best_idx = idx
#             else:
#                 if best_rouge == -1 or rouge_recall < best_rouge: # Initialize best_rouge correctly for 'remember'
#                     best_rouge = rouge_recall
#                     best_idx = idx

#         if best_idx == -1 and responses:
#             best_idx = 0

#         best_answer = {
#             "response": responses[best_idx]["response"],
#             "rougeL_recall": responses[best_idx]["rougeL_recall"],
#             "exact_match": responses[best_idx]["exact_match"]
#         } if best_idx != -1 else {"response": "", "rougeL_recall": 0, "exact_match": 0}

#         result = {
#             "prompt": prompt,
#             "ground_truth": ground_truth,
#             "responses": responses,
#             "best_answer": best_answer
#         }

#         os.makedirs(result_dir, exist_ok=True) # Ensure directory exists
#         with open(os.path.join(result_dir, f"generations_n{n}.json"), "a") as f:
#             json.dump(result, f, ensure_ascii=False)
#             f.write("\n")

#         # Final cleanup for each `n` loop
#         del decoded_outputs, responses
#         torch.cuda.empty_cache()


def eval_text_similarity(model, tokenizer, batch, generation_args):
    """Evaluate text similarity between model-generated outputs and ground truth using ROUGE scores."""

    def eval_rouge_recall_batch(gen_outputs, ground_truths):
        scorer = rouge_scorer.RougeScorer(["rouge1", "rougeL"], use_stemmer=True)
        evals = []
        for gen, gt in zip(gen_outputs, ground_truths):
            rouge_scores = scorer.score(gt, gen)
            evals.append(
                {
                    "rouge1_recall": rouge_scores["rouge1"].recall,
                    "rougeL_f1": rouge_scores["rougeL"].fmeasure,
                    "rougeL_recall": rouge_scores["rougeL"].recall,
                }
            )
        return evals
    # print("Evaluating text similarity with generation args:", generation_args)
    batch = {k: v.to(model.device) for k, v in batch.items()}
    input_ids = batch["input_ids"]
    labels = batch["labels"]
    input_texts = tokenizer.batch_decode(
        input_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
    )
    tokens = [label[label != IGNORE_INDEX] for label in labels]
    full_texts = tokenizer.batch_decode(
        tokens, skip_special_tokens=True, clean_up_tokenization_spaces=True
    )
    ground_truths = [
        full_text.replace(input_text, "").strip()
        for input_text, full_text in zip(input_texts, full_texts)
    ]

    attention_mask = batch["attention_mask"]

    # convert to a simple dict from DictConfig
    generation_args = OmegaConf.to_container(generation_args, resolve=True)
    stopwords = generation_args.pop("stopwords", None)
    if stopwords is not None:
        assert isinstance(stopwords, list)
        sc = stop_sequences_criteria(
            tokenizer, stopwords, input_ids.shape[1], input_ids.shape[0]
        )
        generation_args["stopping_criteria"] = sc
        
    # print("Generation args:", generation_args)
    output = model.generate(
        input_ids,
        attention_mask=attention_mask,
        **generation_args,
        pad_token_id=tokenizer.eos_token_id,
    )
    gen_texts = tokenizer.batch_decode(
        output[:, input_ids.shape[-1] :],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True,
    )

    # cut off at stopwords
    if stopwords is None:
        stopwords = []
    stopwords = [tokenizer.decode([tokenizer.eos_token_id])] + stopwords
    for i in range(len(gen_texts)):
        raw_text = gen_texts[i]
        for word in stopwords:
            if word and word in raw_text:
                raw_text = raw_text.split(word)[0]
        raw_text = raw_text.strip()
        gen_texts[i] = raw_text

    scores = eval_rouge_recall_batch(gen_texts, ground_truths)
    scores = [
        {
            **rouge_evals,
            "input": input_text,
            "ground_truth": ground_truth,
            "generation": gen_text,
        }
        for rouge_evals, input_text, ground_truth, gen_text in zip(
            scores, input_texts, ground_truths, gen_texts
        )
    ]
    return scores


def extract_target_texts_from_processed_data(tokenizer, batch):
    """Extract and detokenize text from activated positions in the batch."""
    labels = batch["labels"]
    labels = [elem[elem != -100] for elem in labels]
    texts = [
        tokenizer.decode(elem.tolist(), skip_special_tokens=True) for elem in labels
    ]
    return texts

