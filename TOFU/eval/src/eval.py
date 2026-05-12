import hydra
from omegaconf import DictConfig
from transformers import set_seed  # Added to set global seeds
from trainer.utils import seed_everything
from model import get_model
from evals import get_evaluators
import time


@hydra.main(version_base=None, config_path="../configs", config_name="eval.yaml")
def main(cfg: DictConfig):
    """Entry point of the code to evaluate models
    Args:
        cfg (DictConfig): Config to train
    """
    
    model_cfg = cfg.model
    seed = cfg.get("seed", 42)
    print("*"*50)
    print("seed",seed)
    seed_everything(cfg.seed)
    template_args = model_cfg.template_args
    assert model_cfg is not None, "Invalid model yaml passed in train config."
    model, tokenizer = get_model(model_cfg)
    param_pairs = [
#         (0.0, 0.0),
#         (0.2, 0.2),
#         (0.2, 0.8),
#         (0.8, 0.2),
#         (0.8, 1.0),
        (1.0, 1.0),
#         (0.2,1.0),
#         (0.8,0.8),
#         (1.0,0.8),
#         (1.0,0.2),
    ]
    eval_cfgs = cfg.eval
    print(eval_cfgs)
    evaluators = get_evaluators(eval_cfgs)
    for evaluator_name, evaluator in evaluators.items():
        for temperature, top_p in param_pairs:
            eval_args = {
                "template_args": template_args,
                "top_p": top_p,
                "model": model,
                "tokenizer": tokenizer,
                "temperature": temperature,
                "seed": seed,
            }
            start_time=time.time()
            _ = evaluator.evaluate(**eval_args)
            end_time=time.time()
            print("duration time",end_time-start_time)


if __name__ == "__main__":
    main()
