import os
import argparse
import mlflow
#import logging
import json
import packaging.version

#logging.getLogger('transformers').setLevel(logging.DEBUG)

import torch
from torch.profiler import profile, record_function, ProfilerActivity, schedule

from datasets import load_from_disk
from sentence_transformers import (
    SentenceTransformer,
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
    SentenceTransformerModelCardData,
)
from sentence_transformers.losses import MultipleNegativesRankingLoss
from sentence_transformers.training_args import BatchSamplers
from sentence_transformers.evaluation import TripletEvaluator

from transformers import TrainerCallback
from transformers.integrations import MLflowCallback
from transformers.utils import flatten_dict, logging, ENV_VARS_TRUE_VALUES

#logging.basicConfig(level=logging.DEBUG) 
logging.set_verbosity_debug()
logger = logging.get_logger(__name__)

        
class SafeGranularProfilerCallback(TrainerCallback):
    def __init__(self, trace_dir="./logdir", start_step=10, end_step=20):
        self.trace_dir = trace_dir
        self.start_step = start_step
        self.end_step = end_step
        self.profiler = None
        self.step_count = 0
        self.enabled = int(os.environ.get("RANK", "0")) == 0  # Only on main rank

    def on_train_begin(self, args, state, control, **kwargs):
        #state.is_world_process_zero
        if self.enabled:
            self.profiler = profile(
                schedule=schedule(wait=5, 
                                  warmup=2, 
                                  active=self.end_step - self.start_step,
                                  repeat=2
                                  ),
                activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                on_trace_ready=torch.profiler.tensorboard_trace_handler(self.trace_dir),
                with_stack=False,
                experimental_config=torch._C._profiler._ExperimentalConfig(verbose=True),
                profile_memory=True,
                record_shapes=True
            )
            self.profiler.__enter__()

    def on_step_end(self, args, state, control, **kwargs):
        if self.enabled:
            if not self.profiler:
                return
            self.step_count += 1
            if self.start_step <= self.step_count < self.end_step:
                self.profiler.step()
            elif self.step_count == self.end_step:
                self.profiler.__exit__(None, None, None)
                self.profiler = None
                self.enabled = False

    def on_train_end(self, args, state, control, **kwargs):
        if self.enabled:
            # Ensure profiler is cleanly closed in case training ends early
            self.profiler.__exit__(None, None, None)
            self.profiler = None


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--per_device_batch_size', type=int, default=16)
    parser.add_argument('--data_dir', type=str, default='/local_disk0/data')
    parser.add_argument('--mlflow_run_name', type=str, default='cli_run')
    parser.add_argument('--output_dir', type=str, default='/local_disk0/tmp')
    
    args = parser.parse_args()
    
    ## When running with torchrun we can get the world_size
    world_size = str(os.environ['WORLD_SIZE'])
    run_id = os.getenv('MLFLOW_RUN_ID', None)

    if run_id is None:
        with mlflow.start_run() as run:
            run_id = run.info.run_id
            os.environ['MLFLOW_RUN_ID'] = run_id

    output_dir = os.path.join(args.output_dir, f"{args.mlflow_run_name}-run-{str(run_id)}")
    os.makedirs(output_dir, exist_ok=True)
    
    
    # 1. Load a model to finetune with 2. (Optional) model card data
    model = SentenceTransformer(
        "microsoft/mpnet-base",
        model_card_data=SentenceTransformerModelCardData(
            language="en",
            license="apache-2.0",
            model_name="MPNet base trained on AllNLI triplets",
        )
    )

    # 3. Load a dataset to finetune on
    dataset = load_from_disk(args.data_dir, keep_in_memory=True)
    train_dataset = dataset["train"].select(range(10_000))
    eval_dataset = dataset["dev"]
    test_dataset = dataset["test"]

    ### TODO Set Drop last to True and add DataLoader in rather than use dataset direct

    # 4. Define a loss function
    loss = MultipleNegativesRankingLoss(model)

    # 5. (Optional) Specify training arguments
    args = SentenceTransformerTrainingArguments(
        # Required parameter:
        output_dir=output_dir,
        #output_dir='/local_disk0/tmp',
        # Optional training parameters:
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.per_device_batch_size,
        per_device_eval_batch_size=args.per_device_batch_size,
        warmup_ratio=0.1,
        fp16=True,  # Set to False if GPU can't handle FP16
        bf16=False,  # Set to True if GPU supports BF16
        batch_sampler=BatchSamplers.NO_DUPLICATES,  # MultipleNegativesRankingLoss benefits from no duplicates
        # Optional tracking/debugging parameters:
        eval_strategy="steps",
        eval_steps=100,
        metric_for_best_model="eval_loss",
        greater_is_better=True,
        load_best_model_at_end = True,
        save_strategy="steps",
        save_steps=100,
        save_total_limit=2,
        logging_steps=100,
        report_to=['tensorboard', 'mlflow']
        #run_name=f"mpnet-base-all-nli-triplet_{world_size}_gpus",  # Used in W&B if `wandb` is installed
    )

    # 6. (Optional) Create an evaluator & evaluate the base model
    dev_evaluator = TripletEvaluator(
        anchors=eval_dataset["anchor"],
        positives=eval_dataset["positive"],
        negatives=eval_dataset["negative"],
        main_distance_function='cosine',
        name="all-nli-dev",
        write_csv=False # this will do a progressive write out
    )
    dev_evaluator(model)


    ##### Folder #####
    chrome_trace_output_path = os.path.join(output_dir, "chrome_trace.json")
    tensorboard_trace = os.path.join(output_dir, f"tb_log")

    # 7. Create a trainer & train
    trainer = SentenceTransformerTrainer(
            model=model,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            loss=loss,
            evaluator=dev_evaluator,
            callbacks=[
                SafeGranularProfilerCallback(trace_dir=output_dir),
            ]
        )
    
    trainer.train()
