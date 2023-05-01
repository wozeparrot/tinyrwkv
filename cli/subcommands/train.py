from tinygrad.nn.optim import Optimizer, AdamW, get_parameters, get_state_dict
from tinygrad.tensor import Tensor
from tokenizers import Tokenizer
from tqdm import trange
import numpy as np

from argparse import Namespace, _SubParsersAction, ArgumentParser
from typing import cast
import gc
import math
import os
import pickle

from tinyrwkv import RWKV_GPT
from tinyrwkv.utils.model import count_parameters


def generate_parser(subparsers: "_SubParsersAction[ArgumentParser]") -> None:
    parser = subparsers.add_parser(
        "tra",
        help="pretrain or finetune a model (if finetuning the model needs to be preprocessed with `ptr`)",
    )
    parser.add_argument(
        "--tokenizer_path",
        help="path to the tokenizer file",
        type=str,
        default="tokenizer.json",
    )
    parser.add_argument(
        "--resume_path", help="path to resume from a checkpoint file", type=str
    )
    parser.add_argument(
        "--checkpoint_path",
        help="directory to save checkpoints to",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--training_data", help="path to training data", type=str, required=True
    )

    # training parameters
    parser.add_argument("--ctx_size", help="context size", type=int, default=1024)
    parser.add_argument("--batch_size", help="batch size", type=int, default=1)
    parser.add_argument(
        "--epochs", help="number of epochs to train for", type=int, default=1
    )
    parser.add_argument(
        "--start_epoch", help="epoch to start training at", type=int, default=0
    )
    parser.add_argument(
        "--steps", help="number of steps to train for per epoch", type=int, default=1
    )
    parser.add_argument(
        "--gradient_accumulation", help="gradient accumulation", type=int, default=1
    )
    parser.add_argument(
        "--start_lr", help="starting learning rate", type=float, default=1e-5
    )
    parser.add_argument(
        "--end_lr", help="ending learning rate", type=float, default=1e-6
    )
    parser.add_argument(
        "--warmup_steps", help="number of warmup steps", type=int, default=0
    )
    parser.add_argument(
        "--optimizer",
        help="optimizer to use (adamw or lion)",
        type=str,
        default="adamw",
    )
    parser.add_argument("--adam_b1", help="beta1 for AdamW", type=float, default=0.9)
    parser.add_argument("--adam_b2", help="beta2 for AdamW", type=float, default=0.999)
    parser.add_argument(
        "--adam_wd", help="weight decay for AdamW", type=float, default=0.01
    )
    parser.add_argument("--lion_b1", help="beta1 for Lion", type=float, default=0.9)
    parser.add_argument("--lion_b2", help="beta2 for Lion", type=float, default=0.99)
    parser.add_argument(
        "--lion_wd", help="weight decay for Lion", type=float, default=0.0
    )

    # wandb
    parser.add_argument("--wandb", help="use wandb", action="store_true")
    parser.add_argument("--wandb_project", help="wandb project name", type=str)

    parser.set_defaults(func=train)


def train(args: Namespace) -> None:
    Tensor.training = True

    # sanity checks
    assert args.warmup_steps < args.steps, "warmup steps must be less than steps"

    # load tokenizer
    tokenizer = Tokenizer.from_file(args.tokenizer_path)

    # load model
    print("Loading model...")
    if args.resume_path is not None:
        model = RWKV_GPT(args.resume_path, args.ctx_size)
    else:
        raise NotImplementedError("TODO: implement pretraining from scratch")
    print(f"Model has ~{count_parameters(model) / 1000 / 1000}M parameters")
    assert (
        model.vocab_size == tokenizer.get_vocab_size()
    ), "vocab size mismatch (are you using the correct tokenizer?)"
    gc.collect()

    # setup optimizer
    print("Loading optimizer...")
    params = get_parameters(model)

    if args.optimizer == "adamw":
        optimizer = AdamW(
            params, lr=args.start_lr, b1=args.adam_b1, b2=args.adam_b2, wd=args.adam_wd
        )

        # load checkpointed optimizer state if it exists
        if args.resume_path is not None:
            if os.path.exists("optimizer-" + args.resume_path):
                with open("optimizer-" + args.resume_path, "rb") as f:
                    optimizer_state = pickle.load(f)

                optimizer.t.assign(optimizer_state["t"])
                for i in range(len(optimizer.m)):
                    optimizer.m[i].assign(optimizer_state["m"][i])
                for i in range(len(optimizer.v)):
                    optimizer.v[i].assign(optimizer_state["v"][i])
    elif args.optimizer == "lion":
        optimizer = Lion(
            params, lr=args.start_lr, b1=args.lion_b1, b2=args.lion_b2, wd=args.lion_wd
        )

        # load checkpointed optimizer state if it exists
        if args.resume_path is not None:
            if os.path.exists("optimizer-" + args.resume_path):
                with open("optimizer-" + args.resume_path, "rb") as f:
                    optimizer_state = pickle.load(f)

                for i in range(len(optimizer.ea)):
                    optimizer.ea[i].assign(optimizer_state["m"][i])

    else:
        raise NotImplementedError(f"unknown optimizer {args.optimizer}")
    gc.collect()

    # decay learning rate
    optimizer.lr = args.end_lr + 0.5 * (args.start_lr - args.end_lr) * (
        1
        + math.cos(
            ((args.start_epoch * args.steps) / (args.epochs * args.steps)) * math.pi
        )
    )

    # ensure that the checkpoint directory exists
    os.makedirs(args.checkpoint_path, exist_ok=True)

    print("Loading training data...")
    train_data = np.load(args.training_data).astype(int)
    print(f"Training data has {len(train_data) / 1000 / 1000}M tokens")
    gc.collect()

    # setup wandb
    if args.wandb:
        import wandb

        wandb.init(project=args.wandb_project)
        wandb.config.update(args)

    warming_up = True if args.start_epoch == 0 else False
    tokens_processed = 0
    optimizer.zero_grad()
    for epoch in range(args.start_epoch, args.epochs):
        for step in (t := trange(args.steps)):
            # calculate new learning rate for this step and epoch
            if warming_up and args.warmup_steps > 0:
                optimizer.lr = args.start_lr * (step / args.warmup_steps)
                if step >= args.warmup_steps:
                    warming_up = False
            else:
                optimizer.lr = args.end_lr + 0.5 * (args.start_lr - args.end_lr) * (
                    1
                    + math.cos(
                        ((step + (epoch * args.steps)) / (args.epochs * args.steps))
                        * math.pi
                    )
                )

            # gradient accumulation
            losses = []
            accuracies = []
            for mini_batch in range(args.gradient_accumulation):
                # sample training data
                sample = np.random.randint(
                    0, len(train_data) - (model.ctx_size + 1), size=args.batch_size
                )
                sampled = [
                    train_data[samp : samp + (model.ctx_size + 1)] for samp in sample
                ]

                x = Tensor([samp[:-1] for samp in sampled], requires_grad=False)
                y = np.array([samp[1:] for samp in sampled])

                # forward pass
                out = model.forward(x)
                out_lsm = out.log_softmax()
                loss = sparse_categorical_crossentropy(out_lsm, y)

                # scale gradients for gradient accumulation
                loss = loss / args.gradient_accumulation

                # backward pass
                loss.backward()

                # realize gradients
                for param in optimizer.params:
                    if param.grad is not None:
                        param.grad.realize()

                # keep track of some stats
                accuracy = (np.argmax(out.numpy(), axis=-1) == y).mean()
                loss = loss.numpy()
                accuracies.append(accuracy)
                losses.append(loss)

                # update tqdm
                t.set_description(
                    "mini batch %d | loss %.4f, acc %.4f | aloss %.4f, aacc %.4f"
                    % (
                        mini_batch,
                        loss * args.gradient_accumulation,
                        accuracy,
                        (sum(losses) / len(losses)) * args.gradient_accumulation,
                        sum(accuracies) / len(accuracies),
                    )
                )

            # update parameters
            optimizer.step()
            optimizer.zero_grad()

            # keep track of some stats
            tokens_processed += args.batch_size * model.ctx_size

            # wandb logging
            if args.wandb:
                wandb.log(
                    {
                        "loss": (sum(losses) / len(losses))
                        * args.gradient_accumulation,
                        "accuracy": sum(accuracies) / len(accuracies),
                        "lr": optimizer.lr,
                        "tokens_processed": tokens_processed,
                    }
                )

        # save model
        print("Saving model...")
        with open(
            os.path.join(args.checkpoint_path, f"epoch_{epoch + 1}.pkl"), "wb"
        ) as f:
            weights = {}
            for key, param in get_state_dict(model).items():
                weights[key] = param.numpy()
            pickle.dump(weights, f)

        # save optimizer
        print("Saving optimizer...")
        with open(
            os.path.join(args.checkpoint_path, f"optimizer-epoch_{epoch + 1}.pkl"), "wb"
        ) as f:
            if args.optimizer == "adamw":
                optimizer = cast(AdamW, optimizer)
                t = optimizer.t.numpy()
                m = []
                for tensor in optimizer.m:
                    m.append(tensor.numpy())
                v = []
                for tensor in optimizer.v:
                    v.append(tensor.numpy())

                pickle.dump({"t": t, "m": m, "v": v}, f)
            elif args.optimizer == "lion":
                optimizer = cast(Lion, optimizer)
                ea = []
                for tensor in optimizer.ea:
                    ea.append(tensor.numpy())

                pickle.dump({"ea": ea}, f)


def sparse_categorical_crossentropy(out, Y):
    channels = out.shape[-1]
    YY = Y.flatten().astype(np.int32)
    y = np.zeros((YY.shape[0], channels), np.float32)
    y[range(y.shape[0]), YY] = -1.0 * channels
    y = y.reshape(list(Y.shape) + [channels])
    y = Tensor(y)
    return out.mul(y).mean()


class Lion(Optimizer):
    def __init__(
        self,
        params: list[Tensor],
        lr: float = 1e-4,
        b1: float = 0.9,
        b2: float = 0.999,
        wd: float = 0.0,
    ):
        super().__init__(params)
        self.lr, self.b1, self.b2, self.wd = lr, b1, b2, wd

        self.ea = [
            Tensor.zeros(*t.shape, device=t.device, requires_grad=False)
            for t in self.params
        ]

    def step(self):
        for i, t in enumerate(self.params):
            assert t.grad is not None
            g = t.grad.realize()

            update = self.ea[i] * self.b1 + g * (1 - self.b1)
            self.ea[i].assign(self.ea[i] * self.b2 + g * (1 - self.b2))
            t.assign(t.detach() + update.sign() * -self.lr)
        self.realize(self.ea)
