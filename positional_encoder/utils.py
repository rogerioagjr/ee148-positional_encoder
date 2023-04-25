import torch
from torch import Tensor, nn
from torch.utils.data import dataset
from torchtext.vocab.vocab import Vocab
from typing import Callable, Tuple

from torchtext.datasets import WikiText2
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator

import time

import math


def data_process(raw_text_iter: dataset.IterableDataset, vocab: Vocab, tokenizer: Callable) -> Tensor:
    """Converts raw text into a flat Tensor."""
    data = [torch.tensor(vocab(tokenizer(item)), dtype=torch.long) for item in raw_text_iter]
    return torch.cat(tuple(filter(lambda t: t.numel() > 0, data)))


def batchify(data: Tensor, bsz: int) -> Tensor:
    """Divides the data into ``bsz`` separate sequences, removing extra elements
    that wouldn't cleanly fit.

    Arguments:
        data: Tensor, shape [N]
        bsz: int, batch size

    Returns:
        Tensor of shape ``[N // bsz, bsz]``
    """
    seq_len = data.size(0) // bsz
    data = data[:seq_len * bsz]
    data = data.view(bsz, seq_len).t().contiguous()
    return data


def get_wikitext2_data(train_batch_size: int, eval_batch_size: int, device: str = 'cpu') \
        -> Tuple[Tensor, Tensor, Tensor]:
    """
    Helper Function to get tokenized and batched train, validation, and test splits from the WikiText-2 dataset
    (https://paperswithcode.com/dataset/wikitext-2)

    Arguments:
        train_batch_size: int
        eval_batch_size: int
        device: str,

    Returns:
        tuple (train_data, val_data, test_data, vocab: Vocab), where train_data has shape ``[seq_len, train_batch_size]`` and
        val_data, test_data have shape ``[seq_len, eval_batch_size]``
    """
    train_iter = WikiText2(split='train')
    tokenizer = get_tokenizer('basic_english')
    vocab = build_vocab_from_iterator(map(tokenizer, train_iter), specials=['<unk>'])
    vocab.set_default_index(vocab['<unk>'])

    # ``train_iter`` was "consumed" by the process of building the vocab,
    # so we have to create it again
    train_iter, val_iter, test_iter = WikiText2()
    train_data = data_process(train_iter, vocab=vocab, tokenizer=tokenizer)
    val_data = data_process(val_iter, vocab=vocab, tokenizer=tokenizer)
    test_data = data_process(test_iter, vocab=vocab, tokenizer=tokenizer)

    train_data = batchify(train_data, train_batch_size).to(device)  # shape ``[seq_len, batch_size]``
    val_data = batchify(val_data, eval_batch_size).to(device)
    test_data = batchify(test_data, eval_batch_size).to(device)

    return train_data, val_data, test_data, vocab


def get_batch(source: Tensor, i: int, max_seq_len: int = 35) -> Tuple[Tensor, Tensor]:
    """
    Arguments:
        source: Tensor, shape ``[full_seq_len, batch_size]``
        i: int
        max_seq_len: int,

    Returns:
        tuple (data, target), where data has shape ``[seq_len, batch_size]`` and
        target has shape ``[seq_len * batch_size]``
    """
    seq_len = min(max_seq_len, len(source) - 1 - i)
    data = source[i:i+seq_len]
    target = source[i+1:i+1+seq_len].reshape(-1)
    return data, target


def generate_square_subsequent_mask(sz: int) -> Tensor:
    """Generates an upper-triangular matrix of ``-inf``, with zeros on ``diag``."""
    return torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal=1)


def train(model: nn.Module, train_data: Tensor, max_seq_len: int, criterion: Callable, lr: float,
          n_tokens: int, epoch: int , device: str = 'cpu', verbose: bool = True) -> None:
    """
    Arguments:
        model: nn.Module
        train_data: Tensor, shape ``[full_seq_len, batch_size]``
        max_seq_len: int
        criterion: Callable, returns the loss function
        lr: float
        n_tokens: int
        epoch: int
        device: str
        verbose: bool
    """

    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.95)

    model.train()  # turn on train mode
    total_loss = 0.
    log_interval = 200
    start_time = time.time() if verbose else None
    src_mask = generate_square_subsequent_mask(max_seq_len).to(device)

    num_batches = len(train_data) // max_seq_len
    for batch, i in enumerate(range(0, train_data.size(0) - 1, max_seq_len)):
        data, targets = get_batch(train_data, i)
        seq_len = data.size(0)
        if seq_len != max_seq_len:  # only on last batch
            src_mask = src_mask[:seq_len, :seq_len]
        output = model(data, src_mask)
        loss = criterion(output.view(-1, n_tokens), targets)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()

        total_loss += loss.item()
        if verbose and batch % log_interval == 0 and batch > 0:
            lr = scheduler.get_last_lr()[0]
            ms_per_batch = (time.time() - start_time) * 1000 / log_interval
            cur_loss = total_loss / log_interval
            ppl = math.exp(cur_loss)
            print(f'| epoch {epoch:3d} | {batch:5d}/{num_batches:5d} batches | '
                  f'lr {lr:02.2f} | ms/batch {ms_per_batch:5.2f} | '
                  f'loss {cur_loss:5.2f} | ppl {ppl:8.2f}')
            total_loss = 0
            start_time = time.time()


def evaluate(model: nn.Module, eval_data: Tensor) -> float:
    model.eval()  # turn on evaluation mode
    total_loss = 0.
    src_mask = generate_square_subsequent_mask(bptt).to(device)
    with torch.no_grad():
        for i in range(0, eval_data.size(0) - 1, bptt):
            data, targets = get_batch(eval_data, i)
            seq_len = data.size(0)
            if seq_len != bptt:
                src_mask = src_mask[:seq_len, :seq_len]
            output = model(data, src_mask)
            output_flat = output.view(-1, n_tokens)
            total_loss += seq_len * criterion(output_flat, targets).item()
    return total_loss / (len(eval_data) - 1)
