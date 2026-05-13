"""Seq2Seq with attention for German-to-English translation on Multi30K.

The script is written for the Huawei Cloud ModelArts MindSpore 2.2 image.
It downloads data, trains the model, evaluates BLEU, and saves report assets.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import re
import time
import urllib.request
import zipfile
from collections import Counter, OrderedDict
from pathlib import Path
from typing import Iterable

import numpy as np
from nltk.translate.bleu_score import corpus_bleu
from tqdm import tqdm

import mindspore
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import Tensor, load_checkpoint, load_param_into_net, save_checkpoint


DATA_URL = "https://modelscope.cn/api/v1/datasets/SelinaRR/Multi30K/repo?Revision=master&FilePath=Multi30K.zip"
SPECIAL_TOKENS = ["<unk>", "<pad>", "<bos>", "<eos>"]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    mindspore.set_seed(seed)


def configure_context(device_target: str) -> None:
    if device_target.upper() == "CPU":
        os.environ["ASCEND_RT_VISIBLE_DEVICES"] = ""
    mindspore.set_context(mode=mindspore.GRAPH_MODE, device_target=device_target)


def make_dropout(dropout: float) -> nn.Cell:
    """Create Dropout for MindSpore 2.x while staying compatible with older labs."""
    try:
        return nn.Dropout(p=dropout)
    except TypeError:
        return nn.Dropout(keep_prob=1.0 - dropout)


def tokenize(text: str) -> list[str]:
    return [tok.lower() for tok in re.findall(r"\w+|[^\w\s]", text.rstrip(), flags=re.UNICODE)]


def find_dataset_root(search_root: Path) -> Path | None:
    if not search_root.exists():
        return None
    for de_file in search_root.rglob("train.de"):
        root = de_file.parent.parent
        if (
            (root / "train" / "train.de").exists()
            and (root / "train" / "train.en").exists()
            and (root / "valid" / "valid.de").exists()
            and (root / "test" / "test.de").exists()
        ):
            return root
    return None


def extract_existing_zip(search_root: Path) -> None:
    """Extract an existing Multi30K zip when download skipped because it exists."""
    zip_candidates = [search_root / "Multi30K.zip"]
    zip_candidates.extend(sorted(search_root.glob("*.zip")))
    seen: set[Path] = set()

    for zip_path in zip_candidates:
        zip_path = zip_path.resolve()
        if zip_path in seen or not zip_path.exists():
            continue
        seen.add(zip_path)
        try:
            print(f"Extracting existing dataset archive: {zip_path}")
            with zipfile.ZipFile(zip_path) as zf:
                zf.extractall(search_root)
        except zipfile.BadZipFile:
            print(f"Skip invalid zip archive: {zip_path}")


def ensure_dataset(data_dir: Path) -> Path:
    expected = data_dir / "train" / "train.de"
    if expected.exists():
        return data_dir

    data_dir.parent.mkdir(parents=True, exist_ok=True)
    discovered_root = find_dataset_root(data_dir.parent)
    if discovered_root is not None:
        return discovered_root

    extract_existing_zip(data_dir.parent)
    if expected.exists():
        return data_dir

    discovered_root = find_dataset_root(data_dir.parent)
    if discovered_root is not None:
        return discovered_root

    try:
        from download import download

        download(DATA_URL, str(data_dir.parent), kind="zip", replace=True)
    except Exception as exc:
        print(f"download package failed, falling back to urllib: {exc}")
        zip_path = data_dir.parent / "Multi30K.zip"
        urllib.request.urlretrieve(DATA_URL, zip_path)
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(data_dir.parent)

    if expected.exists():
        return data_dir

    extract_existing_zip(data_dir.parent)
    if expected.exists():
        return data_dir

    discovered_root = find_dataset_root(data_dir.parent)
    if discovered_root is not None:
        return discovered_root

    if not expected.exists():
        raise FileNotFoundError(
            f"Cannot find {expected}. Please check whether Multi30K.zip was extracted correctly."
        )
    return data_dir


class Multi30K:
    """Load Multi30K split files as tokenized German/English sentence pairs."""

    def __init__(self, path: Path):
        self.data = self._load(path)

    @staticmethod
    def _load(path: Path) -> list[tuple[list[str], list[str]]]:
        members = {p.suffix.lstrip("."): p for p in path.iterdir() if p.is_file()}
        de_path = members["de"]
        en_path = members["en"]
        with de_path.open("r", encoding="utf-8") as de_file:
            de_lines = [line for line in de_file.read().splitlines() if line.strip()]
        with en_path.open("r", encoding="utf-8") as en_file:
            en_lines = [line for line in en_file.read().splitlines() if line.strip()]

        pair_count = min(len(de_lines), len(en_lines))
        de = [tokenize(line) for line in de_lines[:pair_count]]
        en = [tokenize(line) for line in en_lines[:pair_count]]
        return list(zip(de, en))

    def __getitem__(self, idx: int) -> tuple[list[str], list[str]]:
        return self.data[idx]

    def __len__(self) -> int:
        return len(self.data)


class Vocab:
    """Build vocabulary from a word-frequency dictionary."""

    special_tokens = SPECIAL_TOKENS

    def __init__(self, word_count_dict: OrderedDict[str, int], min_freq: int = 1):
        self.word2idx: dict[str, int] = {}
        for idx, tok in enumerate(self.special_tokens):
            self.word2idx[tok] = idx

        filtered_dict = OrderedDict(
            (word, count) for word, count in word_count_dict.items() if count >= min_freq
        )
        for word in filtered_dict.keys():
            if word not in self.word2idx:
                self.word2idx[word] = len(self.word2idx)

        self.idx2word = {idx: word for word, idx in self.word2idx.items()}
        self.unk_idx = self.word2idx["<unk>"]
        self.pad_idx = self.word2idx["<pad>"]
        self.bos_idx = self.word2idx["<bos>"]
        self.eos_idx = self.word2idx["<eos>"]

    def _word2idx(self, word: str) -> int:
        return self.word2idx.get(word, self.unk_idx)

    def _idx2word(self, idx: int) -> str:
        if idx not in self.idx2word:
            raise ValueError(f"input index {idx} is not in vocabulary.")
        return self.idx2word[idx]

    def encode(self, word_or_list: str | list[str]) -> int | list[int]:
        if isinstance(word_or_list, list):
            return [self._word2idx(word) for word in word_or_list]
        return self._word2idx(word_or_list)

    def decode(self, idx_or_list: int | list[int]) -> str | list[str]:
        if isinstance(idx_or_list, list):
            return [self._idx2word(int(idx)) for idx in idx_or_list]
        return self._idx2word(int(idx_or_list))

    def __len__(self) -> int:
        return len(self.word2idx)


def build_vocab(dataset: Multi30K, min_freq: int = 2) -> tuple[Vocab, Vocab]:
    de_words: list[str] = []
    en_words: list[str] = []
    for de, en in dataset:
        de_words.extend(de)
        en_words.extend(en)

    de_count_dict = OrderedDict(sorted(Counter(de_words).items(), key=lambda t: t[1], reverse=True))
    en_count_dict = OrderedDict(sorted(Counter(en_words).items(), key=lambda t: t[1], reverse=True))
    return Vocab(de_count_dict, min_freq=min_freq), Vocab(en_count_dict, min_freq=min_freq)


class Iterator:
    """Create mini-batches and pad source/target sequences."""

    def __init__(
        self,
        dataset: Multi30K,
        de_vocab: Vocab,
        en_vocab: Vocab,
        batch_size: int,
        max_len: int = 32,
        drop_remainder: bool = False,
    ):
        self.dataset = dataset
        self.de_vocab = de_vocab
        self.en_vocab = en_vocab
        self.batch_size = batch_size
        self.max_len = max_len
        self.drop_remainder = drop_remainder
        if drop_remainder:
            self.len = len(dataset) // batch_size
        else:
            self.len = math.ceil(len(dataset) / batch_size)

    @staticmethod
    def _pad(idx_list: Iterable[list[int]], vocab: Vocab, max_len: int) -> tuple[list[list[int]], list[int]]:
        idx_pad_list: list[list[int]] = []
        idx_len: list[int] = []
        for item in idx_list:
            if len(item) > max_len - 2:
                idx_pad_list.append([vocab.bos_idx] + item[: max_len - 2] + [vocab.eos_idx])
                idx_len.append(max_len)
            else:
                item_len = len(item) + 2
                idx_pad_list.append(
                    [vocab.bos_idx] + item + [vocab.eos_idx] + [vocab.pad_idx] * (max_len - item_len)
                )
                idx_len.append(item_len)
        return idx_pad_list, idx_len

    @staticmethod
    def _sort_by_length(src: list[list[int]], trg: list[list[int]]) -> tuple[list[list[int]], list[list[int]]]:
        data = sorted(zip(src, trg), key=lambda t: len(t[0]), reverse=True)
        sorted_src, sorted_trg = zip(*data)
        return list(sorted_src), list(sorted_trg)

    def _encode_and_pad(self, batch_data: list[tuple[list[str], list[str]]]) -> tuple[list[list[int]], list[int], list[list[int]]]:
        src_data, trg_data = zip(*batch_data)
        src_idx = [self.de_vocab.encode(list(item)) for item in src_data]
        trg_idx = [self.en_vocab.encode(list(item)) for item in trg_data]
        src_idx, trg_idx = self._sort_by_length(src_idx, trg_idx)
        src_idx_pad, src_len = self._pad(src_idx, self.de_vocab, self.max_len)
        trg_idx_pad, _ = self._pad(trg_idx, self.en_vocab, self.max_len)
        return src_idx_pad, src_len, trg_idx_pad

    def __call__(self):
        for batch_idx in range(self.len):
            start = batch_idx * self.batch_size
            end = start + self.batch_size
            batch_data = self.dataset.data[start:end]
            if self.drop_remainder and len(batch_data) < self.batch_size:
                continue
            src_idx, src_len, trg_idx = self._encode_and_pad(batch_data)
            yield (
                Tensor(src_idx, mindspore.int32),
                Tensor(src_len, mindspore.int32),
                Tensor(trg_idx, mindspore.int32),
            )

    def __len__(self) -> int:
        return self.len


class Encoder(nn.Cell):
    def __init__(self, input_dim: int, emb_dim: int, enc_hid_dim: int, dec_hid_dim: int, dropout: float, compute_dtype):
        super().__init__()
        self.embedding = nn.Embedding(input_dim, emb_dim)
        self.rnn = nn.GRU(
            input_size=emb_dim,
            hidden_size=enc_hid_dim,
            num_layers=1,
            has_bias=True,
            batch_first=False,
            bidirectional=True,
        )
        self.fc = nn.Dense(enc_hid_dim * 2, dec_hid_dim).to_float(compute_dtype)
        self.dropout = make_dropout(dropout)

    def construct(self, src, src_len):
        embedded = self.dropout(self.embedding(src))
        outputs, hidden = self.rnn(embedded, seq_length=src_len)
        hidden = ops.concat((hidden[-2, :, :], hidden[-1, :, :]), axis=1)
        hidden = ops.tanh(self.fc(hidden))
        return outputs, hidden


class Attention(nn.Cell):
    def __init__(self, enc_hid_dim: int, dec_hid_dim: int, compute_dtype):
        super().__init__()
        self.attn = nn.Dense((enc_hid_dim * 2) + dec_hid_dim, dec_hid_dim).to_float(compute_dtype)
        self.v = nn.Dense(dec_hid_dim, 1, has_bias=False).to_float(compute_dtype)

    def construct(self, hidden, encoder_outputs, mask):
        src_len = encoder_outputs.shape[0]
        hidden = ops.tile(hidden.expand_dims(1), (1, src_len, 1))
        encoder_outputs = encoder_outputs.swapaxes(0, 1)
        energy = ops.tanh(self.attn(ops.concat((hidden, encoder_outputs), axis=2)))
        attention = self.v(energy).squeeze(2)
        attention = ops.select(mask, attention, ops.full_like(attention, -1e10))
        return ops.softmax(attention, axis=1)


class Decoder(nn.Cell):
    def __init__(
        self,
        output_dim: int,
        emb_dim: int,
        enc_hid_dim: int,
        dec_hid_dim: int,
        dropout: float,
        attention: Attention,
        compute_dtype,
    ):
        super().__init__()
        self.output_dim = output_dim
        self.attention = attention
        self.embedding = nn.Embedding(output_dim, emb_dim)
        self.rnn = nn.GRU(
            input_size=(enc_hid_dim * 2) + emb_dim,
            hidden_size=dec_hid_dim,
            num_layers=1,
            has_bias=True,
            batch_first=False,
            bidirectional=False,
        )
        self.fc_out = nn.Dense((enc_hid_dim * 2) + dec_hid_dim + emb_dim, output_dim).to_float(compute_dtype)
        self.dropout = make_dropout(dropout)

    def construct(self, inputs, hidden, encoder_outputs, mask):
        inputs = inputs.expand_dims(0)
        embedded = self.dropout(self.embedding(inputs))

        a = self.attention(hidden, encoder_outputs, mask)
        a_unsqueezed = a.expand_dims(1)
        encoder_outputs = encoder_outputs.swapaxes(0, 1)
        weighted = ops.bmm(a_unsqueezed, encoder_outputs)
        weighted = weighted.swapaxes(0, 1)

        rnn_input = ops.concat((embedded, weighted), axis=2)
        output, hidden = self.rnn(rnn_input, hidden.expand_dims(0))

        embedded = embedded.squeeze(0)
        output = output.squeeze(0)
        weighted = weighted.squeeze(0)
        prediction = self.fc_out(ops.concat((output, weighted, embedded), axis=1))
        return prediction, hidden.squeeze(0), a


class Seq2Seq(nn.Cell):
    def __init__(self, encoder: Encoder, decoder: Decoder, src_pad_idx: int, teacher_forcing_ratio: float, dtype):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_pad_idx = src_pad_idx
        self.teacher_forcing_ratio = Tensor(teacher_forcing_ratio, mindspore.float32)
        self.dtype = dtype
        self.uniform = ops.UniformReal(seed=0)

    def create_mask(self, src):
        return (src != self.src_pad_idx).swapaxes(0, 1)

    def construct(self, src, src_len, trg, trg_len=None):
        if trg_len is None:
            trg_len = trg.shape[0]

        outputs = []
        encoder_outputs, hidden = self.encoder(src, src_len)
        inputs = trg[0, :]
        mask = self.create_mask(src)

        for t in range(1, trg_len):
            output, hidden, _ = self.decoder(inputs, hidden, encoder_outputs, mask)
            outputs.append(output)
            top1 = ops.argmax(output, axis=1).astype(mindspore.int32)
            if self.training:
                random_values = self.uniform((trg.shape[1],))
                teacher_force = random_values < self.teacher_forcing_ratio
                inputs = ops.select(teacher_force, trg[t], top1)
            else:
                inputs = top1

        outputs = ops.stack(outputs, axis=0)
        return outputs.astype(self.dtype)


def build_model(args, de_vocab: Vocab, en_vocab: Vocab) -> tuple[Seq2Seq, int]:
    compute_dtype = mindspore.float32
    dtype = mindspore.float32
    attn = Attention(args.enc_hid_dim, args.dec_hid_dim, compute_dtype)
    encoder = Encoder(len(de_vocab), args.enc_emb_dim, args.enc_hid_dim, args.dec_hid_dim, args.enc_dropout, compute_dtype)
    decoder = Decoder(
        len(en_vocab),
        args.dec_emb_dim,
        args.enc_hid_dim,
        args.dec_hid_dim,
        args.dec_dropout,
        attn,
        compute_dtype,
    )
    model = Seq2Seq(encoder, decoder, de_vocab.pad_idx, args.teacher_forcing_ratio, dtype)
    return model, en_vocab.pad_idx


def clip_by_norm(clip_norm, tensor, axis=None):
    t2 = tensor * tensor
    l2sum = t2.sum(axis=axis, keepdims=True)
    pred = l2sum > 0
    l2sum_safe = ops.select(pred, l2sum, ops.ones_like(l2sum))
    l2norm = ops.select(pred, ops.sqrt(l2sum_safe), l2sum)
    intermediate = tensor * clip_norm
    cond = l2norm > clip_norm
    tensor_clip = intermediate / ops.select(cond, l2norm, clip_norm)
    return tensor_clip


def train_one_epoch(model, iterator: Iterator, train_step, clip: float, epoch: int) -> float:
    model.set_train(True)
    total_loss = 0.0
    total_steps = 0
    with tqdm(total=len(iterator), desc=f"Epoch: {epoch}") as progress:
        for src, src_len, trg in iterator():
            loss = train_step(src, src_len, trg, clip)
            total_loss += float(loss.asnumpy())
            total_steps += 1
            progress.set_postfix({"loss": f"{total_loss / total_steps:.2f}"})
            progress.update(1)
    return total_loss / max(total_steps, 1)


def evaluate(model, iterator: Iterator, forward_fn) -> float:
    model.set_train(False)
    total_loss = 0.0
    total_steps = 0
    with tqdm(total=len(iterator), desc="Evaluate") as progress:
        for src, src_len, trg in iterator():
            loss = forward_fn(src, src_len, trg)
            total_loss += float(loss.asnumpy())
            total_steps += 1
            progress.set_postfix({"loss": f"{total_loss / total_steps:.2f}"})
            progress.update(1)
    return total_loss / max(total_steps, 1)


def ids_to_tokens(indexes: list[int], vocab: Vocab) -> list[str]:
    if vocab.eos_idx in indexes:
        indexes = indexes[: indexes.index(vocab.eos_idx)]
    return [tok for tok in vocab.decode(indexes) if tok not in SPECIAL_TOKENS]


def prepare_source(sentence: str | list[str], de_vocab: Vocab, max_len: int) -> tuple[list[str], list[int], int]:
    if isinstance(sentence, str):
        tokens = tokenize(sentence)
    else:
        tokens = [token.lower() for token in sentence]

    if len(tokens) > max_len - 2:
        src_len = max_len
        padded_tokens = ["<bos>"] + tokens[: max_len - 2] + ["<eos>"]
    else:
        src_len = len(tokens) + 2
        padded_tokens = ["<bos>"] + tokens + ["<eos>"] + ["<pad>"] * (max_len - src_len)
    src_indexes = de_vocab.encode(padded_tokens)
    return padded_tokens, src_indexes, src_len


def translate_sentence(sentence: str | list[str], de_vocab: Vocab, en_vocab: Vocab, model: Seq2Seq, max_len: int = 32) -> list[str]:
    model.set_train(False)
    _, src_indexes, src_len = prepare_source(sentence, de_vocab, max_len)
    src = Tensor(src_indexes, mindspore.int32).expand_dims(1)
    src_len_tensor = Tensor([src_len], mindspore.int32)
    trg = Tensor([en_vocab.bos_idx], mindspore.int32).expand_dims(1)
    outputs = model(src, src_len_tensor, trg, max_len)
    trg_indexes = [int(ops.argmax(step, axis=1).asnumpy()[0]) for step in outputs]
    return ids_to_tokens(trg_indexes, en_vocab)


def translate_with_attention(
    sentence: str | list[str],
    de_vocab: Vocab,
    en_vocab: Vocab,
    model: Seq2Seq,
    max_len: int = 32,
) -> tuple[list[str], list[str], np.ndarray]:
    model.set_train(False)
    src_tokens, src_indexes, src_len = prepare_source(sentence, de_vocab, max_len)
    src = Tensor(src_indexes, mindspore.int32).expand_dims(1)
    src_len_tensor = Tensor([src_len], mindspore.int32)
    encoder_outputs, hidden = model.encoder(src, src_len_tensor)
    mask = model.create_mask(src)
    inputs = Tensor([en_vocab.bos_idx], mindspore.int32)

    trg_indexes: list[int] = []
    attentions: list[np.ndarray] = []
    for _ in range(max_len - 1):
        output, hidden, attention = model.decoder(inputs, hidden, encoder_outputs, mask)
        top1 = int(ops.argmax(output, axis=1).asnumpy()[0])
        trg_indexes.append(top1)
        attentions.append(attention.asnumpy()[0][:src_len])
        inputs = Tensor([top1], mindspore.int32)
        if top1 == en_vocab.eos_idx:
            break

    trg_tokens = ids_to_tokens(trg_indexes, en_vocab)
    attention_array = np.asarray(attentions[: len(trg_tokens)], dtype=np.float32)
    return src_tokens[:src_len], trg_tokens, attention_array


def calculate_bleu(dataset: Multi30K, de_vocab: Vocab, en_vocab: Vocab, model: Seq2Seq, max_len: int, limit: int = 0) -> float:
    trgs = []
    pred_trgs = []
    eval_data = dataset.data[:limit] if limit > 0 else dataset.data
    for src, trg in tqdm(eval_data, desc="BLEU"):
        pred_trg = translate_sentence(src, de_vocab, en_vocab, model, max_len)
        pred_trgs.append(pred_trg)
        trgs.append([trg])
    return corpus_bleu(trgs, pred_trgs)


def save_training_curve(history: list[dict[str, float]], output_dir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    epochs = [int(row["epoch"]) for row in history]
    train_losses = [row["train_loss"] for row in history]
    valid_losses = [row["valid_loss"] for row in history]

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_losses, marker="o", label="train loss")
    plt.plot(epochs, valid_losses, marker="s", label="valid loss")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.title("Seq2Seq training curve")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "training_curve.png", dpi=160)
    plt.close()


def save_attention_heatmap(src_tokens: list[str], trg_tokens: list[str], attention: np.ndarray, output_dir: Path) -> None:
    if attention.size == 0:
        return

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.figure(figsize=(max(7, len(src_tokens) * 0.6), max(4, len(trg_tokens) * 0.45)))
    plt.imshow(attention, cmap="viridis", aspect="auto")
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.xticks(range(len(src_tokens)), src_tokens, rotation=45, ha="right")
    plt.yticks(range(len(trg_tokens)), trg_tokens)
    plt.xlabel("source tokens")
    plt.ylabel("predicted tokens")
    plt.title("Attention heatmap")
    plt.tight_layout()
    plt.savefig(output_dir / "attention_heatmap.png", dpi=180)
    plt.close()


def write_history_csv(history: list[dict[str, float]], output_dir: Path) -> None:
    with (output_dir / "training_history.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["epoch", "train_loss", "valid_loss", "train_ppl", "valid_ppl", "seconds"])
        writer.writeheader()
        writer.writerows(history)


def write_predictions(test_dataset: Multi30K, de_vocab: Vocab, en_vocab: Vocab, model: Seq2Seq, max_len: int, output_dir: Path) -> list[dict[str, str]]:
    sample_indices = [0, 1, 2, 10, 20]
    rows: list[dict[str, str]] = []
    with (output_dir / "predictions.txt").open("w", encoding="utf-8") as f:
        for idx in sample_indices:
            if idx >= len(test_dataset):
                continue
            src, trg = test_dataset[idx]
            pred = translate_sentence(src, de_vocab, en_vocab, model, max_len)
            row = {
                "index": str(idx),
                "src": " ".join(src),
                "trg": " ".join(trg),
                "pred": " ".join(pred),
            }
            rows.append(row)
            f.write(f"[{idx}]\n")
            f.write(f"src:  {row['src']}\n")
            f.write(f"trg:  {row['trg']}\n")
            f.write(f"pred: {row['pred']}\n\n")
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MindSpore Seq2Seq German-English translation")
    parser.add_argument("--data_dir", type=Path, default=Path("datasets"))
    parser.add_argument("--output_dir", type=Path, default=Path("outputs"))
    parser.add_argument("--device_target", choices=["CPU", "Ascend", "GPU"], default="CPU")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--max_len", type=int, default=32)
    parser.add_argument("--min_freq", type=int, default=2)
    parser.add_argument("--enc_emb_dim", type=int, default=256)
    parser.add_argument("--dec_emb_dim", type=int, default=256)
    parser.add_argument("--enc_hid_dim", type=int, default=512)
    parser.add_argument("--dec_hid_dim", type=int, default=512)
    parser.add_argument("--enc_dropout", type=float, default=0.5)
    parser.add_argument("--dec_dropout", type=float, default=0.5)
    parser.add_argument("--teacher_forcing_ratio", type=float, default=0.5)
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bleu_limit", type=int, default=0, help="0 means evaluate the full test set.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)
    configure_context(args.device_target)
    args.data_dir = ensure_dataset(args.data_dir)

    train_dataset = Multi30K(args.data_dir / "train")
    valid_dataset = Multi30K(args.data_dir / "valid")
    test_dataset = Multi30K(args.data_dir / "test")
    de_vocab, en_vocab = build_vocab(train_dataset, min_freq=args.min_freq)

    print(f"train/valid/test: {len(train_dataset)}/{len(valid_dataset)}/{len(test_dataset)}")
    print(f"vocab: de={len(de_vocab)}, en={len(en_vocab)}")

    train_iterator = Iterator(train_dataset, de_vocab, en_vocab, args.batch_size, args.max_len, drop_remainder=True)
    valid_iterator = Iterator(valid_dataset, de_vocab, en_vocab, args.batch_size, args.max_len, drop_remainder=False)

    model, trg_pad_idx = build_model(args, de_vocab, en_vocab)
    opt = nn.Adam(model.trainable_params(), learning_rate=args.learning_rate)
    loss_fn = nn.CrossEntropyLoss(ignore_index=trg_pad_idx)

    def forward_fn(src, src_len, trg):
        src = src.swapaxes(0, 1)
        trg = trg.swapaxes(0, 1)
        output = model(src, src_len, trg)
        output_dim = output.shape[-1]
        output = output.view(-1, output_dim)
        trg = trg[1:].view(-1)
        loss = loss_fn(output, trg)
        return loss

    grad_fn = mindspore.value_and_grad(forward_fn, None, opt.parameters)

    def train_step(src, src_len, trg, clip):
        loss, grads = grad_fn(src, src_len, trg)
        grads = ops.HyperMap()(ops.partial(clip_by_norm, clip), grads)
        opt(grads)
        return loss

    ckpt_file_name = args.output_dir / "seq2seq.ckpt"
    best_valid_loss = float("inf")
    history: list[dict[str, float]] = []

    for epoch in range(1, args.epochs + 1):
        start = time.time()
        train_loss = train_one_epoch(model, train_iterator, train_step, args.clip, epoch)
        valid_loss = evaluate(model, valid_iterator, forward_fn)
        seconds = time.time() - start
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            save_checkpoint(model, str(ckpt_file_name))

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "valid_loss": valid_loss,
            "train_ppl": float(math.exp(min(train_loss, 20))),
            "valid_ppl": float(math.exp(min(valid_loss, 20))),
            "seconds": seconds,
        }
        history.append(row)
        print(
            f"epoch {epoch}: train_loss={train_loss:.3f}, valid_loss={valid_loss:.3f}, "
            f"train_ppl={row['train_ppl']:.2f}, valid_ppl={row['valid_ppl']:.2f}, seconds={seconds:.1f}"
        )

    write_history_csv(history, args.output_dir)
    save_training_curve(history, args.output_dir)

    if ckpt_file_name.exists():
        param_dict = load_checkpoint(str(ckpt_file_name))
        load_param_into_net(model, param_dict)

    predictions = write_predictions(test_dataset, de_vocab, en_vocab, model, args.max_len, args.output_dir)
    bleu_score = calculate_bleu(test_dataset, de_vocab, en_vocab, model, args.max_len, args.bleu_limit)

    src_tokens, trg_tokens, attention = translate_with_attention(test_dataset[0][0], de_vocab, en_vocab, model, args.max_len)
    save_attention_heatmap(src_tokens, trg_tokens, attention, args.output_dir)

    metrics = {
        "device_target": args.device_target,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "max_len": args.max_len,
        "min_freq": args.min_freq,
        "de_vocab_size": len(de_vocab),
        "en_vocab_size": len(en_vocab),
        "best_valid_loss": best_valid_loss,
        "bleu": bleu_score,
        "bleu_percent": bleu_score * 100,
        "bleu_limit": args.bleu_limit,
        "predictions": predictions,
        "history": history,
        "artifacts": {
            "checkpoint": str(ckpt_file_name),
            "training_history": str(args.output_dir / "training_history.csv"),
            "training_curve": str(args.output_dir / "training_curve.png"),
            "attention_heatmap": str(args.output_dir / "attention_heatmap.png"),
            "predictions": str(args.output_dir / "predictions.txt"),
        },
    }
    with (args.output_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    print(f"BLEU score = {bleu_score * 100:.2f}")
    print(f"Artifacts saved to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
