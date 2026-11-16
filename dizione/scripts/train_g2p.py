"""Train the unknown-word model: letters -> CMUdict phonemes with stress. Cloud only.

Bidirectional GRU layers (optionally preceded by residual convolutions) read
the letters once and a CTC output layer emits the phonemes: one forward pass
per word, no decoder loop, in the spirit of LiteG2P. Every letter is given
``slots`` output positions so that "x" can yield two phonemes.

The defaults are the recipe of the released weights, found on 14 Nov 2026:
64-wide letter embeddings, two GRU layers of 127 units, no convolution, no
dropout (it made this small model under-fit), weight decay 0.05, 80 epochs,
and a running average of the weights. 458,682 parameters.

The target is CMUdict's own notation (ARPABET with stress digits), not the TTS
inventory: the same conversion rules that build the dictionary are applied to
the model's output afterwards, so changing a rule never needs a retrain.

Data: CMUdict only. Words are split by a hash of the spelling (all variants of
a word stay together): 5% test, 2% validation, the rest training. The exported
weights are those of the best validation epoch; the test words are never seen.

    python train_g2p.py --cmudict data/cmudict/cmudict.dict --output out/g2p
"""

import argparse
import hashlib
import json
import math
from pathlib import Path
import random
import re
import time

import numpy as np
import torch
from torch import nn

LETTERS = "abcdefghijklmnopqrstuvwxyz'"
PARAMETER_BUDGET = 462_709      # headroom of the TTS model under its 5M limit
VOWEL_LETTERS = set("aeiouy")
_WORD = re.compile(r"^[a-z]+(?:'[a-z]+)*$")


def read_cmudict(path):
    """word -> list of pronunciations (lists of ARPABET symbols), file order kept."""
    entries = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.split("#")[0].strip()
        if not line:
            continue
        word, *phones = line.split()
        word = re.sub(r"\(\d+\)$", "", word)
        variants = entries.setdefault(word, [])
        if phones and phones not in variants:
            variants.append(phones)
    return entries


def trainable(word, phones):
    """Ordinary spelled words only. Abbreviations and initialisms ("dept",
    "fbi", "tv") are read by rule elsewhere and would teach letter names."""
    if not _WORD.match(word) or not 2 <= len(word) <= 30:
        return False
    if not VOWEL_LETTERS & set(word):
        return False
    return len(phones) <= len(word) + 2


def split_of(word):
    bucket = int(hashlib.sha256(word.encode()).hexdigest()[:8], 16) % 100
    return "test" if bucket < 5 else "validation" if bucket < 7 else "train"


def load_splits(path):
    splits = {"train": [], "validation": [], "test": []}
    skipped = 0
    for word, variants in read_cmudict(path).items():
        kept = [v for v in variants if trainable(word, v)]
        if not kept:
            skipped += 1
            continue
        splits[split_of(word)].append((word, kept))
    return splits, skipped


class G2P(nn.Module):
    def __init__(self, labels, width=64, hidden=127, convs=0, kernel=3, layers=2, slots=2, dropout=0.0):
        super().__init__()
        self.settings = {"width": width, "hidden": hidden, "convs": convs, "kernel": kernel, "layers": layers, "slots": slots}
        self.slots = slots
        self.embed = nn.Embedding(len(LETTERS) + 1, width, padding_idx=0)
        self.slot = nn.Parameter(torch.zeros(slots, width))
        self.norms = nn.ModuleList([nn.LayerNorm(width) for _ in range(convs)])
        self.convs = nn.ModuleList([nn.Conv1d(width, width, kernel, padding=kernel // 2) for _ in range(convs)])
        self.gru = nn.GRU(width, hidden, num_layers=layers, bidirectional=True, batch_first=True,
                          dropout=dropout if layers > 1 else 0.0)
        self.drop = nn.Dropout(dropout)
        self.out = nn.Linear(2 * hidden, labels + 1)      # label 0 is the CTC blank
        nn.init.normal_(self.slot, std=0.1)

    def forward(self, letters, lengths):
        """letters: (batch, max letters) ids, 0 = padding. Returns log-probabilities
        (batch, max letters * slots, labels + 1) and the frame lengths."""
        batch, count = letters.shape
        x = self.embed(letters).unsqueeze(2) + self.slot.view(1, 1, self.slots, -1)
        x = x.reshape(batch, count * self.slots, -1)
        frames = lengths * self.slots
        mask = (torch.arange(count * self.slots, device=letters.device)[None, :] < frames[:, None]).unsqueeze(-1)
        x = self.drop(x) * mask
        for norm, conv in zip(self.norms, self.convs):
            x = (x + torch.relu(conv((norm(x) * mask).transpose(1, 2)).transpose(1, 2))) * mask
        packed = nn.utils.rnn.pack_padded_sequence(x, frames.cpu(), batch_first=True, enforce_sorted=False)
        x, _ = nn.utils.rnn.pad_packed_sequence(self.gru(packed)[0], batch_first=True, total_length=count * self.slots)
        return torch.log_softmax(self.out(self.drop(x)), dim=-1), frames


def encode_word(word):
    return [LETTERS.index(c) + 1 for c in word]


def batches(examples, size, rng=None):
    """Batches of similar length. ``examples`` are (letter ids, label ids)."""
    order = list(range(len(examples)))
    if rng:
        rng.shuffle(order)
        order.sort(key=lambda i: len(examples[i][0]) + rng.random() * 3)
    else:
        order.sort(key=lambda i: len(examples[i][0]))
    groups = [order[i:i + size] for i in range(0, len(order), size)]
    if rng:
        rng.shuffle(groups)
    for group in groups:
        letters = torch.zeros(len(group), max(len(examples[i][0]) for i in group), dtype=torch.long)
        for row, i in enumerate(group):
            letters[row, :len(examples[i][0])] = torch.tensor(examples[i][0])
        yield group, letters, torch.tensor([len(examples[i][0]) for i in group])


def greedy(log_probs, frames):
    best = log_probs.argmax(-1).tolist()
    out = []
    for row, length in zip(best, frames.tolist()):
        labels, previous = [], 0
        for label in row[:length]:
            if label != previous and label != 0:
                labels.append(label)
            previous = label
        out.append(labels)
    return out


def edit_distance(a, b):
    previous = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        current = [i]
        for j, y in enumerate(b, 1):
            current.append(min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + (x != y)))
        previous = current
    return previous[-1]


def strip_stress(phones):
    return [p.rstrip("012") for p in phones]


@torch.no_grad()
def evaluate(model, words, symbols, batch_size=512):
    """Word and phoneme error rates. A word counts as right when the output
    equals any of its dictionary pronunciations; the phoneme error is the edit
    distance to the closest one."""
    model.eval()
    device = next(model.parameters()).device
    examples = [(encode_word(word), []) for word, _ in words]
    totals = {"words": len(words), "wrong": 0, "wrong_no_stress": 0, "wrong_first": 0, "edits": 0, "edits_no_stress": 0,
              "phones": 0}
    samples = []
    for group, letters, lengths in batches(examples, batch_size):
        log_probs, frames = model(letters.to(device), lengths.to(device))
        for i, labels in zip(group, greedy(log_probs, frames)):
            word, variants = words[i]
            predicted = [symbols[label - 1] for label in labels]
            plain = strip_stress(predicted)
            distance, reference = min((edit_distance(predicted, v), v) for v in variants)
            totals["wrong"] += distance > 0
            totals["wrong_first"] += predicted != variants[0]
            totals["wrong_no_stress"] += all(plain != strip_stress(v) for v in variants)
            totals["edits"] += distance
            totals["edits_no_stress"] += min(edit_distance(plain, strip_stress(v)) for v in variants)
            totals["phones"] += len(reference)
            if distance and len(samples) < 60:
                samples.append({"word": word, "predicted": " ".join(predicted), "reference": " ".join(reference)})
    n, p = totals["words"], totals["phones"]
    return {"words": n, "word_error": totals["wrong"] / n, "word_error_first_variant": totals["wrong_first"] / n,
            "word_error_no_stress": totals["wrong_no_stress"] / n, "phoneme_error": totals["edits"] / p,
            "phoneme_error_no_stress": totals["edits_no_stress"] / p}, samples


def export(model, symbols, path, details):
    """Weights as plain float32 arrays plus a JSON description, so any runtime
    (NumPy here, later the portable core) can run the forward pass."""
    arrays = {name.replace(".", "__"): value.detach().cpu().numpy().astype(np.float32) for name, value in model.state_dict().items()}
    np.savez(str(path) + ".npz", **arrays)
    Path(str(path) + ".json").write_text(json.dumps({"letters": LETTERS, "symbols": symbols, **model.settings, **details}, indent=1))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cmudict", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--hidden", type=int, default=127)
    parser.add_argument("--convs", type=int, default=0)
    parser.add_argument("--kernel", type=int, default=3)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--slots", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=3e-3)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--average", type=float, default=0.999, help="Decay of the weight average (0 = off)")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--device", default="cpu", help="cpu or cuda; the model is small enough for either")
    parser.add_argument("--seed", type=int, default=20261114)
    args = parser.parse_args()

    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    splits, skipped = load_splits(args.cmudict)
    symbols = sorted({p for words in splits.values() for _, variants in words for v in variants for p in v})
    index = {symbol: i + 1 for i, symbol in enumerate(symbols)}
    train = [(encode_word(word), [index[p] for p in v]) for word, variants in splits["train"] for v in variants]
    # CTC needs one frame per phoneme plus one per repeated neighbour.
    fits = lambda letters, labels: len(labels) + sum(a == b for a, b in zip(labels, labels[1:])) <= len(letters) * args.slots
    dropped = sum(not fits(*example) for example in train)
    train = [example for example in train if fits(*example)]

    device = torch.device(args.device)
    model = G2P(len(symbols), args.width, args.hidden, args.convs, args.kernel, args.layers, args.slots, args.dropout).to(device)
    parameters = sum(p.numel() for p in model.parameters())
    if parameters > PARAMETER_BUDGET:
        raise SystemExit(f"{parameters} parameters exceed the budget of {PARAMETER_BUDGET}")
    args.output.mkdir(parents=True, exist_ok=True)
    log = (args.output / "log.jsonl").open("a", encoding="utf-8")

    def say(record):
        log.write(json.dumps(record) + "\n")
        log.flush()
        print(json.dumps(record), flush=True)

    say({"event": "start", "parameters": parameters, "budget": PARAMETER_BUDGET, "symbols": len(symbols),
         "words": {name: len(words) for name, words in splits.items()}, "train_pronunciations": len(train),
         "skipped_abbreviations": skipped, "dropped_too_long_for_ctc": dropped, "settings": vars(args) | {"cmudict": str(args.cmudict), "output": str(args.output)}})

    steps = args.epochs * math.ceil(len(train) / args.batch_size)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    schedule = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.learning_rate, total_steps=steps, pct_start=0.06)
    criterion = nn.CTCLoss(blank=0, zero_infinity=True)
    best, started, step = None, time.time(), 0
    # A running average of the weights usually generalises a little better than
    # the last step; each epoch the better of the two on validation is kept.
    average = {name: value.detach().clone() for name, value in model.named_parameters()} if args.average else None
    for epoch in range(1, args.epochs + 1):
        model.train()
        running, count = 0.0, 0
        for group, letters, lengths in batches(train, args.batch_size, rng):
            log_probs, frames = model(letters.to(device), lengths.to(device))
            targets = torch.tensor([label for i in group for label in train[i][1]], device=device)
            target_lengths = torch.tensor([len(train[i][1]) for i in group], device=device)
            loss = criterion(log_probs.transpose(0, 1), targets, frames, target_lengths)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            schedule.step()
            step += 1
            if average:
                decay = min(args.average, (1 + step) / (10 + step))
                with torch.no_grad():
                    for name, value in model.named_parameters():
                        average[name].mul_(decay).add_(value, alpha=1 - decay)
            running += loss.item() * len(group)
            count += len(group)
        metrics, _ = evaluate(model, splits["validation"], symbols)
        record = {"event": "epoch", "epoch": epoch, "loss": round(running / count, 4), "minutes": round((time.time() - started) / 60, 1),
                  **{"validation_" + key: round(value, 4) for key, value in metrics.items() if key != "words"}}
        if best is None or metrics["word_error"] < best:
            best = metrics["word_error"]
            torch.save(model.state_dict(), args.output / "best.pt")
            record["saved"] = "last"
        if average:
            live = {name: value.detach().clone() for name, value in model.named_parameters()}
            with torch.no_grad():
                for name, value in model.named_parameters():
                    value.copy_(average[name])
            averaged, _ = evaluate(model, splits["validation"], symbols)
            record["validation_word_error_averaged"] = round(averaged["word_error"], 4)
            if averaged["word_error"] < best:
                best = averaged["word_error"]
                torch.save(model.state_dict(), args.output / "best.pt")
                record["saved"] = "averaged"
            with torch.no_grad():
                for name, value in model.named_parameters():
                    value.copy_(live[name])
        say(record)

    model.load_state_dict(torch.load(args.output / "best.pt", weights_only=True, map_location=device))
    validation, _ = evaluate(model, splits["validation"], symbols)
    test, samples = evaluate(model, splits["test"], symbols)
    report = {"parameters": parameters, "budget": PARAMETER_BUDGET, "validation": validation, "test": test,
              "train_words": len(splits["train"]), "skipped_abbreviations": skipped, "error_samples": samples}
    (args.output / "report.json").write_text(json.dumps(report, indent=1))
    export(model, symbols, args.output / "g2p", {"parameters": parameters, "test_word_error": test["word_error"],
                                                 "test_phoneme_error": test["phoneme_error"]})
    say({"event": "done", "validation": validation, "test": test})


if __name__ == "__main__":
    main()
