# dizione

dizione turns English text into phonemes. It is the front end of the
[tiny-voca](https://github.com/n4ze3m/tiny-voca) text-to-speech model.

## Features

- **Reads text like a person**: numbers, dates, money, units, abbreviations and chat words.
- **Homographs**: picks the right "read" or "live" from the words around it.
- **Unknown words**: a small model guesses words that are not in the dictionary.
- **Light**: NumPy only, 5.4 MB of data, about 4,400 sentences per second.

## Install

```bash
pip install git+https://github.com/n4ze3m/dizione
hf download n4ze3m/tiny-voca --include "phonemizer/*" --local-dir tiny-voca
export DIZIONE_DATA=tiny-voca/phonemizer
```

## Usage

```bash
python -m dizione --words "Dr. Patel paid $12.99 on 3/14."
```

```python
import dizione

result = dizione.phonemize("I read the book you read yesterday.")
print(result)        # <bos> aɪ <sp> ɹ ˈiː d <sp> ð ə <sp> b ˈʊ k ...
result.ids           # ids into dizione.symbols.SYMBOLS
result.stress        # 0 none, 1 primary, 2 secondary
```

Add your own words:

```python
from dizione.engine import Phonemizer

p = Phonemizer(overrides={"kubernetes": "k ˌuː b ɚ n ˈɛ ɾ iː z"})
```

Or write a word inline: `Say {HH AH0 L OW1} to everyone.`

## Build the data

```bash
pip install -e ".[build]"
bash scripts/build.sh /path/to/work
export DIZIONE_DATA=/path/to/work/built
```

This takes about 90 minutes on four CPU cores.

## Tests

```bash
PYTHONPATH=src python -m unittest discover -s tests
```

## Limits

- American English only.
- About one unknown word in three has an error. Foreign names are the worst.
- Python only for now.

How it works, the scores and the known errors are in [docs/design.md](docs/design.md).

## License

Apache-2.0. The built data comes from CMUdict and the Wikipedia Homograph Data;
their license files are copied next to it.
