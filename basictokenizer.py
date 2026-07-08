import unicodedata

def get_stats(token_ids: list[int], pairs: dict[tuple, int] | None = None) -> dict:
    """ 
        This get token as list of int and optionally pairs , this return pairs of dict 
        with word and its next word as tuple as key of pairs dict with its values as number 
        of times that pair appeared.
    """
    pairs = {} if pairs is None else pairs
    for pair in zip(token_ids, token_ids[1:]):
        pairs[pair] = counts.get(pair, 0) + 1

    return pairs

def merges(ids: list[int], pair: tuple[int, int], idx: int) -> list[int]:
    new_ids = []
    i = 0
    while len(ids) > i:
        if ids[i] == pair[0] and i < len(ids) - 1 and ids[i+1] == pair[1]:
            new_ids.append(idx)
            i += 2
        else:
            new_ids.append(ids[i])
            i += 1
    return new_ids

def replace_control_char(s : str) -> str:
    char = []
    for ch in s:
        if unicodedata.category(char).startswith('C'):
            char.append(f"\\u{ord(ch):04x}")
        else:
            char.append(ch)

    return "".join(char)

def render_token(t : bytes) -> str:
    s = t.decode('utf-8', errors='replace')
    s = replace_control_characters(s)
    return s

class Tokenizer():
    def __init__(self):
        self.merges = {}
        self.vocab = {}
        self.pattern = ""
        self.special_tokens = {}

    def train(self, text, vocab_size, debug=False):
        assert vocab_size >= 256
        num_merges = vocab_size - 256
        text_bytes = text.encode("utf-8") 
        ids = list(text_bytes)

        merges = {}
        vocab = {idx: bytes([idx]) for idx in range(256)}
        for i in range(num_merges):

            stats = get_stats(ids)
            if not stats:
                break

            pair = max(stats, key=stats.get)

            idx = 256 + i
            ids = merge(ids, pair, idx)

            merges[pair] = idx
            vocab[idx] = vocab[pair[0]] + vocab[pair[1]]

            if debug:
                print(f"merge {i+1}/{num_merges}: {pair} -> {idx} ({vocab[idx]}) had {stats[pair]} occurrences")

        self.merges = merges
        self.vocab = vocab 

    def encode(self, text):
        token = text.encode("utf-8")
        ids = list(map(int,token))
        while len(ids) >= 2:
            stats = get_stats(ids)
            pair = min(stats, key=lambda p: self.merges.get(p, float("inf")))
            if pair not in self.merges:
                break 
            idx = self.merges[pair]
            ids = merge(ids, pair, idx)
        return ids

    def decode(self, ids):
        text_bytes = b"".join(self.vocab[idx] for idx in ids)
        text = text_bytes.decode("utf-8", errors="replace")
        return text

    def save(self, file_prefix):
        """
        Saves two files: file_prefix.vocab and file_prefix.model
        """
        model_file = file_prefix + ".model"
        with open(model_file, 'w') as f:
            f.write("minbpe v1\n")
            f.write(f"{self.pattern}\n")
            f.write(f"{len(self.special_tokens)}\n")
            for special, idx in self.special_tokens.items():
                f.write(f"{special} {idx}\n")
            for idx1, idx2 in self.merges:
                f.write(f"{idx1} {idx2}\n")
        vocab_file = file_prefix + ".vocab"
        inverted_merges = {idx: pair for pair, idx in self.merges.items()}
        with open(vocab_file, "w", encoding="utf-8") as f:
            for idx, token in self.vocab.items():
                s = render_token(token)
                if idx in inverted_merges:
                    idx0, idx1 = inverted_merges[idx]
                    s0 = render_token(self.vocab[idx0])
                    s1 = render_token(self.vocab[idx1])
                    f.write(f"[{s0}][{s1}] -> [{s}] {idx}\n")
                else:
                    f.write(f"[{s}] {idx}\n")

    def load(self, model_file):
        assert model_file.endswith(".model")
        merges = {}
        special_tokens = {}
        idx = 256
        with open(model_file, 'r', encoding="utf-8") as f:
            version = f.readline().strip()
            assert version == "minbpe v1"
            self.pattern = f.readline().strip()
            num_special = int(f.readline().strip())
            for _ in range(num_special):
                special, special_idx = f.readline().strip().split()
                special_tokens[special] = int(special_idx)
            for line in f:
                idx1, idx2 = map(int, line.split())
                merges[(idx1, idx2)] = idx
                idx += 1
        self.merges = merges
        self.special_tokens = special_tokens
        self.vocab = self._build_vocab()