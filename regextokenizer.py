import regex as re
import unicodedata

GPT4_SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""

class RegexTokenizer():
    def __init__(self, regex_pattern = GPT4_SPLIT_PATTERN):
        self.regex_pattern = regex_pattern
        self.merges = {}
        self.special_tokens = {}
        self.inverse_special_tokens = {}
        self.compiled_pattern = re.compile(self.regex_pattern)
        self.vocab = self._build_vocab()
        
    def _get_stats(self,ids: list[int], counts: dict[tuple[int, int], int] | None = None) -> dict[tuple[int, int], int]:
        """ 
        This get token as list of int and optionally pairs , this return pairs of dict 
        with word and its next word as tuple as key of pairs dict with its values as number 
        of times that pair appeared.
        """
        counts = {} if counts is None else counts
        for pair in zip(ids, ids[1:]):
            counts[pair] = counts.get(pair, 0) + 1
        return counts

    def _merges(self,ids: list[int], pair: tuple[int, int], idx: int) -> list[int]:
        """
        This fn return new list of token , this merges replace pair with idx in
        ids (tokens) and return 
        """

        new_ids = []
        i = 0
        while i < len(ids) - 1:
            if ids[i] == pair[0] and ids[i+1] == pair[1]:
                new_ids.append(idx) 
                i += 2
            else:
                new_ids.append(ids[i])
                i += 1
        if i == len(ids) - 1:
            new_ids.append(ids[i])
        return new_ids

    def _build_vocab(self):
        """
            This is private fn which builds vocab from start when we load this fn from somewhere,
            This fn exits so whenever you load a tokenizer , it dont have empty vocab
        """
        vocab = {idx: bytes([idx]) for idx in range(256)}
        for (p0, p1), idx in self.merges.items():
            vocab[idx] = vocab[p0] + vocab[p1]
        
        for special, idx in self.special_tokens.items():
            vocab[idx] = special.encode('utf-8')

        return vocab

    def _encode_chunk(self, text_bytes) -> list[int]:
        ids = list(text_bytes)

        while len(ids) >= 2:
            stats = self._get_stats(ids)
            pair = min(stats, key=lambda p: self.merges.get(p, float("inf")))

            if pair not in self.merges:
                break
            
            idx = self.merges[pair]
            ids = self._merges(ids, pair, idx)

        return ids

    def _replace_control_char(self, s : str) -> str:
        char = []
        for ch in s:
            if unicodedata.category(ch).startswith('C'):
                char.append(f"\\u{ord(ch):04x}")
            else:
                char.append(ch)

        return "".join(char)

    def _render_token(self, t : bytes) -> str:
        s = t.decode('utf-8', errors='replace')
        s = self._replace_control_char(s)
        return s


    def register_special_tokens(self, special_tokens_dict: dict[str, int]):
        """
        Registers custom special tokens.
        Example: tokenizer.register_special_tokens({"<|endoftext|>": 10001})
        """
        for token, idx in special_tokens_dict.items():
            self.special_tokens[token] = idx
            self.inverse_special_tokens[idx] = token
            
        self.vocab = self._build_vocab()


    def train(self, vocab_size: int, text: str, debug:bool = False) -> None:
        assert vocab_size >= 256, "PLease give vocab size greater than 255"
        num_merges = vocab_size - 256

        self.vocab = {idx: bytes([idx]) for idx in range(256)}
        text_chunks = re.findall(self.compiled_pattern, text)

        ids = [list(ch.encode("utf-8")) for ch in text_chunks]

        merges = {}

        for i in range(num_merges):
            stats = {}
            for chunk_ids in ids:
                self._get_stats(chunk_ids, stats)

            if not stats:
                break

            pair = max(stats, key=stats.get)
            idx = 256 + i

            ids = [self._merges(chunk_ids, pair, idx) for chunk_ids in ids]

            merges[pair] = idx
            self.vocab[idx] = self.vocab[pair[0]] + self.vocab[pair[1]]

            if debug:
                print(f"merge {i+1}/{num_merges}: {pair} -> {idx} ({self._render_token(self.vocab[idx])}) had {stats[pair]} occurrences")

        self.merges = merges 
        self.vocab = self._build_vocab()


    def decode(self, ids: list[int]) -> str:
        part_bytes = []

        for idx in ids:
            if idx in self.vocab:
                part_bytes.append(self.vocab[idx])
            elif idx in self.inverse_special_tokens:
                part_bytes.append(self.inverse_special_tokens[idx].encode("utf-8"))
            else:
                raise ValueError("I dont think this is in our ctrl")

        text_bytes = b"".join(part_bytes)
        text = text_bytes.decode("utf-8", errors="replace")

        return text

    def encode_ordinary(self, text: str) -> list[int]:
        """
            encoding that ignore special token like your crush!!!
        """
        text_chunks = re.findall(self.compiled_pattern, text)
        ids = []

        for chunk in text_chunks:
            chunk_bytes = chunk.encode("utf-8")
            chunk_ids = self._encode_chunk(chunk_bytes)
            ids.extend(chunk_ids)

        return ids

    def encode(self, text: str, allowed_special = "none_raise"):
        special = None

        if allowed_special == "all":
            special = self.special_tokens
        elif allowed_special == "none":
            special = {}
        elif allowed_special == "none_raise":
            special = {}
            assert all(token not in text for token in self.special_tokens)
        elif isinstance(allowed_special, set):
            special = {k: v for k, v in self.special_tokens.items() if k in allowed_special}
        else:
            raise ValueError(f"allowed_special={allowed_special} not understood")

        if not special:
            return self.encode_ordinary(text)

        special_pattern = "(" + "|".join(re.escape(k) for k in special) + ")"
        special_chunks = re.split(special_pattern, text)

        ids = []

        for char in special_chunks:
            if char in special:
                ids.append(special[char])
            else:
                ids.extend(self.encode_ordinary(char))

        return ids

    def save(self, file_prefix):
        """
        Saves two files: file_prefix.vocab and file_prefix.model
        """
        model_file = file_prefix + ".model"
        
        with open(model_file, 'w', encoding="utf-8") as f:
            f.write("minbpe v1\n")

            f.write(f"{self.regex_pattern}\n")

            f.write(f"{len(self.special_tokens)}\n")
            for special, idx in self.special_tokens.items():
                f.write(f"{special} {idx}\n")

            for idx1, idx2 in self.merges.items():
                f.write(f"{idx1} {idx2}\n")

        vocab_file = file_prefix + '.vocab'
        inverted_merges = {idx: pair for pair, idx in self.merges.items()}

        with open(vocab_file, 'w', encoding="utf-8") as f:
            for idx, token in self.vocab.items():
                s = self._render_token(token)

                if idx in inverted_merges:
                    idx0, idx1 = inverted_merges[idx]
                    s0 = self._render_token(self.vocab[idx0])
                    s1 = self._render_token(self.vocab[idx1])
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
            self.regex_pattern = f.readline().strip()
            num_special = int(f.readline().strip())

            for _ in range(num_special):
                parts = f.readline().strip().split()
                if parts:
                    special, special_idx = parts[0], parts[1]
                    special_tokens[special] = int(special_idx)

            for line in f:
                parts = line.split()
                if len(parts) == 2:
                    idx1, idx2 = map(int, parts)
                    merges[(idx1, idx2)] = idx
                    idx += 1

        self.merges = merges
        self.special_tokens = special_tokens
        self.vocab = self._build_vocab()
