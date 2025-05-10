import unicodedata

######################################
###   General Functions
######################################

# -------------------------------------
# Retrieve statistic of pairs of string
# -------------------------------------

def get_stats(ids: list):
    stats = {} 
    for pairs in zip(ids,ids[1:]):
        stats[pairs] = stats.get(pairs,0) + 1 # Increase the value at "pairs" by 1 
    return(stats)

# -------------------------------------
# Transform all pairs of the str by idx
# -------------------------------------

def merge(ids: str, pair: tuple, idx: str):

    new_ids = [] # new list
    i = 0
    while i < len(ids):

        if i < len(ids) - 1 and ids[i] == pair[0] and ids[i+1] == pair[1]:
            new_ids.append(idx)
            i += 2
        else: 
            new_ids.append(ids[i])
            i += 1
    return(new_ids)

# -------------------------------------
# Escape control characters in a string
# -------------------------------------

def replace_control_characters(s: str) -> str:
    chars = []
    for ch in s:
        if unicodedata.category(ch)[0] != "C":
            chars.append(ch) # this character is ok
        else:
            chars.append(f"\\u{ord(ch):04x}") # escape
    return "".join(chars)

# -------------------------------------
# Render byte tokens as readable strings
# -------------------------------------

def render_token(t: bytes) -> str:
    # pretty print a token, escaping control characters
    s = t.decode('utf-8', errors='replace')
    s = replace_control_characters(s)
    return s


######################################
###   Tokenizer Architectures
######################################

class Tokenizer:
    """Base class for Tokenizers"""

    def __init__(self, length_list_token):
        # default: vocab size of 256 (all bytes), no merges, no patterns
        self.length_list_token = length_list_token
        self.merges = {} # (int, int) -> int
        self.pattern = "" # str
        self.special_tokens = {} # str -> int, e.g. {'<|endoftext|>': 100257}
        self.vocab = self._build_vocab() # int -> bytes

    def train(self, text, vocab_size, verbose=False):
        # Tokenizer can train a vocabulary of size vocab_size from text
        raise NotImplementedError

    def encode(self, text):
        # Tokenizer can encode a string into a list of integers
        raise NotImplementedError

    def decode(self, ids):
        # Tokenizer can decode a list of integers into a string
        raise NotImplementedError

    def _build_vocab(self):
        # Vocab is simply and deterministically derived from merges
        vocab = {idx: bytes([idx]) for idx in range(self.length_list_token)}
        for (p0, p1), idx in self.merges.items():
            vocab[idx] = vocab[p0] + vocab[p1]
        for special, idx in self.special_tokens.items():
            vocab[idx] = special.encode("utf-8")
        return vocab

    def save(self, file_prefix):
        """
        Saves two files: file_prefix.vocab and file_prefix.model
        This is inspired (but not equivalent to!) sentencepiece's model saving:
        - model file is the critical one, intended for load()
        - vocab file is just a pretty printed version for human inspection only
        """
        # write the model: to be used in load() later
        model_file = file_prefix + ".model"
        with open(model_file, 'w') as f:
            # write the version, pattern and merges, that's all that's needed
            f.write("minbpe v1\n")
            f.write(f"{self.pattern}\n")
            # write the special tokens, first the number of them, then each one
            f.write(f"{len(self.special_tokens)}\n")
            for special, idx in self.special_tokens.items():
                f.write(f"{special} {idx}\n")
            # the merges dict
            for idx1, idx2 in self.merges:
                f.write(f"{idx1} {idx2}\n")
        # write the vocab: for the human to look at
        vocab_file = file_prefix + ".vocab"
        inverted_merges = {idx: pair for pair, idx in self.merges.items()}
        with open(vocab_file, "w", encoding="utf-8") as f:
            for idx, token in self.vocab.items():
                # note: many tokens may be partial utf-8 sequences
                # and cannot be decoded into valid strings. Here we're using
                # errors='replace' to replace them with the replacement char �.
                # this also means that we couldn't possibly use .vocab in load()
                # because decoding in this way is a lossy operation!
                s = render_token(token)
                # find the children of this token, if any
                if idx in inverted_merges:
                    # if this token has children, render it nicely as a merge
                    idx0, idx1 = inverted_merges[idx]
                    s0 = render_token(self.vocab[idx0])
                    s1 = render_token(self.vocab[idx1])
                    f.write(f"[{s0}][{s1}] -> [{s}] {idx}\n")
                else:
                    # otherwise this is leaf token, just print it
                    # (this should just be the first 256 tokens, the bytes)
                    f.write(f"[{s}] {idx}\n")

    def load(self, model_file):
        """Inverse of save() but only for the model file"""
        assert model_file.endswith(".model")
        # read the model file
        merges = {}
        special_tokens = {}
        idx = self.length_list_token
        with open(model_file, 'r', encoding="utf-8") as f:
            # read the version
            version = f.readline().strip()
            assert version == "minbpe v1"
            # read the pattern
            self.pattern = f.readline().strip()
            # read the special tokens
            num_special = int(f.readline().strip())
            for _ in range(num_special):
                special, special_idx = f.readline().strip().split()
                special_tokens[special] = int(special_idx)
            # read the merges
            for line in f:
                idx1, idx2 = map(int, line.split())
                merges[(idx1, idx2)] = idx
                idx += 1
        self.merges = merges
        self.special_tokens = special_tokens
        self.vocab = self._build_vocab()


class BasicTokenizer(Tokenizer):
    """BPE Encoder"""
    def __init__(self, length_list_token = 256):
        super().__init__(length_list_token)

    def train(self, text, vocab_size, verbose = False):

        assert vocab_size >= self.length_list_token
        num_merges = vocab_size - self.length_list_token

        text_bytes = text.encode("utf-8")   # Encode into Byte type
        ids = list(map(int,text_bytes))     # Map to int               

        # Iteratively merge the most common pairs into tokens
        merges = {}
        vocab = {idx: bytes([idx]) for idx in range(self.length_list_token)}

        for i in range(num_merges):
            stats = get_stats(ids)                                 # Statistic of the most recent pairs of tokens
            most_rec_pair = (max(stats, key = stats.get))          # Retrieve the most present pair
            idx = self.length_list_token + i
            ids = merge(ids,most_rec_pair, idx)
            merges[most_rec_pair] = idx
            vocab[idx] = vocab[most_rec_pair[0]] + vocab[most_rec_pair[1]]
            
        self.merges = merges
        self.vocab = vocab
                
    def encode(self, text):
        # given a string text, return the token ids
        text_bytes = text.encode("utf-8") # raw bytes
        ids = list(text_bytes)            # list of integers
        while len(ids) >= 2:
            # find the pair with the lowest merge index
            stats = get_stats(ids)
            pair = min(stats, key=lambda p: self.merges.get(p, float("inf")))

            if pair not in self.merges:
                break # nothing else can be merged anymore
            # otherwise let's merge the best pair (lowest merge index)
            idx = self.merges[pair]
            ids = merge(ids, pair, idx)
        return ids

    def decode(self,ids):
        text_bytes = b"".join(self.vocab[idx] for idx in ids)
        text = text_bytes.decode("utf-8", errors = "replace")
        return(text)