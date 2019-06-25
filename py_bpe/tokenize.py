import re
import html
from typing import List, Dict, Tuple
from collections import Counter, defaultdict
from tqdm import tqdm
import pickle
from pathlib import Path
"""
TODO:   - setup.py
        - tests
        - parallelization
        - ulmfit preprocessing 
"""
class BpeTokenizer:
    def __init__(self):
        self.vocab_stoi = None
        self.vocab_itos = None

        # predefine tokens
        self.meta_tokens = dict(unk="<unk>", pad="<pad>", bos="<bos>", eos="<eos>", fld="<fld>")
        self.preprocess_tokens = dict(maj="<maj>", upp="<upp>", rep="<rep>", wrep="<wrep>",
                                     htag="<#tag>", ctag="<$tag>", atag="<@tag>")

        self.preprocess_args = dict(
            lowercase=True,
            spec_add_spaces=True,
            remove_useless_spaces=True,
            fix_html=True,
        )

        self.learn_bpe_args = dict(
            vocab_size=50_000,
            pairable_chars="a-zA-Z", # regex char set that can be merged in a BPE (`unpairable_chars` must be `None`)
            unpairable_chars=None, # regex char set that can't be merged in a BPE (`pairable_chars` must be `None`)
            unpairable_str="", # regex OR set that can't be in a BPE (will include `meta_tokens`, `preprocess_tokens`)

            required_chars = [], # list of chars to force include in vocab
            required_subwords = [], # list of subwords (e.g. morphemes such as `-er`) to force include in vocab
            required_words = [], # list of words to force include in vocab

            num_chars = -1, # number of chars from corpus (in order of freq) to force include in vocab (-1:all, 0:none)
            num_words = 0, # number of words from corpus (in order of freq) to force include in vocab (-1: all, 0: none)

            max_bpe_size = 0, # max number of chars per BPE (0: unlimited)
            bpe_per_merge = 10, # merge multiple pairs per iteration for speed
        )

    @classmethod
    def from_corpus(cls, corpus: str, save_path: Path,
                    preprocess_args: Dict = None, learn_bpe_args: Dict = None):

        bpet = cls()
        # TODO: clean up constructor api, find altenatives to arg dicts
        if preprocess_args is None:
            preprocess_args = {}
        if learn_bpe_args is None:
            learn_bpe_args = {}
        for k,v in preprocess_args.items():
            bpet.preprocess_args[k] = v
        for k,v in learn_bpe_args.items():
            bpet.learn_bpe_args[k] = v

        bpet._learn_bpe_vocab(corpus)
        bpet.save(save_path)
        return bpet

    def _learn_bpe_vocab(self, corpus: str):
        args = self.learn_bpe_args

        corpus = self._preprocess(corpus)
        word_counts, unpairable_counts = self._count_words(corpus, args)
        vocab_stoi, vocab_itos = self._init_vocab(corpus, word_counts, args)

        word_encodings = {word: [c for c in word] for word in word_counts.keys()}

        num_bpe = args['vocab_size'] - len(vocab_itos)
        num_merges = num_bpe//args['bpe_per_merge']
        for _ in tqdm(range(num_merges)):
            # generate new bytepair frequencies
            bp_counts = defaultdict(int)
            bp_words = defaultdict(set)
            for word, encodings in word_encodings.items():
                for bytepair in zip(encodings[:-1], encodings[1:]):
                    bp = "".join(bytepair)
                    if bp not in vocab_stoi and (len(bp) <= args['max_bpe_size'] or not args['max_bpe_size']):
                        bp = " ".join(bytepair) # space to facilitate word encodings update below
                        bp_counts[bp] += word_counts[word]
                        bp_words[bp].add(word)

            if len(bp_counts) == 0:
                break

            # update stoi/itos and word_encodings
            best_bp = sorted(bp_counts, key=bp_counts.get, reverse=True)[:args['bpe_per_merge']]
            for bp in best_bp:
                merged_bp = bp.replace(" ", "")
                vocab_itos += [merged_bp]
                vocab_stoi[merged_bp] = len(vocab_itos) - 1
                for word in bp_words[bp]:
                    word_encodings[word] = " ".join(word_encodings[word]).replace(bp, merged_bp).split(" ")

        self.vocab_stoi = vocab_stoi
        self.vocab_itos = vocab_itos

    def _preprocess(self, text: str) -> str:
        """Preprocess corpus for learning BPEs.

        Note: this preprocessing is not reversible.

        Adapted from: https://docs.fast.ai/text.transform.html#Tokenizer.process_text

        """
        if self.preprocess_args['lowercase']:
            text = text.lower()

        if self.preprocess_args['spec_add_spaces']:
            text = re.sub(r'([/#\n])', r' \1 ', text)

        if self.preprocess_args['remove_useless_spaces']:
            text = re.sub(' {2,}', ' ', text)

        if self.preprocess_args['fix_html']:
            UNK = self.meta_tokens['unk']
            re1 = re.compile(r'  +')
            text = text.replace('#39;', "'").replace('amp;', '&').replace('#146;', "'").replace(
                'nbsp;', ' ').replace('#36;', '$').replace('\\n', "\n").replace('quot;', "'").replace(
                '<br />', "\n").replace('\\"', '"').replace('<unk>', UNK).replace(' @.@ ', '.').replace(
                ' @-@ ', '-').replace(' @,@ ', ',').replace('\\', ' \\ ')
            return re1.sub(' ', html.unescape(text))

        # TODO: char repetition

        # TODO: word repetition

        # TODO: capitalization (first upper, all upper, some upper)

        return text

    def _count_words(self, corpus: str, args: Dict) -> Tuple[Counter, Counter]:
        """Obtains corpus word counts used to learn BPE.

        A word is a continuous sequence of "pairable" characters.
        Tokens made from unpairable characters are also counted in case we want
        to include them by frequency instead of all by default.

        :param corpus: (str) text to learn BPEs from
        :param pairable_chars: (str) regex char set e.g. "a-z" ([ ] not included)
        :param unpairable_chars: (str) regex exclusion char set e.g. "0-9.?!'amp;'" ([ ] and ^ not included)
        :param unpairable_str: (str) regex string set e.g. "(dog|cat|axolotl)"
        :return word_counts: (Counter) count of words
        """
        pairable_chars = args['pairable_chars']
        unpairable_chars = args['unpairable_chars']
        unpairable_str = args['unpairable_str']

        if pairable_chars and unpairable_chars:
            raise ValueError("Only one of `pairable_chars` or `unpairable_chars` can be provided")

        unpairable_counts = Counter(re.findall("|".join(unpairable_str), corpus))
        corpus = re.sub("|".join(unpairable_str),"", corpus)

        # count words (pairabled chars) and unpairable chars
        if pairable_chars:
            word_counts = Counter(re.findall(f"[{pairable_chars}]+", corpus))
            unpairable_counts.update(re.findall(f"[^{pairable_chars}]", corpus))
        elif unpairable_chars:
            word_counts = Counter(re.findall(f"[^{unpairable_chars}]+", corpus))
            unpairable_counts.update(re.findall(f"[{unpairable_chars}]", corpus))
        else:
            word_counts = Counter(re.findall("[a-zA-Z]+", corpus))
            unpairable_counts.update(re.findall("^[a-zA-Z]", corpus))

        return word_counts, unpairable_counts

    def _init_vocab(self, corpus: str, word_counts: Counter, args: Dict) -> Tuple[Dict[str, int], List[str]]:
        """Initialize vocabulary based on predefined tokens (use temporary lookup set to avoid duplicates)

        """
        tmp_vocab_itos = [v for v in self.meta_tokens.values()] + [v for v in self.preprocess_tokens.values()] + \
                     args['required_chars'] + args['required_subwords'] + args['required_words']

        # update vocabulary with corpus words and chars
        if args['num_chars']:
            char_counts = Counter(corpus)
            if args['num_chars'] == -1:
                args['num_chars'] = len(char_counts)
            tmp_vocab_itos += sorted(char_counts, key=char_counts.get, reverse=True)[:args['num_chars']]
        if args['num_words']:
            if args['num_words'] == -1:
                args['num_words'] = len(word_counts)
            tmp_vocab_itos += sorted(word_counts, key=word_counts.get, reverse=True)[:args['num_words']]

        tmp_lookup = set()  # a temporary lookup set
        vocab_itos = [x for x in tmp_vocab_itos if x not in tmp_lookup and tmp_lookup.add(x) is None]
        vocab_stoi = {s:i for i,s in enumerate(vocab_itos)}
        unk_i = vocab_stoi[self.meta_tokens['unk']]
        vocab_stoi = defaultdict(lambda: unk_i, vocab_stoi)

        return vocab_stoi, vocab_itos

    def tokenize(self, text: str) -> List[str]:
        tokens = []
        token = None

        maj_flag = False
        upp_flag = False

        for c in text:
            # expand previous token by one character or append previous token to tokens
            if token is not None:
                new_token = token + c.lower()
                if (new_token not in self.vocab_stoi) or \
                        (len(token)>1 and c.isupper() and maj_flag) or \
                        (c.isupper() and not maj_flag and not upp_flag) or \
                        (c.islower() and upp_flag):
                    if maj_flag:
                        tokens.append(self.preprocess_tokens['maj'])
                        maj_flag = False
                    elif upp_flag:
                        tokens.append(self.preprocess_tokens['upp'])
                        upp_flag = False
                    tokens.append(token)
                    token = None
                else:
                    token = new_token
                    if c.isupper() and maj_flag:
                        upp_flag = True
                        maj_flag = False

            # handle unpairable tokens
            if c.lower() not in self.vocab_stoi:
                tokens.append(c)

            # begin new token
            elif token is None:
                if c.isupper():
                    maj_flag = True
                    token = c.lower()
                else:
                    token = c
        if token:
            tokens.append(token)

        return tokens

    def detokenize(self, tokens: List[str]) -> str:
        for i,token in enumerate(tokens):
            if token == self.preprocess_tokens['maj']:
                tokens[i+1] = tokens[i+1].title() # TODO: fix
            if token == self.preprocess_tokens['upp']:
                tokens[i+1] = tokens[i+1].upper()
        tokens = [token for token in tokens if token not in self.preprocess_tokens.values()]
        text = "".join(tokens)
        return text

    def encode(self, text: str) -> List[int]:
        return [self.vocab_stoi[s] for s in self.tokenize(text)]

    def decode(self, encodings: List[int]) -> str:
        return self.detokenize([self.vocab_itos[i] for i in encodings])

    def save(self, path: Path):
        pickle.dump(self.vocab_itos, path.open('wb'))

    @classmethod
    def load(cls, path: Path):
        bpet = cls()
        itos = pickle.load(path.open('rb'))
        stoi = {s: i for i,s in enumerate(itos)}
        bpet.vocab_itos = itos
        bpet.vocab_stoi = stoi
        return bpet