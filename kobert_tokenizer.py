""" kobert tokenizer (fast version) 

vocab from https://github.com/SKTBrain/KoBERT
"""

# 3rd-party
from transformers.tokenization_utils_fast import PreTrainedTokenizerFast
from SentencePieceBPEBertTokenizer import SentencePieceBPEBertTokenizer


class KoBertTokenizerFast(PreTrainedTokenizerFast):
    """
    Constructs a "Fast" GPT-2 BPE tokenizer (backed by HuggingFace's `tokenizers` library).
    Peculiarities:
    - Byte-level Byte-Pair-Encoding
    - Requires a space to start the input string => the encoding methods should be called with the
      ``add_prefix_space`` flag set to ``True``.
      Otherwise, this tokenizer ``encode`` and ``decode`` method will not conserve
      the absence of a space at the beginning of a string:
    ::
        tokenizer.decode(tokenizer.encode("Hello")) = " Hello"
    This tokenizer inherits from :class:`~transformers.PreTrainedTokenizerFast` which contains most of the methods. Users
    should refer to the superclass for more information regarding methods.
    Args:
        vocab_file (:obj:`str`):
            Path to the vocabulary file.
        merges_file (:obj:`str`):
            Path to the merges file.
        errors (:obj:`str`, `optional`, defaults to "replace"):
            Paradigm to follow when decoding bytes to UTF-8. See `bytes.decode
            <https://docs.python.org/3/library/stdtypes.html#bytes.decode>`__ for more information.
        unk_token (:obj:`string`, `optional`, defaults to `<|endoftext|>`):
            The unknown token. A token that is not in the vocabulary cannot be converted to an ID and is set to be this
            token instead.
        bos_token (:obj:`string`, `optional`, defaults to `<|endoftext|>`):
            The beginning of sequence token.
        eos_token (:obj:`string`, `optional`, defaults to `<|endoftext|>`):
            The end of sequence token.
        add_prefix_space (:obj:`bool`, `optional`, defaults to `False`):
            Whether to add a leading space to the first word.
            This allows to treat the leading word just as any other word.
            (GPT2 tokenizer detect beginning of words by the preceeding space)
        trim_offsets (:obj:`bool`, `optional`, defaults to `True`):
            Whether the post processing step should trim offsets to avoid including whitespaces.
    """

    vocab_files_names = {
        "vocab_file": "vocab.json",
        "merges_file": "merges.txt",
        "special_tokens_map_file": "special_tokens_map.json",
        "added_tokens_file": "added_tokens.json"
    }

    def __init__(
        self,
        vocab_file,
        merges_file,
        unk_token="[UNK]",
        sep_token="[SEP]",
        pad_token="[PAD]",
        cls_token="[CLS]",
        mask_token="[MASK]",
        do_lower_case=False,
        remove_space=True,
        keep_accents=False,
        add_prefix_space=False,
        **kwargs
    ):
        tokenizer = SentencePieceBPEBertTokenizer(
                vocab_file=vocab_file,
                merges_file=merges_file,
                add_prefix_space=add_prefix_space,
                unk_token=unk_token
            )

        super().__init__(
            tokenizer,
            unk_token=unk_token,
            sep_token=sep_token,
            pad_token=pad_token,
            cls_token=cls_token,
            mask_token=mask_token,
            **kwargs,
        )

    @classmethod
    def from_pretrained(cls, *inputs, **kwargs):
        return cls._from_pretrained(*inputs, **kwargs)
