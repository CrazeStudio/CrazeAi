from tokenizers import Tokenizer


def load_tokenizer(path):
    return Tokenizer.from_file(str(path))
