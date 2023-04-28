from tokenizers import Tokenizer

tokenizer = Tokenizer.from_file("tokenizer.json")

tokenizer.add_special_tokens(["<|sep|>", "<|sys|>", "<|inp|>", "<|thk|>", "<|out|>"])

tokenizer.save("tokenizer_tuned_chat.json")
