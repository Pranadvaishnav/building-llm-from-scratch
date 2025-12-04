import importlib.metadata
import tiktoken

print("tiktoken version:", importlib.metadata.version("tiktoken"))

tokenizer = tiktoken.get_encoding("gpt2")

text = (
    "Hello do u like tea? <|endoftext|> i dont ! neither do i drink any <|endoftext|>"
    "In the sunlit terraces "
    "of someunkownPlace"
)
integers = tokenizer.encode(text, allowed_special={"<|endoftext|>"})
print(integers)

strings = tokenizer.decode(integers)
print(strings)

print("lets understand this by an example: this is the example we will encode and decode     :Akwirw ier")
integers = tokenizer.encode("Akwirw ier")
print(integers)

strings = tokenizer.decode(integers)
print(strings)