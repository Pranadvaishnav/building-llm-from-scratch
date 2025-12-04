import tiktoken

tokenizer = tiktoken.get_encoding("gpt2")
with open("the-verdict.txt", "r",encoding="utf-8") as f:
    raw_text = f.read()

encoded_txt = tokenizer.encode(raw_text)
print(len(encoded_txt))
  

context_size = 10

x = encoded_txt[:context_size]
y = encoded_txt[1:context_size+1]

print(f"x: {x}")
print(f"y:       {y}")

for i in range (1, context_size+1):
    context = encoded_txt[:i]
    desired = encoded_txt[i]

    print(context,"------->", desired)

for i in range (1, context_size+1):
    context = encoded_txt[:i]
    desired = encoded_txt[i]

    print(tokenizer.decode(context),"------->", tokenizer.decode([desired]))