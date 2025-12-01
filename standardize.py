with open("the-verdict.txt", "r",encoding="utf-8") as f:
    raw_text = f.read()

print("total number of charahter:", len(raw_text))
print(raw_text[:99])

import re

text = "hello, world. This, is a text."
result = re.split(r'([,.]|\s)',text)
print(result) 
print("1st")
result =[item for item in result if item.strip()]
print(result)
print("2st")

text = "hello, world. Is this-- a text?."
result = re.split(r'([,.:;?_!"()\']|--|\s)',text)
result =[item.strip() for item in result if item.strip()]
print(result)
print("3st")


preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', raw_text)
preprocessed = [item.strip() for item in preprocessed if item.strip()]
print(preprocessed[:30])
print("4st")
 
print(len(preprocessed))
print("5th")
all_words = sorted(set(preprocessed))
vocab_size = len(all_words)

print(vocab_size)
print("6th")

vocab = {token: integer for integer, token in enumerate(all_words)}
for i,item in enumerate(vocab.items()):
    print(item)
    if i>= 50:
        break