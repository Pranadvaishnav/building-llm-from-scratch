with open("the-verdict.txt", "r",encoding="utf-8") as f:
    raw_text = f.read()

import re

preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', raw_text)
preprocessed = [item.strip() for item in preprocessed if item.strip()]

all_words = sorted(set(preprocessed))
vocab_size = len(all_words)

vocab = {token: integer for integer, token in enumerate(all_words)}
 
class SimpleTokenizerV1:
    def __init__(self, vocab):
        self.str_to_int = vocab
        self.int_to_str = {i:s for s,i in vocab.items()}

    def encode(self, text): 
        preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', text)
        preprocessed = [item.strip() for item in preprocessed if item.strip()]

        ids = [self.str_to_int[s] for s in preprocessed]
        return ids
    def decode(self, ids):
        text = " ".join([self.int_to_str[i] for i in ids])
        #replace spaces before the specified punctuations
        text = re.sub(r'\s+([,.?!"()\'])', r'\1',text)
        return text
    

tokenizer = SimpleTokenizerV1(vocab)

text = """"It's the last he painted, you know,"
            Mrs. Gisburn said with pardonable pride."""
ids = tokenizer.encode(text)
print(ids)

decoded = tokenizer.decode(ids)
print(decoded)

all_tokens = sorted(list(set(preprocessed)))
all_tokens.extend (["<|endoftext|>","<|unk|>"])

vocab = {token: integer for integer, token in enumerate(all_tokens)}
print(len(vocab.items()))

for i,item in enumerate(list(vocab.items())[-5:]):
    print(item)

class SimpleTokenizerV2:
    def __init__(self, vocab):
        self.str_to_int = vocab
        self.int_to_str = {i:s for s,i in vocab.items()}

    def encode(self, text): 
        preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', text)
        preprocessed = [item.strip() for item in preprocessed if item.strip()]
        preprocessed = [
            item if item in self.str_to_int
            else "<|unk|>" for item in preprocessed
            ]
        ids = [self.str_to_int[s] for s in preprocessed]
        return ids
    
    def decode(self, ids):
        text = " ".join([self.int_to_str[i] for i in ids])
        #replace spaces before the specified punctuations
        text = re.sub(r'\s+([,.?!"()\'])', r'\1',text)
        return text
    

tokenizer = SimpleTokenizerV2(vocab)
text1 = "hello, do you like tea?"
text2 = "in the sunlit terraces of the palace."

text = " <|endoftext|> ".join((text1, text2))

print(text)

encoded = tokenizer.encode(text)
print(encoded)

decoded = tokenizer.decode(encoded)
print(decoded)