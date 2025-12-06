import tiktoken 
import torch
from torch.utils.data import Dataset, DataLoader


with open("the-verdict.txt", "r",encoding="utf-8") as f:
    raw_text = f.read()

class GPTDatasetV1(Dataset):
    def __init__(self, txt, tokenizer, max_length, stride):
        self.input_ids = []
        self.target_ids = []

        #tokenize the entire text 
        token_ids = tokenizer.encode(txt, allowed_special={"<|endoftext|>"})

        #using sliding window to chunk the book into overlapping sequences of max_length
        for i in range(0,len(token_ids)-max_length, stride):
            input_chunk  = token_ids[i: i + max_length]
            target_chunk = token_ids[i+1: i + max_length + 1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self):
        return len(self.input_ids)
    
    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]

def create_dataloader_v1(txt, batch_size, max_length=256, 
                         stride=128, shuffle=True, drop_last=True,
                         num_workers=0):
        
        # Initialize the tokenizer
        tokenizer = tiktoken.get_encoding("gpt2")

        # Create dataset
        dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)

        # Create dataloader
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=drop_last,
            num_workers=num_workers
        )

        return dataloader
    
if __name__ =="__main__":   
        print("PyTorch version:", torch.__version__)
        dataloader = create_dataloader_v1(
            raw_text, batch_size=1, max_length=4, stride=1, shuffle=False
        )

        data_iter = iter(dataloader)
        first_batch = next(data_iter)
        print(first_batch)

        dataloader = create_dataloader_v1(raw_text,
                                          batch_size=8, 
                                          max_length=4, 
                                          stride=4, 
                                          shuffle=False)

        data_iter = iter(dataloader)
        inputs, targets = next(data_iter)
        print("Inputs:\n", inputs)
        print("\nTargets:\n", targets)