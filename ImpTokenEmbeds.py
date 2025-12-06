import torch
from impDataLoader import GPTDatasetV1
from impDataLoader import create_dataloader_v1
from impDataLoader import raw_text
input_ids = torch.tensor([2,3,5,1])
vocab_size = 50257
output_dim = 768

torch.manual_seed(42)
embedding_layer = torch.nn.Embedding(vocab_size, output_dim)


print(embedding_layer(input_ids))

max_length =4
#here we will assign ids to token
dataloader = create_dataloader_v1(raw_text,
                                          batch_size=8, 
                                          max_length=4, 
                                          stride=4, 
                                          shuffle=False)

data_iter = iter(dataloader)
inputs, targets= next(data_iter)

print("Token IDs:\n", inputs)
print("\nInputs shape:\n", inputs.shape)

#turning them into vectors 
token_embedding = embedding_layer(inputs)
print(token_embedding.shape)

#now will assign them positional vectors as well , 
#it will have same no of dimentions but vocab will be of context size
# |------------------------->dimentions
# |
# |
# |
# |
# v
#context size

context_length = max_length
pos_embedding_layer =torch.nn.Embedding(context_length, output_dim)

#arrangeing it in the form of mattrix of maxlength x dimensions
pos_embeddings = pos_embedding_layer(torch.arange(max_length))
print ("pos")
print(pos_embedding_layer)

#now we need to add these pos.vectors to uk the main embedded vectors 
#we have created the tokens of context size cuz for each new batch it will remain same 
#so we have to define same positions only so will add same vectors to each batch 
# so for 1st position we have 768 dimensional vector and same for 2nd 3rd and 4rth
#the 4x768 mattrix will be converted to 8x4x768 mattrix by dublicating the values
input_embedding = token_embedding + pos_embeddings
print(input_embedding.shape)
