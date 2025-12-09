# %%
import torch
from impDataLoader import GPTDatasetV1
from impDataLoader import create_dataloader_v1
from impDataLoader import raw_text
from ImpTokenEmbeds import input_embedding

inputs = input_embedding
#will now find the context vector 
#the token u will be looking at to find the context vector is called query 
query = inputs[1]

attn_scores_2 = torch.empty(inputs.shape[0],4)#initializing empty tensor to store the attention score
for i, x_i in enumerate(inputs):#looping over the inputs
    attn_scores_2[i] = torch.sum(x_i * query,dim=1)#taking the dot product between every input and query vector
#first element of attention score tensor is dot product between 1st input vector and query vector and so on.........
print("this", attn_scores_2)
#next step is to normalise these scores so that they add up to 1 so that we can express them in percentages and for maintaining training stability.
#simplest way to make all the attentions scores to 1 is to divide each score by the sum of all scores
#we will be emplimenting softmax function for this purpose where we will be exponentiating each score to the power of e and then dividing it by the sum of all exponentiated scores
#there are two types naive and another is stable softmax
def softmax_naive(x):
     return torch.exp(x) / torch.exp(x).sum(dim=0)

attn_weights_2_naive = softmax_naive(attn_scores_2)
print("attention weights (naive):",attn_weights_2_naive)
print("sum of attention weights (naive):", attn_weights_2_naive.sum())
#the output that we would have got from this would have been ok to deal with if our attention scores were small but as they are large then exponentiating made
#even larger ... which went till infinity and again after dividing by sum of all scores (again infinity) we got nan values i.e.(not a number) which useless for us
#and hence we will be using stable softmax function i.e. pytorch version
attn_weights_2 = torch.softmax(attn_scores_2, dim=0)
print("attention weights (stable):",attn_weights_2)
print("sum of attention weights (stable):", attn_weights_2.sum())
print(attn_weights_2 @ inputs)

inputs= torch.tensor(
   [[0.43, 0.15, 0.89], # Your     (x^1)
   [0.55, 0.87, 0.66], # journey  (x^2)
   [0.57, 0.85, 0.64], # starts   (x^3)
   [0.22, 0.58, 0.33], # with     (x^4)
   [0.77, 0.25, 0.10], # one      (x^5)
   [0.05, 0.80, 0.55]] # step     (x^6)
   )  

x_2 = inputs[1] #A
d_in = inputs.shape[1] #B
d_out = 2 #C    

torch.manual_seed(123)
W_query = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)
W_key = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)
W_value = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)

print(W_query)
print(W_key)
print(W_value)

query_2 = x_2 @ W_query
key_2 = x_2 @ W_key
value_2 = x_2 @ W_value
print(query_2)

keys = inputs @ W_key
values = inputs @ W_value
queries = inputs @ W_query
print("keys.shape:", keys.shape)

print("values.shape:", values.shape)

print("queries.shape:", queries.shape)

keys_2 = keys[1] #A
attn_score_22 = query_2.dot(keys_2)
print(attn_score_22)

attn_scores_2 = query_2 @ keys.T # All attention scores for given query
print(attn_scores_2)

attn_scores = queries @ keys.T # omega
print(attn_scores)

d_k = keys.shape[-1]
attn_weights = torch.softmax(attn_scores / d_k**0.5, dim=-1)
print(attn_weights)
print(d_k)

import numpy as np

# Function to compute variance before and after scaling
def compute_variance(dim, num_trials=1000):
    dot_products = []
    scaled_dot_products = []

    # Generate multiple random vectors and compute dot products
    for _ in range(num_trials):
        q = np.random.randn(dim)
        k = np.random.randn(dim)
        
        # Compute dot product
        dot_product = np.dot(q, k)
        dot_products.append(dot_product)
        
        # Scale the dot product by sqrt(dim)
        scaled_dot_product = dot_product / np.sqrt(dim)
        scaled_dot_products.append(scaled_dot_product)
    
    # Calculate variance of the dot products
    variance_before_scaling = np.var(dot_products)
    variance_after_scaling = np.var(scaled_dot_products)

    return variance_before_scaling, variance_after_scaling

# For dimension 5
variance_before_5, variance_after_5 = compute_variance(5)
print(f"Variance before scaling (dim=5): {variance_before_5}")
print(f"Variance after scaling (dim=5): {variance_after_5}")

# For dimension 20
variance_before_100, variance_after_100 = compute_variance(100)
print(f"Variance before scaling (dim=100): {variance_before_100}")
print(f"Variance after scaling (dim=100): {variance_after_100}")

import torch.nn as nn

class SelfAttention_v1(nn.Module):

    def __init__(self, d_in, d_out):
        super().__init__()
        self.W_query = nn.Parameter(torch.rand(d_in, d_out))
        self.W_key   = nn.Parameter(torch.rand(d_in, d_out))
        self.W_value = nn.Parameter(torch.rand(d_in, d_out))

    def forward(self, x):
        keys = x @ self.W_key
        queries = x @ self.W_query
        values = x @ self.W_value
        
        attn_scores = queries @ keys.T # omega
        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)

        context_vec = attn_weights @ values
        return context_vec
    
torch.manual_seed(123)
sa_v1 = SelfAttention_v1(d_in, d_out)
print(sa_v1(inputs))

class SelfAttention_v2(nn.Module):

    def __init__(self, d_in, d_out, qkv_bias=False):
        super().__init__()
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key   = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)

    def forward(self, x):
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)
        
        attn_scores = queries @ keys.T
        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)

        context_vec = attn_weights @ values
        return context_vec
    
torch.manual_seed(789)
sa_v2 = SelfAttention_v2(d_in, d_out)
print(sa_v2(inputs))
