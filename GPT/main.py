import torch
import torch.nn as nn
from torch.nn import functional as F
import os
import sys
import re
import difflib
import json
import csv
from PyQt5.QtCore import QObject, pyqtSignal
from tqdm import tqdm
import torch.optim as optim

# hyperparameters
batch_size = 30 # how many independent sequences will we process in parallel?
block_size = 64 # what is the maximum context length for predictions?
max_iters = 3000  # How many training iterations will we have?
eval_interval = 100  # every ____ iterations, we evaluate the model
learning_rate = 1e-2 # how fast the model "learns" (changes)
device = 'cuda' if torch.cuda.is_available() else 'cpu'  # use a GPU or CPU
eval_iters = 200
n_embd = 64
n_head = 4
n_layer = 4
dropout = 0.2

# ------------
train_override = not True

torch.manual_seed(1337)

def find_closest_word(word, word_list):
    closest_matches = difflib.get_close_matches(word, word_list)
    return closest_matches[0] if closest_matches else word

class Tokenizer:
    def __init__(self):
        self.word2idx = {}
        self.idx2word = {}
        self.vocab_size = 0

    def tokenize(self, text):
        # Split text into words or subwords
        #text = text.lower()
        tokens = text.split()
        return tokens

    def build_vocab(self, texts):
        # Iterate through all texts to build vocabulary
        for text in texts:
            tokens = self.tokenize(text)
            tokens.append(' ')
            for token in tokens:
                if token not in self.word2idx:
                    self.word2idx[token] = self.vocab_size
                    self.idx2word[self.vocab_size] = token
                    self.vocab_size += 1

    def text_to_indices(self, text):
        # Convert text to list of numerical indices
        tokens = self.tokenize(text)
        indices = [self.word2idx[token] for token in tokens]
        return indices

    def indices_to_text(self, indices):
        # Convert list of indices back to text
        try:
            tokens = [self.idx2word[idx] for idx in indices]
            text = ' '.join(tokens)

        except:
            text = []
            for indice in indices:
                if indice in self.idx2word:
                    text.append(self.idx2word[indice])
                for k, v in self.word2idx.items():
                    if v == indice:
                        text.append(k)

            text = ' '.join(text)
    
                
        return text

class Dataset(torch.utils.data.Dataset):

    def __init__(self, path, reverse = False, tokenize = True):
        super().__init__()

        files = os.listdir(path)
        self.items = []
        self.tokens = Tokenizer()
        for file in files:
            with open(os.path.join(path, file), 'rb') as f:
                self.items.append(f.read().decode())

        self.tokens.build_vocab(self.items)
        self.reverse = reverse

        if not tokenize:
            self.items = [item.split() for item in self.items]

        else:
            self.items = [self.tokens.text_to_indices(item) for item in self.items]

        print(self.items[:5])
        self.len = sum(len(item) - 1 for item in self.items)
        self.maxlen = max(len(item) for item in self.items)
        print('MAXLEN', self.maxlen)

    def __getitem__(self, index, tokenize = True):
        length = 0

        for item in self.items:
            if length <= index < length + len(item) - 1:
                if not self.reverse:
                    
                    x = torch.tensor(item[0: index - length + 1], dtype = torch.float32, device = device)
                    x = nn.functional.pad(x, (self.maxlen - x.shape[-1], 0))
                    y = torch.tensor(item[index - length + 1], dtype = torch.float32, device = device)
                    return x, y

                else:
                
                    x = torch.tensor(list(reversed(item[index - length:])), dtype = torch.float32, device = device)
                    x = nn.functional.pad(x, (self.maxlen - x.shape[-1], 0))
                    y = torch.tensor(item[index - length - 1], dtype = torch.float32, device = device)
                    return x, y
                

            length += len(item)

        return self[0]

        '''try:
            return self[index + 1]

        except:
            return self[0]'''

            

    def __len__(self):
        return self.len - 10



#tokens = Tokenizer()

    
#with open('Minecraft2.txt', 'rb') as f:#r"C:\Users\krivi\Downloads\GPT test 2\GPT test 2\data\data.txt", 'r') as f:
#    text = f.read().decode()
#    print(text[:3000])
#    print('Building vocabulary...')
#    tokens.build_vocab([text,])
#    print('Finished')
    



# here are all the unique characters that occur in this text

#vocab_size = tokens.vocab_size#len(chars)
# create a mapping from characters to integers





# data loading


@torch.no_grad()
def estimate_loss(model, get_batch):
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

class Head(nn.Module):
    """ one head of self-attention """

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B,T,C = x.shape
        k = self.key(x)   # (B,T,C)
        q = self.query(x) # (B,T,C)
        # compute attention scores ("affinities")
        wei = q @ k.transpose(-2,-1) * C**-0.5 # (B, T, C) @ (B, C, T) -> (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B, T, T)
        wei = F.softmax(wei, dim=-1) # (B, T, T)
        wei = self.dropout(wei)
        # perform the weighted aggregation of the values
        v = self.value(x) # (B,T,C)
        out = wei @ v # (B, T, T) @ (B, T, C) -> (B, T, C)
        return out

class MultiHeadAttention(nn.Module):
    """ multiple heads of self-attention in parallel """

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

class FeedFoward(nn.Module):
    """ a simple linear layer followed by a non-linearity """

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    """ Transformer block: communication followed by computation """

    def __init__(self, n_embd, n_head):
        # n_embd: embedding dimension, n_head: the number of heads we'd like
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedFoward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

# super simple bigram model
class BigramLanguageModel(nn.Module):

    def __init__(self, vocab_size):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd) # final layer norm
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        # idx and targets are both (B,T) tensor of integers
        tok_emb = self.token_embedding_table(idx) # (B,T,C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # (T,C)
        x = tok_emb + pos_emb # (B,T,C)
        x = self.blocks(x) # (B,T,C)
        x = self.ln_f(x) # (B,T,C)
        logits = self.lm_head(x) # (B,T,vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # crop idx to the last block_size tokens
            idx_cond = idx[:, -block_size:]
            # get the predictions
            logits, loss = self(idx_cond)
            # focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx

class Trainer():
    LossCalculated = pyqtSignal(str, int, int)
    
    def train(self, dirname):

        tokens = Tokenizer() # use the same tokenizer for both models
        losses_train_1 = []
        losses_val_1 = []
        losses_train_2 = []
        losses_val_2 = []

        texts = []

        for filename in tqdm(os.listdir(os.path.join(dirname, 'data'))[:2000]):
            filepath = os.path.join(dirname, 'data', filename)
            with open(filepath, 'rb') as f:#r"C:\Users\krivi\Downloads\GPT test 2\GPT test 2\data\data.txt", 'r') as f:
                text = f.read().decode()
                tokens.build_vocab([text,])
                texts.append(text)
        c = 0
        for text in tqdm(texts):
            print(text)

            # encoder: take a string, output a list of integers
            def encode(text):
                try:
                    return tokens.text_to_indices(text)

                except KeyError:
                    wordlist = list(tokens.word2idx.keys())
                    text = ' '.join([find_closest_word(word, wordlist) for word in text.split()])
                    return tokens.text_to_indices(text)
                    

            decode = tokens.indices_to_text   # decoder: take a list of integers, output a string

            # Train and test splits
            data = torch.tensor(encode(text), dtype=torch.long)
            n = int(0.9*len(data)) # first 90% will be train, rest val
            train_data = data[:n]
            val_data = data[n:]
            def get_batch_(split, train_data, val_data):
                # generate a small batch of data of inputs x and targets y
                data = train_data if split == 'train' else val_data
                ix = torch.randint(len(data) - block_size, (batch_size,))
                x = torch.stack([data[i:i+block_size] for i in ix])
                y = torch.stack([data[i+1:i+block_size+1] for i in ix])
                x, y = x.to(device), y.to(device)
                return x, y

            get_batch = lambda split: get_batch_(split, torch.tensor(list(reversed(train_data.tolist()))), torch.tensor(list(reversed(val_data.tolist())))) 
                
            model_2 = BigramLanguageModel(tokens.vocab_size)
            m_2 = model_2.to(device)
            #torch.save(m, 'GPT.pt')
            # print the number of parameters in the model
            print(sum(p.numel() for p in m_2.parameters())/1e6, 'M parameters')

            # create a PyTorch optimizer
            optimizer_2 = torch.optim.AdamW(model_2.parameters(), lr=learning_rate)
            print('Training backward')
            for iter in range(max_iters):
                try:
                    # every once in a while evaluate the loss on train and val sets
                    if iter % eval_interval == 0 or iter == max_iters - 1:
                        losses = estimate_loss(model_2, get_batch)
                        losses_train_2.append(losses['train'])
                        losses_val_2.append(losses['val'])
                        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
                        # self.LossCalculated.emit('backward', losses['train'], losses['val'])

                    # sample a batch of data
                    xb, yb = get_batch('train')

                    # evaluate the loss
                    logits, loss = model_2(xb, yb)
                    optimizer_2.zero_grad(set_to_none=True)
                    loss.backward()
                    optimizer_2.step()

                except:
                    pass


            get_batch = lambda split: get_batch_(split, train_data, val_data) 
                
            model_1 = BigramLanguageModel(tokens.vocab_size)
            m_1 = model_1.to(device)
            #torch.save(m, 'GPT.pt')
            # print the number of parameters in the model
            print(sum(p.numel() for p in m_1.parameters())/1e6, 'M parameters')

            # create a PyTorch optimizer
            optimizer_1 = torch.optim.AdamW(model_1.parameters(), lr=learning_rate)
            print('Training forward')
            for iter in range(max_iters):
                try:
                    # every once in a while evaluate the loss on train and val sets
                    if iter % eval_interval == 0 or iter == max_iters - 1:
                        losses = estimate_loss(model_1, get_batch)
                        losses_train_1.append(losses['train'])
                        losses_val_1.append(losses['val'])
                        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
                        # self.LossCalculated.emit('forward', losses['train'], losses['val'])

                    # sample a batch of data
                    xb, yb = get_batch('train')

                    # evaluate the loss
                    logits, loss = model_1(xb, yb)
                    optimizer_1.zero_grad(set_to_none=True)
                    loss.backward()
                    optimizer_1.step()

                except:
                    pass

            torch.save(m_1, filename.replace('.txt', 'model_forward.pt'))
            torch.save(m_2, filename.replace('.txt', 'model_backward.pt'))
            encodedecode = {}
            encodedecode['encode'] = tokens.word2idx
            encodedecode['decode'] = tokens.idx2word
            encodedecode['vocabsize'] = tokens.vocab_size
            with open(filename.replace('.txt', '.json'), 'w') as f:
                      f.write(json.dumps(encodedecode, indent = 5))









            # save the model
            print(os.path.dirname(filename))
            torch.save(m_1, os.path.join(dirname, filename.replace('.txt', 'model_forward.pt')))
            torch.save(m_2, os.path.join(dirname, filename.replace('.txt', 'model_backward.pt')))
            encodedecode = {}
            encodedecode['encode'] = tokens.word2idx
            encodedecode['decode'] = tokens.idx2word
            encodedecode['vocabsize'] = tokens.vocab_size
            with open(os.path.join(dirname, filename.replace('.txt', '.json')), 'w') as f:
                      f.write(json.dumps(encodedecode, indent = 5))

            with open(os.path.join(dirname, filename.replace('.txt', '_forward.csv')), 'w') as f:
                writer = csv.writer(f, delimiter = ',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
                writer.writerow(['Train', 'Val'])
                for index in range(len(losses_val_1)):
                    writer.writerow([str(losses_train_1[index].tolist()), str(losses_val_1[index].tolist())])

            with open(os.path.join(filename.replace('.txt', '_backward.csv')), 'w') as f:
                writer = csv.writer(f, delimiter = ',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
                writer.writerow(['Train', 'Val'])
                for index in range(len(losses_val_1)):
                    writer.writerow([str(losses_train_2[index].tolist()), str(losses_val_2[index].tolist())])

            with open('progress.txt', 'w') as f:
                f.write(str(c))

            c += 1


train = Trainer().train
        
def load(folder):
    files = os.listdir(folder)
    pt = [file for file in files if '.pt' in file]
    encodings = os.path.join(folder, [file for file in files if '.json' in file][0])
    datas = [os.path.join(folder, 'data', file) for file in os.path.join(folder, 'data')]
    losses = [file for file in files if '.csv' in file]
    tokens = Tokenizer()
    forward_loss = None
    backward_loss = None
    pt_forward = None
    pt_backward = None

    for file in pt:
        if 'model_forward' in file and not pt_forward:
            pt_forward = folder + '/' + file

        if 'model_backward' in file and not pt_backward:
            pt_backward = folder + '/' + file

    for file in losses:
        if '_forward' in file and not forward_loss:
            forward_loss = folder + '/' + file

        if '_backward' in file and not backward_loss:
            backward_loss = folder + '/' + file
    
    assert None not in (pt_forward, pt_backward)


    model_forward = torch.load(pt_forward, map_location=torch.device(device))
    model_backward = torch.load(pt_backward, map_location=torch.device(device))
    try:
        with open(encodings, 'r') as f:
            contents = f.read()
            encodings = json.loads(contents)
            tokens.word2idx = encodings['encode']
            tokens.idx2word = encodings['decode']
            tokens.vocab_size = int(encodings['vocabsize'])
        print('Encodings restored from original')

    except:
        for data in datas:
            try:
                with open(data, 'rb') as f:
                    data = f.read().decode()
                    tokens.build_vocab([data,])
            except:
                pass

    '''with open(encodings, 'r') as f:
        contents = f.read()
        encodings = json.loads(contents)
        tokens.word2idx = encodings['encode']
        tokens.idx2word = encodings['decode']
        tokens.vocab_size = int(encodings['vocabsize'])'''

    train_loss_forward = []
    val_loss_forward = []
    train_loss_backward = []
    val_loss_backward = []

    try:
    
        if forward_loss and backward_loss:
            with open(forward_loss, 'r') as f:
                
                reader = csv.reader(f)
                for row in reader:
                    try:
                        train_loss_forward.append(int(row[0]))
                        val_loss_forward.append(int(row[1]))

                    except:
                        pass

            with open(backward_loss, 'r') as f:
                
                reader = csv.reader(f)
                for row in reader:
                    try:
                        train_loss_backward.append(int(row[0]))
                        val_loss_backward.append(int(row[1]))

                    except:
                        pass

    except:
        pass

    return model_forward, model_backward, datas, tokens, train_loss_forward, val_loss_forward, train_loss_backward, val_loss_backward


def load_old(folder):
    files = os.listdir(folder)
    pt = [file for file in files if '.pt' in file]
    encodings = os.path.join(folder, [file for file in files if '.json' in file][0])
    data = [file for file in files if '.txt' in file][0]
    losses = [file for file in files if '.csv' in file]
    tokens = Tokenizer()
    forward_loss = None
    backward_loss = None
    pt_forward = None
    pt_backward = None

    for file in pt:
        if 'model_forward' in file and not pt_forward:
            pt_forward = folder + '/' + file

        if 'model_backward' in file and not pt_backward:
            pt_backward = folder + '/' + file

    for file in losses:
        if '_forward' in file and not forward_loss:
            forward_loss = folder + '/' + file

        if '_backward' in file and not backward_loss:
            backward_loss = folder + '/' + file
    
    assert None not in (pt_forward, pt_backward)


    model_forward = torch.load(pt_forward, map_location=torch.device(device))
    model_backward = torch.load(pt_backward, map_location=torch.device(device))
    
    with open(data, 'rb') as f:
        data = f.read().decode()
        tokens.build_vocab([data,])

    '''with open(encodings, 'r') as f:
        contents = f.read()
        encodings = json.loads(contents)
        tokens.word2idx = encodings['encode']
        tokens.idx2word = encodings['decode']
        tokens.vocab_size = int(encodings['vocabsize'])'''

    train_loss_forward = []
    val_loss_forward = []
    train_loss_backward = []
    val_loss_backward = []
    
    if forward_loss and backward_loss:
        with open(forward_loss, 'r') as f:
            
            reader = csv.reader(f)
            for row in reader:
                try:
                    train_loss_forward.append(int(row[0]))
                    val_loss_forward.append(int(row[1]))

                except:
                    pass

        with open(backward_loss, 'r') as f:
            
            reader = csv.reader(f)
            for row in reader:
                try:
                    train_loss_backward.append(int(row[0]))
                    val_loss_backward.append(int(row[1]))

                except:
                    pass

    return model_forward, model_backward, data, tokens, train_loss_forward, val_loss_forward, train_loss_backward, val_loss_backward

def get_all():
    models = []
    directory = os.path.dirname(__file__)
    for root, dirs, files in list(os.walk(directory)):
        if directory == root:
            for directory in dirs:
                try:
                    models.append(Model(directory))
                except (IndexError, AssertionError):
                    continue

    return models
            
class Discriminator(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)


class Dummy_Data(torch.utils.data.Dataset):

    def __init__(self, tokenizer):
        self.t = tokenizer

    def __getitem__(self, index):
        return torch.tensor(index, device = device).long().unsqueeze(0)

    def __len__(self):
        return self.t.vocab_size

    
class Model():
    
    def __init__(self, folder):
        self.folder = folder
        try:
            self.model_forward, self.model_backward, self.data, self.tokens, self.train_loss_forward, self.val_loss_forward, self.train_loss_backward, self.val_loss_backward = load(folder)

        except:
            self.model_forward, self.model_backward, self.data, self.tokens, self.train_loss_forward, self.val_loss_forward, self.train_loss_backward, self.val_loss_backward = load_old(folder)

            
        self.decode = self.tokens.indices_to_text
        self.optimizer_1 = optim.Adam(self.model_forward.parameters(), lr=0.001)
        self.optimizer_2 = optim.Adam(self.model_backward.parameters(), lr=0.001)

    def __str__(self):
        return str(self.folder)

    def encode(self, text):
        try:
            return self.tokens.text_to_indices(text)

        except KeyError:
            wordlist = list(self.tokens.word2idx.keys())
            text = ' '.join([find_closest_word(word, wordlist) for word in text.split()])
            return self.tokens.text_to_indices(text)

    def generate(self, prompt, length = 50, reverse_length = 50, wait_until_period = True, exact = False):
        string = prompt
        prompt = orig = torch.tensor(self.encode(prompt), dtype = torch.long, device = device).unsqueeze(0)
        prompt = self.model_forward.generate(prompt, max_new_tokens = length)
        while wait_until_period and '.' not in self.decode([prompt.squeeze(0).tolist()[-1],]) and not exact:
            prompt = self.model_forward.generate(prompt, max_new_tokens = 1)

        f = self.decode(prompt.squeeze(0).tolist())
        
        other_direction = torch.tensor(list(reversed(prompt.squeeze(0).tolist())), dtype = torch.long, device = device).unsqueeze(0)#torch.tensor(list(reversed(self.encode(f))), dtype = torch.long, device = device).unsqueeze(0)
        b = self.decode(self.model_backward.generate(other_direction, max_new_tokens = reverse_length + 1).squeeze(0).tolist())
        while b.count('.') <= f.count('.') and not exact:
            other_direction = self.model_backward.generate(other_direction, max_new_tokens = 1)
            
            b = self.decode(other_direction.squeeze(0).tolist())
        b = ' '.join(reversed(b.split()[:-1]))# + ' '
        
        return b  #, self.decode(prompt.squeeze(0).tolist())


    def train(self, data = None):
        if not data:
            data = os.path.join(self.folder, 'data')
            
        data1 = Dataset(data)
        data2 = Dataset(data, True)
        train_data1 = torch.utils.data.DataLoader(data1, batch_size = 1)
        train_data2 = torch.utils.data.DataLoader(data2, batch_size = 1)

        def train(model, optimizer, loss_fn, train_loader, val_loader, epochs=20, device="cpu"):
            for epoch in range(1, epochs+1):
                training_loss = 0.0
                valid_loss = 0.0
                model.train()
                for batch in train_loader:
                    optimizer.zero_grad()
                    inputs, targets = batch
                    inputs = inputs.to(device)
                    targets = targets.to(device)
                    output = model(inputs)
                    loss = loss_fn(output, targets)
                    loss.backward()
                    optimizer.step()
                    training_loss += loss.data.item() * inputs.size(0)
                training_loss /= len(train_loader.dataset)
                torch.cuda.empty_cache()
                model.eval()
                num_correct = 0 
                num_examples = 0
                for batch in val_loader:
                    inputs, targets = batch
                    inputs = inputs.to(device)
                    output = model(inputs)
                    targets = targets.to(device)
                    loss = loss_fn(output,targets) 
                    valid_loss += loss.data.item() * inputs.size(0)
                    correct = torch.eq(torch.max(F.softmax(output, dim=1), dim=1)[1], targets)
                    num_correct += torch.sum(correct).item()
                    num_examples += correct.shape[0]
                valid_loss /= len(val_loader.dataset)

                print('Epoch: {}, Training Loss: {:.2f}, Validation Loss: {:.2f}, accuracy = {:.2f}'.format(epoch, training_loss,
                valid_loss, num_correct / num_examples))

        train(self.model_forward, self.optimizer_1, nn.CrossEntropyLoss(), train_data1, 30, device)
        # train(self.model_backward, self.optimizer_2, nn.CrossEntropyLoss(), train_data2, 30, device)

    def gan(self, data = None):
        if not data:
            data = os.path.join(self.folder, 'data')

        
        discriminator = Discriminator(11089, 129, 1)
        discriminator.to(device)
        criterion = nn.CrossEntropyLoss()
        dummy_loader = torch.utils.data.DataLoader(Dummy_Data(self.tokens), batch_size = 65536)
        data1 = Dataset(data)
        data2 = Dataset(data, True)
        train_data1 = torch.utils.data.DataLoader(data1, batch_size = 128)
        train_data2 = torch.utils.data.DataLoader(data2, batch_size = 128)
        discriminator_optimizer = optim.Adam(discriminator.parameters(), lr=0.001)
        
        
        def gan_train(generator, generator_optimizer, discriminator, discriminator_optimizer, real_train_loader, dummy_loader, criterion, num_epochs = 100):
            for epoch in range(1, num_epochs + 1):
                for batch in real_train_loader:
                    inputs, _ = batch
                    # train discriminator
                    discriminator.train()
                    generator.eval()
                    discriminator.zero_grad()
                    dim = (302 - inputs.shape[-1]) // 2
                    preds = discriminator(nn.functional.pad(inputs, (11089 - inputs.shape[-1], 0)))
                    real_loss = criterion(preds, torch.ones_like(preds))
                    real_loss.backward()
                torch.cuda.empty_cache()
                for inpt in dummy_loader:
                    print(inpt)
                    fake_batch = generator(inpt)
                    fake_preds = discriminator(fake_batch)
                    fake_loss = criterion(fake_preds, torch.zeros_like(fake_preds))
                    fake_loss.backward()
                    discriminator_optimizer.step()

                    # train generator
                    discriminator.eval()
                    generator.train()
                    generator.zero_grad()
                    forged_batch = generator(inpt)
                    forged_preds = discriminator(forged_batch)
                    forged_loss = criterion(forged_preds, torch.ones_like(forged_preds))
                    forged_loss.backward()
                    generator_optimizer.step()
                print('Epoch: ', epoch)
                print(forged_loss.item())

        gan_train(self.model_forward, self.optimizer_1, discriminator, discriminator_optimizer, train_data1, dummy_loader, criterion)
        # gan_train(self.model_backward, self.optimizer_2, discriminator, discriminator_optimizer, train_data2, dummy_loader, criterion)

    def save(self, folder = None):
        if not folder:
            folder = self.folder + '2'
            
        shutil.copytree(self.folder, folder)
        files = os.listdir(self.folder)
        forward_pt = [name for name in files if 'forward' in name and '.pt' in name][0]
        backward_pt = [name for name in files if 'backward' in name and '.pt' in name][0]

        torch.save(os.path.join(folder, forward_pt), forward_pt)
        torch.save(os.path.join(folder, backward_pt), backward_pt)
        
        


#print(str(get_all()[0]))

def print_colored_text(text, clr = 'BUILTIN'):
    color = sys.stdout.shell
    color.write(text, clr)
    sys.stdout.flush()
# data = os.path.join('math', 'data')
# data1 = Dataset(data, False, False)
# data2 = Dataset(data, True, False)
# train_data1 = torch.utils.data.DataLoader(data1, batch_size = 1)
# train_data2 = torch.utils.data.DataLoader(data2, batch_size = 1)

#test = True

#train('math2')

if __name__ == '__main__' and test:
    inpt = 'blah'
    m = Model('Minecraft2_model_expertise')
    m.gan()
    m.save()
    print('SYSTEMS UP AND RUNNING')
    while inpt:
        inpt = input()
        if not inpt:
            break
        print_colored_text(m.generate(inpt, 50, 50, exact = True))



