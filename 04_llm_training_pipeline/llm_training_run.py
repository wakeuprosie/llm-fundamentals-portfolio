from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch
from torch.utils.data import Dataset, DataLoader

# 1. Import model and tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")

# 2. Create a synthetic dataset class
class SyntheticLanguageDataset(Dataset):
    def __init__(self, num_samples=1024, seq_length=512, vocab_size=50257):
        self.num_samples = num_samples
        self.seq_length = seq_length
        self.vocab_size = vocab_size

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Generate random integer token IDs
        input_ids = torch.randint(0, self.vocab_size, (self.seq_length,))
        # For causal language modeling, labels are often the same as inputs (shifted internally by the model)
        labels = input_ids.clone() 
        return input_ids, labels

# 3. Create training loop 
def train():
    # Move the model to the first GPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Running baseline on: {device}")

    # create a model instance
    my_model = model.to(device)

    # Initiate synthethic data set
    dataset = SyntheticLanguageDataset(num_samples=128)

    # Initiate the dataloader
    dataloader = DataLoader(
      dataset, 
      batch_size=8, 
      shuffle=True,
      num_workers=2,
      pin_memory =True
    )

    # Optimize the training process
    optimizer = torch.optim.AdamW(my_model.parameters(), lr=1e-4)

    # Training loop
    for epoch in range(1): #We will do 1 epoch for testing
      for step, (input_ids, labels) in enumerate(dataloader):
        input_ids, labels = input_ids.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = my_model(input_ids, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        if step % 10 == 0:
          print(f"Epoch {epoch}, Step {step}, Loss: {loss.item()}")

# 4. Entry Point
if __name__ == "__main__":
    # Standard python execution, no torchrun needed
    train()