import torch
import torch.nn as nn

def train_model(model, train_loader):
    # Define the loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Training the model
    num_epochs = 5

    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        for batch_idx, (data, targets) in enumerate(train_loader):
            # forward pass
            scores = model(data)
            loss = criterion(scores, targets)

            # backward pass
            optimizer.zero_grad()
            loss.backward()

            # gradient descent
            optimizer.step()
