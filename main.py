import torch
from torch import nn
from torch.utils.data import DataLoader
from model import SpectralSuperResolution
from data import AudioDataset

def train(lr_path, hr_path, epochs=50, batch_size=16, learning_rate=1e-3):
    # Load the dataset
    dataset = AudioDataset(hr_path, lr_path, segment_length=44100)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Initialize the model
    model = SpectralSuperResolution()
    if torch.cuda.is_available():
        model = model.cuda()

    # Set up the loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Train the model
    for epoch in range(epochs):
        for batch_idx, (lr, hr) in enumerate(dataloader):
            if torch.cuda.is_available():
                lr, hr = lr.cuda(), hr.cuda()

            # Forward pass
            outputs = model(lr)
            loss = criterion(outputs, hr)

            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print(f"Epoch [{epoch+1}/{epochs}], Batch [{batch_idx+1}/{len(dataloader)}], Loss: {loss.item()}")

    # Save the model
    torch.save(model.state_dict(), "model.pth")

# Use this to start the training
train("lr", "hr")