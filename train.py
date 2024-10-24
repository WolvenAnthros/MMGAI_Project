from base_imports import *
from time_embedding import TimeEmbedding
from data_collection import *
from model import UNet, DiffusionModel
from circuit_simulator import totally_mixed_state


def linear_beta_schedule(timesteps, start=0.0001, end=0.02):
    return torch.linspace(start, end, timesteps)


def get_index_from_list(vals, t, x_shape):
    """
    Returns a specific index t of a passed list of values vals
    while considering the batch size.
    """
    batch_size = t.shape[0]
    out = vals.gather(-1, t.cpu())
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)


def forward_diffusion_sample(x_0, t, sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod, eps=None):
    """
    Takes an image and a timestep as input and
    returns the noisy version of it
    """
    if eps is None:
        eps = totally_mixed_state
    alpha_t = get_index_from_list(sqrt_alphas_cumprod, t, x_0.shape)
    sqrt_one_minus_alpha_t = get_index_from_list(sqrt_one_minus_alphas_cumprod, t, x_0.shape)
    return alpha_t * x_0 + sqrt_one_minus_alpha_t * eps, eps


# Define the number of timesteps
timesteps = 300
betas = linear_beta_schedule(timesteps=timesteps)
alphas = 1. - betas
alphas_cumprod = torch.cumprod(alphas, axis=0)
sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)

# model = UNet(in_channels=130, out_channels=1, embed_dim=128)
model = DiffusionModel(embed_dim=84)
optimizer = optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.MSELoss()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
sqrt_alphas_cumprod = sqrt_alphas_cumprod.to(device)
sqrt_one_minus_alphas_cumprod = sqrt_one_minus_alphas_cumprod.to(device)


# Training loop
def train_model(dataloader, epochs=10):
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for data_tensors, guidance_tensors in dataloader:
            data_tensors = data_tensors.to(device)
            guidance_tensors = guidance_tensors.to(device)
            # guidance_tensors = encoder(guidance_tensors)
            t = torch.randint(0, timesteps, (data_tensors.size(0),), device=device).long()

            x_noisy, noise = forward_diffusion_sample(data_tensors, t, sqrt_alphas_cumprod,
                                                      sqrt_one_minus_alphas_cumprod)
            predicted_noise = model(x_noisy, guidance_tensors, t)

            loss = loss_fn(predicted_noise, noise)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        # Print the average loss for the epoch
        avg_loss = running_loss / len(dataloader)
        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {avg_loss:.4f}")


# Train the model
train_model(dataloader)

# Save the model
torch.save(model.state_dict(), 'diffusion_model.pth')
