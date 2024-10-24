import data_collection
from base_imports import *
from time_embedding import TimeEmbedding

fixed_output_size = 8


class EncoderBlock(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, output_dim=8):
        super(EncoderBlock, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = x.view(x.size(0), 1, 8)

        return x


class UNet(nn.Module):
    def __init__(self, in_channels, out_channels, embed_dim=128):
        super(UNet, self).__init__()
        self.time_embedding = TimeEmbedding(embed_dim)
        self.data_encoder = EncoderBlock(in_channels=data_collection.batch_size, out_channels=64,
                                         fixed_output_size=fixed_output_size)

        self.enc1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.enc2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.enc3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.bottleneck = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU()
        )

        self.upconv3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec3 = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU()
        )

        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU()
        )

        self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU()
        )

        self.out_conv = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x, guidance, t):
        # Get time embedding
        t_emb = self.time_embedding(t)
        t_emb = t_emb.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, x.size(2), x.size(3))

        # Combine the input tensor, guidance tensor, and time embedding
        x = torch.cat([x, guidance, t_emb], dim=1)

        # Encoder
        enc1 = self.enc1(x)
        enc2 = self.enc2(self.pool1(enc1))
        enc3 = self.enc3(self.pool2(enc2))

        # Bottleneck
        bottleneck = self.bottleneck(self.pool3(enc3))

        # Decoder
        dec3 = self.dec3(torch.cat([self.upconv3(bottleneck), enc3], dim=1))
        dec2 = self.dec2(torch.cat([self.upconv2(dec3), enc2], dim=1))
        dec1 = self.dec1(torch.cat([self.upconv1(dec2), enc1], dim=1))

        # Output layer
        output = self.out_conv(dec1)
        return output


class DiffusionModel(nn.Module):
    def __init__(self, embed_dim=128):
        super(DiffusionModel, self).__init__()
        self.time_embedding = TimeEmbedding(embed_dim)
        self.data_encoder = EncoderBlock(in_channels=data_collection.batch_size, out_channels=64,
                                         fixed_output_size=fixed_output_size)
        self.model = nn.Sequential(
            nn.Conv2d(130, 64, kernel_size=3, padding=1),  # Input is concatenated data, guidance, and time embedding
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 1, kernel_size=3, padding=1)
        )

    def forward(self, x, guidance, t):
        t_emb = self.time_embedding(t)
        guidance = self.data_encoder(guidance)
        t_emb = t_emb.unsqueeze(-1).expand(-1, x.size(1), x.size(2))
        x = torch.cat([x, guidance, t_emb], dim=1)
        return self.model(x)
