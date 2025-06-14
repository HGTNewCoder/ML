import torch
from torch.utils.tensorboard import SummaryWriter
import torchvision.datasets as datasets
import torchvision
from tqdm import tqdm
from torch import nn
from model import Discriminator, Generator
from torchvision import transforms
from torch.utils.data import DataLoader

DEVICE = torch.device("cuda") if torch.cuda.is_available() else "cpu"
LR = 0.001
Z_DIM = 128
IMG_DIM = 784
BATCH_SIZE = 16
NUM_EPOCH = 50

disc = Discriminator(IMG_DIM).to(DEVICE)
gen = Generator(Z_DIM, IMG_DIM).to(DEVICE)
fixed_noise = torch.randn((BATCH_SIZE, Z_DIM)).to(DEVICE)

transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)) #Mean and STD of MNIST
])

dataset = datasets.MNIST(transform=transforms, download=True, train=True, root="dataset/")
train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
opt_disc = torch.optim.Adam(disc.parameters(), lr=LR)
opt_gen = torch.optim.Adam(gen.parameters(), lr=LR)
criterion = nn.BCELoss(reduction="sum")
writer_fake = SummaryWriter(f"runs/GAN_MNIST/fake")
writer_real = SummaryWriter(f"runs/GAN_MNIST/real")
step = 0

for epoch in range(NUM_EPOCH):
    loop = tqdm(train_loader, leave=True)
    for i, (real, _) in enumerate(loop):
        #Forward
        real = real.view(-1, IMG_DIM).to(DEVICE)
        batch_size = real.shape[0]
        
        #Train Discriminator max log(D(real) + log(1-D(G(z))))
        noise = torch.randn(batch_size, Z_DIM).to(DEVICE)
        fake = gen(noise)
        disc_real = disc(real).view(-1)
        disc_fake = disc(fake).view(-1)
        
        loss_disc_real = criterion(disc_real, torch.ones_like(disc_real)) # Based on BCELoss formula, we choose this
        loss_disc_fake = criterion(disc_fake, torch.zeros_like(disc_fake)) # Based on BCELoss formula, we choose this
        loss_disc = 1/2 * (loss_disc_fake + loss_disc_real)
        
        disc.zero_grad()
        loss_disc.backward(retain_graph=True) #Why retain_graph
        opt_disc.step()
        
        #Train Generator min log(1 - D(G(z))) <-> max log D(G(z))
        output = disc(fake).view(-1)
        lossG = criterion(output, torch.ones_like(output))
        
        gen.zero_grad()
        lossG.backward()
        opt_gen.step()
        
        if i == 0:
            loop.set_description(f"Epoch [{epoch}/{NUM_EPOCH}]")
            loop.set_postfix(D_loss=loss_disc.item(), G_loss=lossG.item())

            with torch.no_grad():
                fake = gen(fixed_noise).reshape(-1, 1, 28, 28)
                data = real.reshape(-1, 1, 28, 28)
                img_grid_fake = torchvision.utils.make_grid(fake, normalize=True)
                img_grid_real = torchvision.utils.make_grid(data, normalize=True)

                writer_fake.add_image(
                    "Mnist Fake Images", img_grid_fake, global_step=step
                )
                writer_real.add_image(
                    "Mnist Real Images", img_grid_real, global_step=step
                )
                step += 1
