import os
import argparse
import shutil
import time
import csv
from PIL import Image
from tqdm import tqdm

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import torchvision
import torchvision.transforms as transforms

from GAN.model import Discriminator_64_G, Generator_64_G
from GAN.model import Discriminator_64_RGB, Generator_64_RGB
from GAN.model import Discriminator_256_G, Generator_256_G
from GAN.model import Discriminator_256_RGB, Generator_256_RGB
from GAN.model import Discriminator_512_G, Generator_512_G
from GAN.model import Discriminator_512_RGB, Generator_512_RGB

class PokemonCardsDataset(Dataset):
    def __init__(self, root_dir, args, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = [os.path.join(root_dir, filename) for filename in os.listdir(root_dir) if filename.endswith('.png')]
        self.color = 'RGB' if args.color else 'L'

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert(self.color)
        if self.transform:
            image = self.transform(image)
        return image

def parse_args():
    parser = argparse.ArgumentParser(description='Train GAN')

    parser.add_argument('--device', type=str, default='cuda', help='Device to train on')
    parser.add_argument('--benchmark', type=bool, default=True, help='Enable cudnn benchmark (only for fixed size inputs)')
    parser.add_argument('--mode', type=str, default='train', help='Mode to run the script in')
    parser.add_argument('--exp_name', type=str, default='exp000', help='Name of the experiment')
    parser.add_argument('--data_dir', type=str, default='data', help='Path to the data directory')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--image_size', type=int, choices=[64, 256, 512], default=64, help='Size of the image')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs to train for')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--color', action='store_true', help='Use color images')
    parser.add_argument('--print_freq', type=int, default=10, help='Print frequency')
    parser.add_argument('--resume', type=bool, default=False, help='Resume training')
    parser.add_argument('--resume_epoch', type=int, default=0, help='Epoch to resume training from')
    parser.add_argument('--resume_dir', type=str, default='', help='Path to the model to resume training from')

    return parser.parse_args()

def initialize_csv(file_path):
    headers = [
        "Epoch", "Total_Epochs", "Batch", "Total_Batches",
        "D_Loss", "G_Loss", "D_Real", "D_Fake1", "D_Fake2",
        "Elapsed_Time_Epoch", "Elapsed_Time_Batch",
        "Average_Time_s"
    ]
    if not os.path.exists(file_path):
        with open(file_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(headers)

def append_to_csv(file_path, data):
    with open(file_path, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(data)

def train(args, writer):
    data_dir = args.data_dir
    image_size = args.image_size
    batch_size = args.batch_size
    start_epoch = 0
    epochs = args.epochs
    exp_name = args.exp_name

    if args.resume:
        start_epoch = args.resume_epoch
        d_weights = os.path.join(args.resume_dir, f'd_epoch{args.resume_epoch}.pth')
        g_weights = os.path.join(args.resume_dir, f'g_epoch{args.resume_epoch}.pth')



    if args.device == 'cuda':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if torch.cuda.is_available():
            print(f'Using {torch.cuda.get_device_name(0)}')
        else:
            print('Using CPU')
    else:
        device = torch.device('cpu')

    if args.color:
        if image_size == 64:
            print('Using 64x64 color images')
            D = Discriminator_64_RGB().to(device)
            G = Generator_64_RGB().to(device)
        elif image_size == 256:
            print('Using 256x256 color images')
            D = Discriminator_256_RGB().to(device)
            G = Generator_256_RGB().to(device)
        elif image_size == 512:
            print('Using 512x512 color images')
            D = Discriminator_512_RGB().to(device)
            G = Generator_512_RGB().to(device)
    else:
        if image_size == 64:
            print('Using 64x64 grayscale images')
            D = Discriminator_64_G().to(device)
            G = Generator_64_G().to(device)
        elif image_size == 256:
            print('Using 256x256 grayscale images')
            D = Discriminator_256_G().to(device)
            G = Generator_256_G().to(device)
        elif image_size == 512:
            print('Using 512x512 grayscale images')
            D = Discriminator_512_G().to(device)
            G = Generator_512_G().to(device)

    if args.resume:
        D.load_state_dict(torch.load(d_weights))
        G.load_state_dict(torch.load(g_weights))

    criterion = nn.BCELoss().to(device)

    optimizerD = optim.Adam(D.parameters(), lr=0.0002, betas=(0.5, 0.999))
    optimizerG = optim.Adam(G.parameters(), lr=0.0002, betas=(0.5, 0.999))

    samples_dir = os.path.join('GAN/samples', exp_name)
    result_dir = os.path.join('GAN/results', exp_name)

    if not os.path.exists(samples_dir):
        os.makedirs(samples_dir)
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    metrics_path = os.path.join(result_dir, 'train.csv')

    if not os.path.exists(metrics_path):
        initialize_csv(metrics_path)

    fixed_noise = torch.randn(args.batch_size, 100, 1, 1, device=device)

    # Define the transformations
    transform = transforms.Compose([
        transforms.Resize([image_size, image_size]),
        transforms.ToTensor(),
        transforms.Normalize([0.5, ], [0.5, ])
    ])

    # Load the dataset
    dataset = PokemonCardsDataset(root_dir=data_dir, args=args, transform=transform)

    # Create the dataloader
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, pin_memory=True)

    for epoch in range(start_epoch, epochs):

        # Training
        batches = len(dataloader)

        D.train()
        G.train()

        epoch_start_time = time.time()

        for i, real in enumerate(dataloader):
            batch_start_time = time.time()
            real = real.to(device)
            label_size = real.size(0)

            # Create label. Real label is 1, fake label is 0.
            real_label = torch.full([label_size, 1], 1.0, dtype=real.dtype, device=device)
            fake_label = torch.full([label_size, 1], 0.0, dtype=real.dtype, device=device)

            # Create an image that follows the normal distribution.
            noise = torch.randn([label_size, 100, 1, 1], device=device)

            # Initialize the discriminator gradient.
            D.zero_grad()

            # Calculate the loss of the real image.
            output = D(real)
            # print(f"Output: {output.shape}")
            loss_D_real = criterion(output, real_label)
            loss_D_real.backward()

            # Calculate the loss of the fake image.
            D_real = output.mean().item()

            fake = G(noise)
            output = D(fake.detach())
            loss_D_fake = criterion(output, fake_label)
            loss_D_fake.backward()

            D_fake1 = output.mean().item()

            # Update the model.
            loss_D = loss_D_real + loss_D_fake
            optimizerD.step()

            # Initialize the generator gradient.
            G.zero_grad()

            # Calculate the loss of the fake image.
            output = D(fake)

            # Adversarial loss.
            loss_G = criterion(output, real_label)
            loss_G.backward()
            optimizerG.step()

            D_fake2 = output.mean().item()

            # Write the loss during training to the log and TensorBoard.

            iters = epoch * batches + i + 1

            writer.add_scalar('Train_Adversarial/Discriminator_Loss', loss_D.item(), iters)
            writer.add_scalar('Train_Adversarial/Generator_Loss', loss_G.item(), iters)
            writer.add_scalar('Train_Adversarial/D(x)', D_real, iters)
            writer.add_scalar('Train_Adversarial/D(G(z))_1', D_fake1, iters)
            writer.add_scalar('Train_Adversarial/D(G(z))_2', D_fake2, iters)

            if (i + 1) % args.print_freq == 0 or (i + 1) == batches:
                elapsed_epoch = time.time() - epoch_start_time
                elapsed_batch = time.time() - batch_start_time
                
                avg_time = elapsed_batch / args.print_freq
                min_time_epoch = elapsed_epoch // 60
                sec_time_epoch = elapsed_epoch % 60
                min_time_batch = elapsed_batch // 60
                sec_time_batch = elapsed_batch % 60
                print(
                    f"Epoch {epoch + 1}/{epochs}".center(50, "#")
                    + "\n"
                    + f"Batch {i + 1}/{batches}".center(50, "=")
                    + "\n"
                    + f"\tD Loss: {loss_D.item():.6f} G Loss: {loss_G.item():.6f}\n"
                    + f"\tD(Real): {D_real:.6f}\n"
                    + f"\tD(Fake1)/D(Fake2): {D_fake1:.6f}/{D_fake2:.6f}\n"
                    + f"\tElapsed Time (Epoch): {min_time_epoch:.0f}m {sec_time_epoch:.0f}s\n"
                    + f"\tElapsed Time (Batch): {min_time_batch:.0f}m {sec_time_batch:.0f}s\n"
                    + f"\tAverage Time: {avg_time:.0f}s\n"
                )

                data_to_append = [
                    epoch + 1, epochs, i + 1, batches,
                    loss_D.item(), loss_G.item(), D_real, D_fake1, D_fake2,
                    elapsed_epoch,
                    elapsed_batch,
                    avg_time
                ]

                append_to_csv(metrics_path, data_to_append)


        ############################

        torch.save(D.state_dict(), os.path.join(samples_dir, f"d_epoch{epoch + 1}.pth"))
        torch.save(G.state_dict(), os.path.join(samples_dir, f"g_epoch{epoch + 1}.pth"))

        # Each epoch validates the model once.
        with torch.no_grad():
            # Switch model to eval mode.
            G.eval()
            fake = G(fixed_noise).detach()
            torchvision.utils.save_image(fake, os.path.join(samples_dir, f"epoch_{epoch + 1}.bmp"), normalize=True)

    
    # Save the weight of the model under the last Epoch in this stage.
    torch.save(D.state_dict(), os.path.join(result_dir, "d-last.pth"))
    torch.save(G.state_dict(), os.path.join(result_dir, "g-last.pth"))
   
def validate(args, writer):

    if args.device == 'cuda':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if torch.cuda.is_available():
            print(f'Using {torch.cuda.get_device_name(0)}')
        else:
            print('Using CPU')
    else:
        device = torch.device('cpu')

    eval_dir = os.path.join("GAN/results", "test", args.exp_name)

    if args.color:
        if args.image_size == 64:
            G = Generator_64_RGB().to(device)
        elif args.image_size == 512:
            G = Generator_512_RGB().to(device)
    else:
        if args.image_size == 64:
            G = Generator_64_G().to(device)
        elif args.image_size == 512:
            raise ValueError('Image size 512 not supported for grayscale images')
    model_path = f'results/{args.exp_name}/g-last.pth'

    if os.path.exists(eval_dir):
        shutil.rmtree(eval_dir)
    os.makedirs(eval_dir)

    G.load_state_dict(torch.load(model_path))

    G.eval()
    # G.half()

    with torch.no_grad():
        for i in range(100):
            fixed_noise = torch.randn([64, 100, 1, 1], device=device)
            # fixed_noise = fixed_noise.half()
            image = G(fixed_noise)
            torchvision.utils.save_image(image, os.path.join(eval_dir, f"{i:03d}.bmp"))
            print(f"The {i + 1:03d} image is being created using the model...")


if __name__ == '__main__':
    args = parse_args()
    writer = SummaryWriter(os.path.join("GAN/samples",  "logs", args.exp_name))
    cudnn.benchmark = args.benchmark
    torch.manual_seed(args.seed)

    if args.mode == 'train':
        print("Training...")
        train(args, writer)
    elif args.mode == 'validate':
        print("Validating...") 
        validate(args, writer)
    else:
        raise ValueError(f'Invalid mode: {args.mode}')