import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
import argparse
from tqdm import tqdm
from datetime import datetime

from .generator import Generator
from .discriminator import Discriminator
from .losses import PerceptualLoss
from ..data.dataset import get_dataloaders


class Pix2PixTrainer:
    def __init__(self, config):
        self.config = config

        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        # Initialize models
        self.generator = Generator(
            in_channels=config['input_channels'],
            out_channels=config['output_channels'],
            features=config['generator_features']
        ).to(self.device)

        self.discriminator = Discriminator(
            input_channels=config['input_channels'],
            output_channels=config['output_channels'],
            features=config['discriminator_features']
        ).to(self.device)

        # Initialize optimizers
        self.optimizer_G = optim.Adam(
            self.generator.parameters(),
            lr=config['lr_generator'],
            betas=(config['beta1'], 0.999)
        )

        self.optimizer_D = optim.Adam(
            self.discriminator.parameters(),
            lr=config['lr_discriminator'],
            betas=(config['beta1'], 0.999)
        )

        # Loss functions
        self.criterion_GAN = nn.BCEWithLogitsLoss()
        self.criterion_L1 = nn.L1Loss()
        self.criterion_perceptual = PerceptualLoss().to(self.device)

        # Learning rate schedulers
        self.scheduler_G = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer_G,
            T_max=config['num_epochs'],
            eta_min=1e-6
        )
        self.scheduler_D = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer_D,
            T_max=config['num_epochs'],
            eta_min=1e-6
        )

        # Data loaders
        self.train_loader, self.val_loader = get_dataloaders(
            train_root=config['train_root'],
            val_root=config['val_root'],
            batch_size=config['batch_size'],
            num_workers=config['num_workers']
        )

        # Store individual dataset loaders for per-dataset visualization
        self.val_datasets_individual = self._get_individual_val_loaders(config)

        # Setup checkpoint
        self.checkpoint_dir = Path(config['checkpoint_dir'])
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Setup tensorboard
        self.writer = SummaryWriter(log_dir=config['log_dir'])
        print(f"\n{'='*60}")
        print(f"TensorBoard logging to: {config['log_dir']}")
        print(f"To view: tensorboard --logdir={config['log_dir']}")
        print(f"{'='*60}\n")

        self.current_epoch = 0
        self.global_step = 0

    def _get_individual_val_loaders(self, config):
        """Create individual dataloaders for each validation dataset."""
        from ..data.dataset import SatGenDataset
        from pathlib import Path

        val_root = Path(config['val_root'])
        individual_loaders = {}

        for subdir in val_root.iterdir():
            if subdir.is_dir() and not subdir.name.startswith('.'):
                img_dir = subdir / 'images'
                gt_dir = subdir / 'gt'
                if img_dir.exists() and gt_dir.exists():
                    dataset = SatGenDataset(images_dir=img_dir, labels_dir=gt_dir, augment=False)
                    loader = torch.utils.data.DataLoader(
                        dataset,
                        batch_size=4,  # Get 4 samples per dataset
                        shuffle=True,
                        num_workers=0,
                        pin_memory=True
                    )
                    individual_loaders[subdir.name] = loader

        return individual_loaders

    def train_epoch(self):
        self.generator.train()
        self.discriminator.train()

        total_g_loss = 0
        total_d_loss = 0

        pbar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch}")

        for batch_idx, (input_imgs, target_imgs) in enumerate(pbar):
            input_imgs = input_imgs.to(self.device)  # [B, 2, 256, 256]
            target_imgs = target_imgs.to(self.device)  # [B, 3, 256, 256]

            # batch_size = input_imgs.size(0)

            # Train Discriminator
            self.optimizer_D.zero_grad()

            fake_imgs = self.generator(input_imgs)

            pred_real = self.discriminator(input_imgs, target_imgs)
            target_real = torch.ones_like(pred_real) * 0.9
            loss_real = self.criterion_GAN(pred_real, target_real)

            pred_fake = self.discriminator(input_imgs, fake_imgs.detach())
            target_fake = torch.zeros_like(pred_fake)
            loss_fake = self.criterion_GAN(pred_fake, target_fake)

            loss_D = (loss_real + loss_fake) * 0.5
            loss_D.backward()
            self.optimizer_D.step()

            # Train Generator
            self.optimizer_G.zero_grad()
            fake_imgs = self.generator(input_imgs)

            pred_fake = self.discriminator(input_imgs, fake_imgs)
            target_real = torch.ones_like(pred_fake)
            loss_GAN = self.criterion_GAN(pred_fake, target_real)

            loss_L1 = self.criterion_L1(fake_imgs, target_imgs)
            loss_perceptual = self.criterion_perceptual(fake_imgs, target_imgs)

            loss_G = self.config['lambda_gan'] * loss_GAN + self.config['lambda_l1'] * loss_L1 + self.config['lambda_perceptual'] * loss_perceptual
            loss_G.backward()
            self.optimizer_G.step()

            # Update metrics
            total_g_loss += loss_G.item()
            total_d_loss += loss_D.item()

            # tqdm
            pbar.set_postfix({
                'G_loss': f"{loss_G.item():.4f}",
                'D_loss': f"{loss_D.item():.4f}",
                'L1': f"{loss_L1.item():.4f}",
                'Perceptual': f"{loss_perceptual.item():.4f}"
            })

            # Tensorboard logs
            if batch_idx % self.config['log_interval'] == 0:
                self.writer.add_scalar('Train/Generator_Loss', loss_G.item(), self.global_step)
                self.writer.add_scalar('Train/Discriminator_Loss', loss_D.item(), self.global_step)
                self.writer.add_scalar('Train/L1_Loss', loss_L1.item(), self.global_step)
                self.writer.add_scalar('Train/GAN_Loss', loss_GAN.item(), self.global_step)
                self.writer.add_scalar('Train/Perceptual_Loss', loss_perceptual.item(), self.global_step)

            self.global_step += 1

        avg_g_loss = total_g_loss / len(self.train_loader)
        avg_d_loss = total_d_loss / len(self.train_loader)

        return avg_g_loss, avg_d_loss

    @torch.no_grad()
    def validate(self):
        self.generator.eval()
        self.discriminator.eval()

        total_g_loss = 0
        total_d_loss = 0
        total_l1_loss = 0
        total_perceptual_loss = 0

        for input_imgs, target_imgs in tqdm(self.val_loader, desc="Validating"):
            input_imgs = input_imgs.to(self.device)
            target_imgs = target_imgs.to(self.device)

            fake_imgs = self.generator(input_imgs)

            pred_real = self.discriminator(input_imgs, target_imgs)
            pred_fake = self.discriminator(input_imgs, fake_imgs)

            loss_real = self.criterion_GAN(pred_real, torch.ones_like(pred_real))
            loss_fake = self.criterion_GAN(pred_fake, torch.zeros_like(pred_fake))
            loss_D = (loss_real + loss_fake) * 0.5

            loss_GAN = self.criterion_GAN(pred_fake, torch.ones_like(pred_fake))
            loss_L1 = self.criterion_L1(fake_imgs, target_imgs)
            loss_perceptual = self.criterion_perceptual(fake_imgs, target_imgs)
            loss_G = loss_GAN + self.config['lambda_l1'] * loss_L1 + self.config['lambda_perceptual'] * loss_perceptual

            total_g_loss += loss_G.item()
            total_d_loss += loss_D.item()
            total_l1_loss += loss_L1.item()
            total_perceptual_loss += loss_perceptual.item()

        avg_g_loss = total_g_loss / len(self.val_loader)
        avg_d_loss = total_d_loss / len(self.val_loader)
        avg_l1_loss = total_l1_loss / len(self.val_loader)
        avg_perceptual_loss = total_perceptual_loss / len(self.val_loader)

        # Log to tensorboard
        self.writer.add_scalar('Val/Generator_Loss', avg_g_loss, self.current_epoch)
        self.writer.add_scalar('Val/Discriminator_Loss', avg_d_loss, self.current_epoch)
        self.writer.add_scalar('Val/L1_Loss', avg_l1_loss, self.current_epoch)
        self.writer.add_scalar('Val/Perceptual_Loss', avg_perceptual_loss, self.current_epoch)

        # Log sample images from each dataset separately
        if self.current_epoch % self.config['image_log_interval'] == 0:
            print(f"\n  Logging validation images to TensorBoard...")

            for dataset_name, loader in self.val_datasets_individual.items():
                try:
                    input_imgs, target_imgs = next(iter(loader))
                    input_imgs = input_imgs[:4].to(self.device)  # Up to 4 images per dataset
                    target_imgs = target_imgs[:4].to(self.device)
                    fake_imgs = self.generator(input_imgs)

                    # Denormalize images from [-1, 1] to [0, 1]
                    fake_imgs = (fake_imgs + 1) / 2
                    target_imgs = (target_imgs + 1) / 2
                    input_imgs = (input_imgs + 1) / 2

                    # Log with dataset-specific tags
                    self.writer.add_images(f'Val_{dataset_name}/Input_Buildings', input_imgs[:, 0:1, :, :], self.current_epoch)
                    self.writer.add_images(f'Val_{dataset_name}/Input_Roads', input_imgs[:, 1:2, :, :], self.current_epoch)
                    self.writer.add_images(f'Val_{dataset_name}/Generated', fake_imgs, self.current_epoch)
                    self.writer.add_images(f'Val_{dataset_name}/Target', target_imgs, self.current_epoch)

                    print(f"    ✓ Logged 4 samples from {dataset_name}")
                except Exception as e:
                    print(f"    ✗ Failed to log images from {dataset_name}: {e}")

        return avg_g_loss, avg_d_loss, avg_l1_loss, avg_perceptual_loss

    def save_checkpoint(self, filename='checkpoint.pth'):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': self.current_epoch,
            'global_step': self.global_step,
            'generator_state_dict': self.generator.state_dict(),
            'discriminator_state_dict': self.discriminator.state_dict(),
            'optimizer_G_state_dict': self.optimizer_G.state_dict(),
            'optimizer_D_state_dict': self.optimizer_D.state_dict(),
            'scheduler_G_state_dict': self.scheduler_G.state_dict(),
            'scheduler_D_state_dict': self.scheduler_D.state_dict(),
            'config': self.config
        }
        save_path = self.checkpoint_dir / filename
        torch.save(checkpoint, save_path)
        print(f"Checkpoint saved to {save_path}")

    def load_checkpoint(self, checkpoint_path):
        """Load model checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.generator.load_state_dict(checkpoint['generator_state_dict'])
        self.discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
        self.optimizer_G.load_state_dict(checkpoint['optimizer_G_state_dict'])
        self.optimizer_D.load_state_dict(checkpoint['optimizer_D_state_dict'])

        # Load scheduler states if available
        if 'scheduler_G_state_dict' in checkpoint:
            self.scheduler_G.load_state_dict(checkpoint['scheduler_G_state_dict'])
        if 'scheduler_D_state_dict' in checkpoint:
            self.scheduler_D.load_state_dict(checkpoint['scheduler_D_state_dict'])

        self.current_epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        print(f"Checkpoint loaded from {checkpoint_path}")

    def train(self):
        """Main training loop."""
        print(f"\n{'='*60}")
        print(f"TRAINING CONFIGURATION")
        print(f"{'='*60}")
        print(f"Device: {self.device}")
        print(f"Training samples: {len(self.train_loader.dataset)}")
        print(f"Validation samples: {len(self.val_loader.dataset)}")

        # Show per-dataset validation counts
        if self.val_datasets_individual:
            print(f"\nValidation datasets:")
            for dataset_name, loader in self.val_datasets_individual.items():
                print(f"  - {dataset_name}: {len(loader.dataset)} images")

        print(f"\nBatch size: {self.config['batch_size']}")
        print(f"Learning rate: G={self.config['lr_generator']}, D={self.config['lr_discriminator']}")
        print(f"Loss weights: L1={self.config['lambda_l1']}, Perceptual={self.config['lambda_perceptual']}")
        print(f"{'='*60}\n")

        best_val_loss = float('inf')

        for epoch in range(self.current_epoch + 1, self.config['num_epochs']):
            self.current_epoch = epoch

            # Train
            train_g_loss, train_d_loss = self.train_epoch()

            # Validate
            val_g_loss, val_d_loss, val_l1_loss, val_perceptual_loss = self.validate()

            # Step learning rate schedulers
            self.scheduler_G.step()
            self.scheduler_D.step()

            # Log learning rates
            self.writer.add_scalar('LR/Generator', self.optimizer_G.param_groups[0]['lr'], epoch)
            self.writer.add_scalar('LR/Discriminator', self.optimizer_D.param_groups[0]['lr'], epoch)

            print(f"\nEpoch {epoch}/{self.config['num_epochs']-1}")
            print(f"  Train - G: {train_g_loss:.4f}, D: {train_d_loss:.4f}")
            print(f"  Val   - G: {val_g_loss:.4f}, D: {val_d_loss:.4f}, L1: {val_l1_loss:.4f}, Perceptual: {val_perceptual_loss:.4f}")
            print(f"  LR    - G: {self.optimizer_G.param_groups[0]['lr']:.6f}, D: {self.optimizer_D.param_groups[0]['lr']:.6f}")

            # Save checkpoint
            if (epoch + 1) % self.config['save_interval'] == 0:
                self.save_checkpoint(f'checkpoint_epoch_{epoch}.pth')

            if val_g_loss < best_val_loss:
                best_val_loss = val_g_loss
                self.save_checkpoint('best_model.pth')
                print(f"  New best model saved (val_loss: {val_g_loss:.4f})")

        # Save final model
        self.save_checkpoint('final_model.pth')
        self.writer.close()
        print("Training completed!")


def get_default_config():
    """Get default training configuration."""
    return {
        # Model architecture
        'input_channels': 2,
        'output_channels': 3,
        'generator_features': 64,
        'discriminator_features': [64, 128, 256, 512],

        # Training hyperparameters
        'num_epochs': 200,
        'batch_size': 1,
        'lr_generator': 0.0002,
        'lr_discriminator': 0.0002,
        'beta1': 0.5,
        'lambda_l1': 50,
        'lambda_perceptual': 0,
        'lambda_gan': 0.5,

        'train_root': 'data/processed/train_sam3',
        'val_root': 'data/processed/test_sam3',

        # Logging and checkpoints
        'checkpoint_dir': 'checkpoints',
        'log_dir': f'runs/pix2pix_{datetime.now().strftime("%Y%m%d_%H%M%S")}',
        'log_interval': 10, # every nth BATCH
        'save_interval': 1,  # every nth EPOCH
        'image_log_interval': 1,  # every nth EPOCH

        'num_workers': 4 # Only useful for dataloaders
    }


def main():
    parser = argparse.ArgumentParser(description='Train Pix2Pix model for road generation')
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=0.0003)
    parser.add_argument('--lambda-l1', type=int, default=100)
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints')
    parser.add_argument('--resume', type=str, default=None)

    args = parser.parse_args()

    config = get_default_config()
    config['num_epochs'] = args.epochs
    config['batch_size'] = args.batch_size
    config['lr_generator'] = args.lr
    config['lr_discriminator'] = args.lr
    config['lambda_l1'] = args.lambda_l1
    config['checkpoint_dir'] = args.checkpoint_dir

    trainer = Pix2PixTrainer(config)

    if args.resume:
        trainer.load_checkpoint(args.resume)
        print("Resuming from checkpoint:", args.resume)

    trainer.train()


if __name__ == '__main__':
    main()
