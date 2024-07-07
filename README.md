# Generative PokeSets

## Card Generation

```batch
python ./generate_set.py --image_size 512 --color
```

## Generative Adversarial Networks (GANs)

### Training

```batch
python .\main_gan.py --device cpu --mode train --exp_name exp_draft --data_dir data/cards --batch_size 64 --image_size 512 --epochs 100 --color True
```

Resume training from a checkpoint:

```batch
python .\main_gan.py --device cpu --mode train --exp_name exp_512_rgb_100_suite --data_dir data/cards_512_RGB --batch_size 64 --image_size 512 --epochs 150 --color True --print_freq 1 --resume True --resume_epoch 51 --resume_dir GAN/samples/exp_512_rgb_100_suite
```

![Sample GIF](https://github.com/aristo6253/generative-pokeset/GAN/train_128_rgb.gif)

### Validation

```batch
python .\main.py --device cpu --mode validate --exp_name exp001 --data_dir data/cards --batch_size 64 --image_size 512 --color True
```

Early Validation:

```batch
python .\main_gan.py --mode validate --exp_name exp_64_rgb_100 --val_early --val_epoch 55 --image_size 64 --color
```

## Diffusion Models

## Variational Autoencoders (VAEs)
