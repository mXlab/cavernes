# Experiments

## Experiment with caves_categories DCGAN 128x128

### Training

Label: categories_color128x128

Commandline:
```python3 main.py --dataroot ~/Documents/cavernes/datasets/caves_categories/ --dataset=folder --cuda --ngpu 4 --outf categories_color128x128 --imageSize 128 --ndf 64 --ngf 64 --batchSize 64 --niter 200```


### Generation

#### Drunk (epoch 199)

Label:
categories_color128x128_generated_199

Commandline:
```python3 generate.py --dataroot ~/Documents/cavernes/datasets/caves_categories/ --dataset=folder --cuda --ngpu 4 --netG experiments/categories_color128x128/netG_epoch_199.pth --imageSize 128 --ngf 64 --batchSize 64 --niter 2000 --outf experiments/categories_color128x128_generated_199```

#### Drunk (epoch 187)

Label:
categories_color128x128_generated_187

Commandline:
```python3 generate.py --dataroot ~/Documents/cavernes/datasets/caves_categories/ --dataset=folder --cuda --ngpu 4 --netG experiments/categories_color128x128/netG_epoch_187.pth --imageSize 128 --ngf 64 --batchSize 64 --niter 2000 --outf experiments/categories_color128x128_generated_187```

#### Multi-lerp (epoch 187)

Label:
categories_color128x128_generated_lerp_187

Commandline:
```python3 generate.py --dataroot ~/Documents/cavernes/datasets/caves_categories/ --dataset=folder --cuda --ngpu 4 --netG experiments/categories_color128x128/netG_epoch_187.pth --imageSize 128 --ngf 64 --batchSize 64 --niter 2000 --algo lerp --nlerp 100 --outf experiments/categories_color128x128_generated_lerp_187 --imageFormat jpg```
