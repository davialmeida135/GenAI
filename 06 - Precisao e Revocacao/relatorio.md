


WGan
https://colab.research.google.com/drive/1RKtlyr4-RQ3W7IN90xUopGL_8UmBC1pF#scrollTo=BxoQbkRS-OT_
```python
class Gerador(nn.Module):

    def __init__(self, z_dim=100):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(z_dim, 7*7*128),
            nn.Unflatten(dim=1, unflattened_size=(128, 7, 7)),
            nn.BatchNorm2d(128),
            nn.ConvTranspose2d(128, 64, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.ReLU(True),
            nn.BatchNorm2d(64),
            nn.ConvTranspose2d(64, 1, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        return self.net(x)
```

GanFC
https://colab.research.google.com/drive/1WDo8sdP0ODHq2odkd3qWYNtMRXKQgH6y#scrollTo=5-DQ-Q-h47K2
```python
gen_model = nn.Sequential(
    nn.Linear(z_size, gen_hidden_size),
    nn.LeakyReLU(),
    nn.Linear(gen_hidden_size, np.prod(image_size)),
    nn.Tanh()
)
```

DCGan
https://colab.research.google.com/drive/15tVRsKXBJ5LvAzt_5CfLzbXaYypod3Zm#scrollTo=BxoQbkRS-OT_
```python
class Gerador(nn.Module):
    def __init__(self, z_dim=100):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(z_dim, 7*7*128),
            nn.Unflatten(dim=1, unflattened_size=(128, 7, 7)),
            nn.BatchNorm2d(128),
            nn.ConvTranspose2d(128, 64, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.ReLU(True),
            nn.BatchNorm2d(64),
            nn.ConvTranspose2d(64, 1, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.Tanh()
        )

    def forward(self, z):
        return self.net(z)
```