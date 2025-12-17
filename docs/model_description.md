## Opis modeli
|       Nazwa modelu       |    Data    |  Model |             Dane             | Obrazek | Wielkosć zbioru | Batch | Loss |   Compute     |
|:------------------------:|:----------:|:------:|:----------------------------:|---------|:---------------:|:-----:|:----:|:-------------:|
| 128x128_2707_simple_unet | 27.07.2025 | UNetV1 |        Large covid CT        | 128x128 |      16235      |   32  |  MSE | GTX 1650      |  
| 128x128_2807_simple_unet | 28.07.2025 | UNetV1 | Large covid CT i augmentacje | 128x128 |      81175      |   32  |  MSE | GTX 1650      |
| 128x128_3007_simple_unet | 30.07.2025 | UNetV1 | Large covid CT i augmentacje | 128x128 |      81175      |   32  |  L1  | GTX 1650      |
| 362x362_2908_unetv2      | 29.08.2025 | UNetV2 |           Lodopab            | 362x362 |      35820      |   16  |  MSE | RTX 5090 (5$) |
| 256x256_0212_pix2pix | 02.12.2025 | Pix2Pix_256_V1 | Anonimized256 | 256x256 | 206143 | 16 | D_loss + 300 * L1 | RTX 3090 |
| 128x128_1412_pix2pix | 14.12.2025 | Pix2Pix_128 | Anonimized128 | 128x128 | 206143 | 8 | D_loss + 100 * L1 + 200 * MSE | RTX 3090 |