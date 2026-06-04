# BLUEBERRY IMAGE CLASSIFICATION

- Comparing same model setups on vanilla dataset (6 classes)

| Model             | F1 macro | Loss |
| :---------------- | :------: | ----: |
| res18mod-norm-wsampl-aug | 70 | 0.29944 |
|         |   True   | 23.99 |

- Comparing model setups on expanded dataset (6 classes)

| Model                     | F1 macro | Loss  |
| :----------------         | :------: | ----: |
| baseline-norm-weightsampl | 73.6790  | 0.2790 |
| convnext-norm-wsampl      | 78.6892  | 0.2679 |
| res18-norm-wsampl-aug1    | 76.0066  | 0.2442 |
| convnext-norm-wsampl-aug1 | 81.0928  | 0.2090 |
| effv2s-norm-wsampl        | 77.1493  | 0.2498 |
| swintp4-norm-wsampl       | 77.2467  | 0.2423 |

NOTE: Transforms results only available in expanded bluebberies datasets(all models steadily above vanilla performance)

## Conclusions:

- ConvNext family seems to work better in this specific problem setup
  - tiny variant seems the best overall choice (comparable performance with base at ~ half the training time).
- 