# Large Language Model Parameter-Efficient Fine-Tuning (PEFT)

## Low-Rank Adaptation (LoRA)
LoRA freezes pre-trained model weights and injects trainable rank decomposition matrices into each layer of the Transformer architecture, reducing trainable parameters by up to 99%.

## Quantized LoRA (QLoRA)
QLoRA introduces 4-bit NormalFloat (NF4) quantization, double quantization, and paged optimizers to fine-tune 70B parameter models on a single 48GB GPU.

## Hyperparameter Selection
- Learning Rate: 2e-4 for rank 16 LoRA adaptations.
- Warmup Ratio: 0.03 with cosine learning rate decay.
- Batch Size: Gradient accumulation steps of 4 or 8 to achieve effective batch size of 64.

## Evaluation & Overfitting Mitigation
Evaluate validation loss every 100 steps. Apply early stopping if evaluation loss diverges from training loss for more than 3 consecutive evaluation cycles.
