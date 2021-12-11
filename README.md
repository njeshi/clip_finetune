#### Evaluate CLIP image feature extractors
____

CLIP: Combines general knowledge of image and text for a variety of zero-shot applications. 
It is a multi-modal vision and language model. It can be used for image-text similarity and for zero-shot image classification. 
CLIP uses a ViT like transformer to get visual features and a causal language model to get the text features. 
Both the text and visual features are then projected to a latent space with identical dimension. 
The dot product between the projected image and text features is then used as a similar score.

Repo structure:

- `clip` the 'clip' dir from the official implementation [openai/clip](https://github.com/openai/CLIP)
- `src` contains the clip wrappers, dataset parser and torch loaders
- `train_finetune` finetunes CLIP on different backbone models and training modes
- `zero_shot_classifier` tests CLIP zero shot capabilites on the chosen dataset
- `prompts_validation` tests accuracy for different templates
- `few_shot_learning` evaluates k-shot for different k values; uses the model itself to train
- `few_shot_linear_probing` trains an additional linear classifier on top of its visual encoder 
and follows a few-shot training manner; uses scikit-learn’s L-BFGS implementation with maximum 1,000 iterations.


</br>
CLIP Architecture
<img src="https://i.ibb.co/QHZ1ksT/Screen-Shot-2021-12-11-at-2-00-08-PM.png" />

</br>


Zero shot accuracy on Food101

| Model    | Top-1 Accuracy | Top-5 Accuracy |
| -------- | -------------- | -------------- |
| ViT-B/16 | 86.12          | 97.70          |
| ViT-B/32 | 80.69          | 95.84          |
| RN50     | 77.31          | 94.79          |
| RN101    | 80.61          | 95.86          |


</br>
Few-shot linear probing

Linear probe CLIP trains an additional linear classifier on top of its visual encoder and follows a few-shot training manner.

<img src="https://i.ibb.co/MfxK6Gy/vit16kshotlinearprobe.png"/> 

Zero-shot CLIP is as good as 64-shot linear probe CLIP.
