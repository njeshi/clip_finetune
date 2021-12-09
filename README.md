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
