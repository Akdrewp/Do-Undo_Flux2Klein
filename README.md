# Physics Aware Spatially Consistent Image Generation

## Introduction

Top performing VLM's have never been more impressive in image editing and generation. Given an image and an instruction to edit a particular part the highest end models complete it flawlessly.

But given an image and some action for some object the generation often falls short, losing out on the physical consistency within the world.

### Do-Undo

[Do-Undo](https://arxiv.org/pdf/2512.13609) is a paper that attempts to fix this issue with cycle consistency training. By forcing a model to learn an action and its exact physical reversal, the model learns to isolate the physical interaction without destroying the surrounding environment or hallucinating non-existent details.

The paper outlines:

I<sub>O</sub>: Initial image before action \
P<sub>F</sub>: Forward action prompt \
I<sub>F</sub>: Final image after action \
P<sub>F</sub>: reverse action prompt \
&Icirc;<sub>F</sub>: Generated forward image \
&Icirc;<sub>F</sub>: Generated reverse image

Training a VLM to recreate the final and original images from the forward and reverse prompt respectively. As well as adding a third metric to ensure consistency: I<sub>O</sub> ≈ &Icirc;<sub>F</sub>.

Using [EpicKitchens](https://epic-kitchens.github.io/) and the newest version open source VLM BAGEL to train to get good results in ensuring spatial consistency.

### My-Undo

The paper has good results but the EpicKitchens dataset is a bit skewed toward kitchen based objects and the newest BAGEL model is generally hard to run on lesser hardware.

I attempt to remedy this by using a (relatively) easy model to run Flux-2-Klein-9B as well as a more diverse dataset [HOI4D](https://hoi4d.github.io/) to get similiar results.

## Results

### Method

1. To obtain the image pairs from the dataset the annotations for each video were proccessed and all actions that have a corresponding undo action were obtained. In total ~6300 pairs for 6000 pairs for training set and 300 in the test set

2. Using [Qwen2-VL-2B](https://huggingface.co/Qwen/Qwen2-VL-2B) and the HOI4D dataset to create sets of start and final image as well as five inbetween frames and known object to create prompt for each pair of images.

3. With the prompts the Flux model was finetuned with LoRA with a loss function 

    L<sub>total</sub> = MSE(I<sub>o</sub>, &Icirc;<sub>o</sub>) + MSE(I<sub>f</sub>, &Icirc;<sub>f</sub>) + &lambda;<sub>c</sub> L1(I<sub>o</sub>, &Icirc;<sub>o</sub>)

    where &lambda;<sub>c</sub> is some consistency factor, the paper reccomends 0.5 but it was found that hallucinations happened too much probably from the model being smaller so 0.8 was chosen.

Using a RTX PRO 6000 WS and LoRA rank of 128 and an alpha of 64 was used in the training over 8 epochs. Although loss didn't decrease very much from epoch 3 onward. 

### Examples

First column is normal images, second is base Flux2-Klein, third is LoRA model.

![Toy Car Comparison](/images/red_car.png)
Forward: Pick up the red toy car with the right hand by gripping its handle and lifting it off the table. \
Reverse: Put down the red toy car with the right hand by lowering it until its base rests flat on the table.

Here it is seen the base model hallucinates a different car is the final reversed image while the finetuned version keeps the same car from the original image.

![Purple Trash Comparison](/images/purple_trash_can.png)
Forward: Pick up the purple and white object from the table by grasping its handle and lifting it up. \
Reverse: Put down the purple and white object on the table by placing it back down on the table.

Again the base model completely loses the fact that there is a trash can but the finetuned model keeps it and moves it.

![Blue Scissors Comparison](/images/blue_scissors.png)
Forward: Pick up the pair of blue scissors with the right hand by grasping the handles and lifting them off the table. \
Reverse: Put down the pair of blue scissors with the right hand by lowering them until their handles touch the table.

Here the base model hallucinates a completely different pair of scissors while the finetuned model keeps the same scissors

### Statistics
#### L1 difference over 100 test samples
FINAL THESIS BENCHMARK (100 STEPS) \
BASE MODEL | Avg L1 :  131.88 \
LORA MODEL | Avg L1 :  77.51

The finetuned model had L1 difference decrease by 41.2%

## Conclusion

This implementation demonstrates that physical cycle consistency can be successfully taught to smaller-scale generative models. By using the HOI4D dataset and a higher consistency weight (&lambda;<sub>c</sub> = 0.8), we mitigated common "object-drift" hallucinations. This proves that high-end hardware like that required for BAGEL isn't a strict barrier to achieving spatially consistent, physics-aware image manipulation.


