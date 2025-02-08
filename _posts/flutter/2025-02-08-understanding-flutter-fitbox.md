---
layout: post
title: Understanding Flutter BoxFit
date: 2025-02-08
tags: [Flutter]
---

While working on Images in Flutter you are going to encounter the fit property which takes a BoxFit enum.
There are many types of BoxFit and they can be overwhelming at the start and every one of them works in a certain way so understanding them correctly will help you to choose the right option.
In this post I will try to compare between each one of them.

## BoxFit types:
- fill
- contain
- cover
- fitWidth
- fitHeight
- none
- scaleDown


### fill

fill as the name suggests it tries to completely fill the container and it displays the entire image.
It does not crop the image.

Using fill can distort the image(change the aspect ratio which leads to image appears stretched, squished, or otherwise altered from its original proportions).

If the image size is 300x300 (widthxheight) this means the aspect ratio(the ratio of the width to the height of an image or screen) is 1:1

and the container it is trying to fill is 1000X400 this means the aspect ratio is 2.5:1 which is equivalent to 5:2

so the image needs to upscale to 1000x400 which changes its aspect ratio from 1:1 to 2.5:1


### cover 

cover also fills the entire container but it preserves the aspect ratio and it may not display the entire image.
if the image is 200x500 then aspect ratio is 0.4:1 which is equivalent  to 2:5  and the container is 1000x400 aspect ratio is 5:2

so the image is already filling the height but it is not filling width so we need to upscale the image (and save its aspect ratio 2:5) so the image size would become 1000x2500 (upscaling five times)

so the result would be that the image will be clipped from the top and bottom.

another example image with size 300x300 container is 500x500 the entire image will appear because the container and image have the same aspect ratio 1:1 so the image just upscale 1.6667

### contain 

contain does not necessarily fill the entire container it can leave some spaces. the entire image appear and the aspect ratio is preserved.

if image is 1000x400 and container is 200x500 the image will downscale to 200x80

so the image will leave some empty spots on the top and bottom.

### fitWidth
preserve the aspect ratio it scales the image unit the image and container has the same width.

### fitHeight
preserve the aspect ratio it scales the image unit the image and container has the same height the width 

### none
No scaling is applied. The image stays at its original size inside the container. If the image is larger than the container, only part of it will be visible. If it's smaller, there will be empty space around it.

### scaleDown
It acts like BoxFit.none, but it scales down the image only if it's bigger than the container.
If the image fits inside the container, it is not resized.
If the image is larger than the container, it shrinks to fit while maintaining its aspect ratio.


### **Comparison Table**

| Fit Mode      | Keeps Aspect Ratio? | Can Crop? | Can Distort? | Fills Container? |
|--------------|-----------------|---------|------------|----------------|
| `fill`       | ❌ No           | ❌ No   | ✅ Yes      | ✅ Yes         |
| `cover`      | ✅ Yes          | ✅ Yes  | ❌ No       | ✅ Yes         |
| `contain`    | ✅ Yes          | ❌ No   | ❌ No       | ❌ No (may leave space) |
| `fitWidth`   | ✅ Yes          | ✅ Yes  | ❌ No       | ✅ Yes (only width) |
| `fitHeight`  | ✅ Yes          | ✅ Yes  | ❌ No       | ✅ Yes (only height) |
| `none`       | ✅ Yes          | ✅ Yes  | ❌ No       | ❌ No (no scaling) |
| `scaleDown`  | ✅ Yes          | ❌ No   | ❌ No       | ❌ No (only scales down) |


