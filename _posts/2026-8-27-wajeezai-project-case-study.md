---
layout: post
title: "WajeezAI: Multimodal Lecture Summarization"
date: 2026-08-27
tags: [NLP, Speech Recognition, Multimodal AI, Projects, PyTorch]
---

# WajeezAI

> An AI-powered system that transforms university lectures into structured notes by combining the professor's speech with visual lecture content.

<!-- Hero image / project screenshot -->

![WajeezAI](/assets/img/wajeezai_simplified.png)

## Overview

WajeezAI was our graduation project, developed to help students turn lengthy university
lectures into concise and structured notes.

The system processes both **lecture audio** and **visual content such as slides**, 
extracts meaningful information from each modality, and combines them to generate
structured lecture notes.

<!-- Short demo video -->

## Demo

{% include youtube.html id="zkBxyzC7OFQ" %}

---

## The Problem

Taking notes during a lecture requires students to divide their attention between
listening, understanding, and writing.

This becomes even more difficult when important information is distributed across
different modalities:

- The professor explains concepts verbally.
- Slides contain definitions, diagrams, and examples.
- Whiteboards may contain additional explanations.
- Some information discussed by the professor may not appear on the slides at all.

We wanted to explore whether an AI system could combine these sources and produce
useful lecture notes automatically.

---

## What We Built

WajeezAI processes a lecture through several stages:

1. Convert the professor's speech into text.
2. Extract textual information from lecture images.
3. Represent the extracted content using semantic embeddings.
4. Associate spoken content with relevant visual content.
5. Combine the multimodal information.
6. Generate structured lecture notes.
7. Export the resulting notes into a usable document format.

<!-- Architecture diagram -->

![WajeezAI Pipeline](/assets/img/wajeezai_hero.png)

---
<!-- I am here -->
## How It Works 
### 1. Speech Recognition

The lecture environment makes speech recognition challenging, particularly due to the **Syrian dialect**, background noise from students, and frequent switching between **Arabic and English technical terms**.

A major challenge was the lack of publicly available data that matched our target environment. Most existing Arabic speech datasets do not adequately represent the combination of **Syrian dialect**, **university lectures**, **background noise**, and **Arabic–English code-switching** found in our recordings. In particular, technical lectures frequently contain English terms embedded naturally within Arabic sentences, making this a different setting from standard Arabic speech recognition benchmarks.

Because of this, we decided to build our own domain-specific dataset. We collected approximately **13 hours of real university lecture audio** covering different lectures and recording conditions. We then manually transcribed the recordings, preserving the way technical terms were actually spoken rather than converting them into purely Arabic text.

<!-- Dataset collection / annotation images -->

![Lecture Dataset](/assets/img/speaker_distribution_subplots.png)

![Lecture Dataset](/assets/img/speaker_distribution.png)

This process gave us training data that better represented the conditions in which WajeezAI would actually be used.

We then fine-tuned **Whisper Large with LoRA** on this dataset to adapt the model to the Syrian dialect, lecture environment, and frequent Arabic–English code-switching.

For example, the baseline Whisper model produced:

> هلق رح ناخذ ديكريز اند كونكر

while our fine-tuned model produced:

> هلق رح ناخذ decrease and conquer

The fine-tuned model was therefore better able to preserve English technical terminology within Arabic speech instead of transliterating the English words into Arabic.



<table dir="rtl">
  <thead>
    <tr>
      <th>المحتوى المنطوق</th>
      <th>Whisper الأساسي</th>
      <th>Whisper بعد Fine-tuning</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>هلق رح ناخذ <bdi>decrease and conquer</bdi></td>
      <td>هلق رح ناخذ ديكريز اند كونكر</td>
      <td>هلق رح ناخذ <bdi>decrease and conquer</bdi></td>
    </tr>
    <tr>
      <td>حنحكي اليوم عن ال <bdi>Fuzzy Logic</bdi></td>
      <td>سنحكي اليوم عن الفزي لوجيك</td>
      <td>حنحكي اليوم عن ال <bdi>Fuzzy Logic</bdi></td>
    </tr>
    <tr>
      <td>بعدين منعمل <bdi>forward propagation</bdi></td>
      <td>بعدين منعمل فوروارد بروباغيشن</td>
      <td>بعدين منعمل <bdi>forward propagation</bdi></td>
    </tr>
    <tr>
      <td>هاد ال<bdi>algorithm</bdi> بيعتمد على <bdi>divide and conquer</bdi></td>
      <td>هاد الالغوريتم يعتمد على ديڤايد اند كونكر</td>
      <td>هذا ال<bdi>algorithm</bdi> بيعتمد على <bdi>divide and conquer</bdi></td>
    </tr>
  </tbody>
</table>


Finally, the fine-tuned model produces **timestamped segments of the professor's speech**, which are later used in the multimodal processing pipeline to connect spoken content with the corresponding visual lecture material.

![asr comparison](/assets/img/asr_comparison.png)

> NOTE!

> Due to privacy considerations and the presence of identifiable voices and lecture content, the dataset is currently private and is not publicly available.

### 2. Visual Content Extraction

Lecture slides and photos taken by students of whiteboards contain much more than plain text. They can include **diagrams, flowcharts, sketches, mathematical expressions, and other visualizations** that are important for understanding the lecture.

This made a traditional OCR-based approach insufficient. We needed a system that could not only extract text, but also **understand and describe visual elements** within an image.

We initially experimented with several popular OCR solutions, including **PaddleOCR** and **EasyOCR**. However, they struggled significantly with **Arabic handwritten text**, and their performance was also unsatisfactory on some relatively simple typed Arabic text. More importantly, conventional OCR could not provide meaningful descriptions of diagrams, sketches, or other non-textual elements.

To address both problems, we moved toward using a **Vision-Language Model (VLM)** capable of processing text and visual information together.

Because our available computational resources were limited, running a large VLM locally was not practical. We therefore used **Gemma 4 31B** through Google's API, allowing us to take advantage of its visual understanding capabilities without requiring the computational resources to host the model ourselves.

The model showed strong capabilities in both text extraction and visual understanding. In particular, it could extract complex Arabic and English text, describe visual elements, and **identify visualizations using bounding boxes.**

These bounding boxes allowed us to locate and crop diagrams, flowcharts, and other relevant visual elements from the original image and include them alongside the generated text in the final lecture document.

![Visual Content Extraction](/assets/img/vlm_output.png)


The previous example demonstrates how the VLM processes a lecture slide containing both textual and visual information.

Rather than treating the slide as a flat image, the model produces two complementary forms of output:

1. **Structured textual content** — The text on the slide is extracted while preserving its organization into headings, bullet points, and mathematical expressions.
2. **Visual descriptions with bounding boxes** — Non-textual elements are identified and described along with their coordinates within the original image.

For example, from a slide discussing **Ring and Mesh Network Topologies**, the model identified the relevant textual content, including their advantages and disadvantages, as well as the mathematical formula:

$$L = \frac{n(n-1)}{2}$$

At the same time, it detected two visual elements and provided their bounding boxes:

* **Ring topology diagram:** `[88, 85, 210, 358]`
* **Mesh topology connection formula:** `[321, 2, 398, 184]`

The coordinates are represented as bounding boxes in the format:

```text
[x₁, y₁, x₂, y₂]
```

These coordinates allow us to locate the corresponding regions in the original image and **crop the visual elements automatically**. The extracted descriptions and cropped visualizations can then be incorporated into the generated lecture document alongside the corresponding textual content.

This was particularly useful for slides where important information was conveyed through **diagrams, flowcharts, mathematical expressions, or sketches**, which would otherwise be difficult to capture using conventional OCR alone.


### 3. Semantic Alignment

We initially explored using semantic embeddings to determine which parts of the
professor's speech corresponded to each lecture image.

Each ASR segment and each image representation was embedded into the same semantic
space.

We then used similarity search to retrieve the most relevant speech segments for
each image.

<!-- Similarity diagram -->

![Semantic Alignment](/assets/img/wajeezai_semantic.png)

---
## A Challenge We Encountered

Our initial alignment approach based on semantic similarity performed reasonably well,
but we found that it was missing an important source of information: **time**.

In a typical lecture, a professor is likely to discuss a slide, whiteboard, or other
visual element around the time it is being presented. This temporal relationship can
provide a strong additional signal when determining which part of the lecture
corresponds to a particular image.

However, this introduced another problem:

> **How do we know when an image was taken relative to the lecture audio?**

A photo taken on a student's phone has its own timestamp, but that timestamp does not
necessarily tell us where the image falls within the lecture recording. The phone's
clock could be different from the recording device's clock, and the image may also
have been taken some time after the student started recording.

### Synchronizing Images with the Lecture

To solve this, we built a simple mobile application that students can use to record
the lecture and capture images during the recording.

When the student starts recording, the application establishes **time zero** for
the lecture. Whenever the student takes a picture using the application, we record
the elapsed time since the beginning of the recording along with the image.

## Improving the Alignment

For each image, we calculated its temporal relationship with every ASR chunk based
on the image timestamp and the time interval of the corresponding speech segment.

Instead of treating all ASR chunks equally, we assigned a higher temporal similarity
to chunks occurring closer to the time at which the image was captured.

We modeled this relationship using a **Gaussian function**, producing a smooth temporal
weight that decreases as the distance between the image and speech increases.

![Temporal Similarity](/assets/img/temporal_alignment.png)

The final alignment score combines both sources of information:

$$
S_{final} = \alpha S_{semantic} + \beta S_{temporal}
$$

where:

- $$S_{semantic}$$ represents the semantic similarity between the image and ASR chunk.
- $$S_{temporal}$$ represents their temporal proximity.
- $$\alpha$$ and $$\beta$$ control the contribution of each signal.

The temporal similarity is calculated as:

$$
S_{temporal} =
e^{-\frac{(t_{image}-t_{chunk})^2}{2\sigma^2}}
$$

where $$t_{image}$$ is the timestamp of the image and $$t_{chunk}$$ is the center
timestamp of the ASR segment:

$$
t_{chunk} = \frac{t_{start}+t_{end}}{2}
$$

This allowed the system to favor speech that is both **semantically relevant and
temporally close** to the visual content.

Importantly, temporal similarity does not replace semantic similarity. Instead, the
two signals complement each other: semantic similarity helps determine **what the
professor is talking about**, while temporal similarity helps determine **when they
are talking about it**.


## Organizing the Aligned Content

After the alignment stage, each lecture image is associated with the relevant ASR chunks from the professor's speech. However, these aligned fragments are not yet suitable as readable lecture material.

To transform them into a coherent document, we use **Gemini 2.5 Flash-Lite** to organize the aligned visual and textual information. The model receives each image together with its associated speech segments and restructures the content into a clear, readable form while preserving the information conveyed during the lecture.

This stage focuses on **organization and readability** rather than alignment. The resulting content forms the basis of the final structured lecture document.

![Temporal Similarity](/assets/img/wajeez_notes.png)

## What I Learned

Working on WajeezAI taught me several things that were difficult to appreciate from
working with individual models alone.

### Multimodal Systems Are More Than Combining Models

Having a speech recognition model and a vision model does not automatically produce
a good multimodal system. The outputs need to be aligned and interpreted together.

### Similarity Does Not Mean Relevance

One of the most important lessons was that semantic similarity is relative.

A search system can always find the *closest* result, even when none of the available
results are actually relevant.

### End-to-End Engineering

The project also required integrating AI models into a complete application rather
than evaluating them independently.

---

## Technologies

**AI & Machine Learning**

* PyTorch
* Speech Recognition
* NLP
* Semantic Search
* Multimodal AI
* OCR
* Generative AI
* Embeddings

**Application**

* Flutter
* FastAPI
* Streamlit

---

## What I Would Improve

There are several directions I would explore in a future version:

* Develop more robust speech-slide alignment.
* Improve detection of speech without visual counterparts.
* Improve the quality and structure of generated notes.
