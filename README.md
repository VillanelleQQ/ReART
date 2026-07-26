<p align="center">

<h1 align="center">
ReART: Reference-Guided Retrieval and Refinement for Emotion-Aware Art Generation
</h1>
<p align="center">
    <strong>
    Qianqian Tang<sup>*</sup>
    ·
    Jiayi Gao<sup>*</sup>
    ·
    Ting Lei
    ·
    Yang Liu<sup>†</sup>
    </strong>
</p>

## Introduction

Emotion-aware artistic image generation requires models to simultaneously
satisfy semantic content, artistic style, and emotional expression.

We present **ReART**, a **Reference-Guided Retrieval and Refinement framework**
for emotion-aware art generation.

ReART addresses the challenge that artistic captions often contain
underspecified visual attributes by introducing a structured retrieval and
refinement pipeline.

Our framework consists of two stages:

**Stage I: Retrieval-Guided Artistic Generation**

We decompose captions and reference artworks into structured visual fields and
perform field-wise retrieval from style-specific reference pools. Retrieved
references are assigned different visual roles and incorporated with structured
prompts to guide the initial generation.


**Stage II: AAS-Driven Constrained Refinement**

We introduce an Attribute Alignment Score (AAS)-driven refinement loop that
automatically diagnoses misaligned dimensions, constructs repair plans, routes
relevant references, and performs constrained editing while preserving
correctly generated regions.


ReART achieves competitive performance in the **AffectiveArt 2026 Grand
Challenge Track 1**, ranking **2nd overall** with a perfect **AAS score of
1.00**.

## Method Overview

![[pipeline4.png]]

Given an artistic caption, ReART follows a two-stage pipeline:

1. **Structured Visual Representation**

   Captions and reference artworks are converted into structured visual records.

2. **Field-wise Reference Retrieval**

   Visual references are retrieved independently for different visual dimensions,
   including subject, composition, brushwork, material, color, and mood.

3. **Reference-Guided Generation**

   Retrieved references are combined with structured prompts to generate the
   initial artwork.

4. **Constrained Refinement**

   Generated images are evaluated with AAS, and failed dimensions are refined
   through targeted reference routing and controlled editing.



## Installation

The codebase was tested with:

- Python 3.10
- PyTorch
- CUDA 12.x


Create the environment:

```bash
conda env create -f environment.yml

conda activate affectiveart
```


## Inference

The released code implements the main ReART pipeline:

### 1. Structured Representation

Convert captions and annotations into structured visual fields.

```bash
python gpt_caption_compressor.py
```

### 2. Reference Retrieval

Retrieve field-aligned references:

```bash
python fieldwise_retrieval_simple.py
```

### 3. Reference-Guided Generation

Generate initial artworks:

```bash
python generate_gpt_image2.py
```

### 4. Refinement

Perform AAS-driven constrained refinement:

```bash
python run_refine_pipeline.py
```








