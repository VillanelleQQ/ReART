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

<p align="center">
<img src="./pipeline.png" width="95%">
</p>

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

## Code structure

```text
data/          # annotation filtering and structured visual records
retrieval/     # field-wise reference retrieval
generation/    # prompt construction and initial generation
evaluation/    # local AAS evaluation and threshold gate
refinement/    # diagnosis, planning, routing, and constrained editing
utils/         # shared utilities
```

## Installation

```bash
conda env create -f environment.yml
conda activate affectiveart
```

Set the API key:

```bash
# Linux/macOS
export OPENAI_API_KEY="YOUR_API_KEY"

# PowerShell
$env:OPENAI_API_KEY="YOUR_API_KEY"
```

All commands below are run from the repository root.

## Complete example: caption → generation → refinement

The example uses:

```text
Ukiyo-e scene of a sailboat and small rowboat on rippling water before Mount Fuji, with travelers near a stone embankment in the foreground, delicate linework, muted colors, and a calm atmosphere.
```

The input record is provided at `examples/ukiyoe_sailboat_caption.json`, with sample ID `ukiyoe_sailboat_fuji`.

### 1. Download EmoArt-130k

Download the complete dataset:

```bash
hf download printblue/EmoArt-130k \
  --repo-type dataset \
  --local-dir datasets/EmoArt-130k
```

This example only uses the **Ukiyo-e** subset. Extract it and select its annotations:

```bash
mkdir -p datasets/EmoArt-130k/images

tar -xzf "datasets/EmoArt-130k/Ukiyo-e.tar.gz" \
  -C datasets/EmoArt-130k/images

python -m data.filter_annotations \
  --input datasets/EmoArt-130k/Annotation.json \
  --output datasets/EmoArt-130k/ukiyoe_annotations.json \
  --field image_path \
  --include "Ukiyo-e"
```

### 2. Structure the caption

```bash
python -m data.structure_records \
  --input examples/ukiyoe_sailboat_caption.json \
  --output runs/ukiyoe/structured_test.json \
  --checkpoint-dir runs/ukiyoe/checkpoints/test \
  --batch-size 1 \
  --workers 1 \
  --model gpt-5.4
```

### 3. Structure the Ukiyo-e reference pool

```bash
python -m data.structure_records \
  --input datasets/EmoArt-130k/ukiyoe_annotations.json \
  --output datasets/EmoArt-130k/ukiyoe_structured.json \
  --checkpoint-dir runs/ukiyoe/checkpoints/references \
  --batch-size 20 \
  --workers 8 \
  --model gpt-5.4
```

### 4. Retrieve field-aligned references

```bash
python -m retrieval.fieldwise \
  --test runs/ukiyoe/structured_test.json \
  --reference datasets/EmoArt-130k/ukiyoe_structured.json \
  --output runs/ukiyoe/retrieval_top10.json \
  --model-name BAAI/bge-m3 \
  --device cuda \
  --top-k 10
```

Use `--device cpu` when CUDA is unavailable.

### 5. Build the generation prompt

```bash
python -m generation.build_prompts \
  --test runs/ukiyoe/structured_test.json \
  --primary-topk runs/ukiyoe/retrieval_top10.json \
  --primary-ref datasets/EmoArt-130k/ukiyoe_structured.json \
  --primary-label "Ukiyo-e" \
  --primary-k 5 \
  --sample-id ukiyoe_sailboat_fuji \
  --output-dir runs/ukiyoe/generation
```

### 6. Generate the initial image

```bash
python -m generation.generate \
  --prompt-file runs/ukiyoe/generation/ukiyoe_sailboat_fuji/prompt.txt \
  --prompt-bundle-file runs/ukiyoe/generation/ukiyoe_sailboat_fuji/prompt_bundle.json \
  --output-dir runs/ukiyoe/generation/ukiyoe_sailboat_fuji/gpt-image-2 \
  --image-name ukiyoe_sailboat_fuji.png \
  --model gpt-image-2 \
  --use-references \
  --image-root datasets/EmoArt-130k/images \
  --annotation-json datasets/EmoArt-130k/ukiyoe_annotations.json
```

The initial image is saved to:

```text
runs/ukiyoe/generation/ukiyoe_sailboat_fuji/gpt-image-2/ukiyoe_sailboat_fuji.png
```

### 7. Evaluate the initial image

Prepare the evaluation input:

```bash
python -m evaluation.prepare \
  --test-json runs/ukiyoe/structured_test.json \
  --image-path runs/ukiyoe/generation/ukiyoe_sailboat_fuji/gpt-image-2/ukiyoe_sailboat_fuji.png \
  --sample-id ukiyoe_sailboat_fuji \
  --flat-output-root runs/ukiyoe/aas_initial_input
```

Run the local AAS evaluator:

```bash
python -m evaluation.aas \
  --run-dir runs/ukiyoe/aas_initial_input \
  --output-dir runs/ukiyoe/aas_initial \
  --sample-id ukiyoe_sailboat_fuji \
  --provider openai \
  --judge-model gpt-5.4
```

Check the content, style, and attribute scores with a threshold of 9:

```bash
python -m evaluation.gate \
  --result runs/ukiyoe/aas_initial/ukiyoe_sailboat_fuji_aas.json \
  --threshold 9
```

If the result is `PASS`, use the initial image as the final output. If the result is `REFINE`, continue below.

### 8. Build the refinement plan

```bash
python -m refinement.pipeline \
  --sample-id ukiyoe_sailboat_fuji \
  --structured-test-json runs/ukiyoe/structured_test.json \
  --retrieval-json runs/ukiyoe/retrieval_top10.json \
  --annotation-json datasets/EmoArt-130k/ukiyoe_structured.json \
  --annotation-json datasets/EmoArt-130k/ukiyoe_annotations.json \
  --current-image runs/ukiyoe/generation/ukiyoe_sailboat_fuji/gpt-image-2/ukiyoe_sailboat_fuji.png \
  --image-root datasets/EmoArt-130k/images \
  --model gpt-5.4 \
  --output-root runs/ukiyoe/refine/iter1
```

### 9. Run constrained editing

```bash
python -m refinement.edit \
  --packet-dir runs/ukiyoe/refine/iter1/ukiyoe_sailboat_fuji \
  --model gpt-image-2
```

The refined image is saved to:

```text
runs/ukiyoe/refine/iter1/ukiyoe_sailboat_fuji/gpt-image-2-refine/refined.png
```

### 10. Re-evaluate the refined image

```bash
python -m evaluation.prepare \
  --test-json runs/ukiyoe/structured_test.json \
  --image-path runs/ukiyoe/refine/iter1/ukiyoe_sailboat_fuji/gpt-image-2-refine/refined.png \
  --sample-id ukiyoe_sailboat_fuji \
  --flat-output-root runs/ukiyoe/aas_refine_1_input

python -m evaluation.aas \
  --run-dir runs/ukiyoe/aas_refine_1_input \
  --output-dir runs/ukiyoe/aas_refine_1 \
  --sample-id ukiyoe_sailboat_fuji \
  --provider openai \
  --judge-model gpt-5.4

python -m evaluation.gate \
  --result runs/ukiyoe/aas_refine_1/ukiyoe_sailboat_fuji_aas.json \
  --threshold 9
```

If refinement is still required, use the preceding `refined.png` as the next `--current-image`, change the output directory to `iter2`, and repeat. The pipeline stops when all three scores reach 9 or after four refinement iterations.
