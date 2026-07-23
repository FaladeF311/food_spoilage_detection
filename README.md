# Food Spoilage Detection: A Food Microbiology Lens

A computer vision project applying transfer learning to detect spoilage in food images (fruits, vegetables, dairy, and bread), with an emphasis on interpretability and food-safety-aware model design decisions.

## Motivation

Food spoilage is primarily a microbiological process — bacteria, yeasts, and molds break down food, causing visible changes in color, texture, and surface appearance (e.g., mold growth, browning from enzymatic and microbial activity, texture degradation). These visual cues are the same signals human inspectors rely on for quality control, and they're a natural fit for computer vision.

This project explores whether a lightweight CNN can learn to recognize these spoilage indicators, and — just as importantly — whether we can interpret *what* the model is actually looking at, to check if its reasoning aligns with real microbiological spoilage cues rather than incidental patterns in the data.

As a Microbiology graduate working in data science, this project sits at the intersection of both fields: understanding *why* food spoils informs how I evaluate whether the model is learning the right thing, not just whether it gets the right answer.

## Dataset

[Fresh and Spoiled Food Image Dataset](https://www.kaggle.com/datasets/maheen00shahid/fresh-and-spoiled-food-image-dataset) (Kaggle), containing 8 categories across 4 food types (fruits, vegetables, dairy, bread), each split into fresh/spoiled — ~2,715 images total.

For this project, the 8 categories were collapsed into a **binary classification task: Fresh vs. Spoiled**, prioritizing a clean, well-understood result over a more complex multi-class setup, given the project timeline. Multi-class (food type + freshness) is noted as future work.

**Known dataset limitations (identified during EDA):** a small number of images appeared potentially mislabeled or ambiguous (e.g., a "spoiled_dairy" image resembling a plain glass of milk, and a "spoiled_vegetables" image resembling moldy bread). This kind of noise is common in public datasets and wasn't cleaned out, in order to reflect realistic conditions — but it's worth flagging as a factor that may cap achievable accuracy.

## Approach

- **Model**: MobileNetV2 (pretrained on ImageNet), used as a frozen feature extractor with a custom classification head (GlobalAveragePooling → Dense(128, ReLU) → Dropout(0.3) → Dense(1, sigmoid)).
- **Why transfer learning**: ~2,700 images is too small to train a CNN from scratch reliably. Reusing ImageNet-pretrained visual features (edges, textures, shapes) and only training a small classification head on top avoids overfitting while still achieving strong performance.
- **Data augmentation**: rotation, width/height shift, horizontal flip, and zoom, applied to the training set to improve generalization given the limited dataset size.
- **Train/validation split**: 80/20 (2,171 train / 544 validation images).

## Results

| Metric | Value |
|---|---|
| Validation Accuracy | 82% |
| Fresh — Precision / Recall | 0.79 / 0.85 |
| Spoiled — Precision / Recall | 0.85 / 0.79 |

Training accuracy reached ~92% while validation stayed at ~82%, indicating mild overfitting — expected given the frozen base model and limited epochs, and reasonable for a first-pass model at this dataset size.

### Threshold tuning

Rather than defaulting to the standard 0.5 classification threshold, I evaluated multiple thresholds (0.5, 0.4, 0.35, 0.3), reasoning that **in a food safety context, a missed spoiled item (false negative) is more costly than a false alarm on a fresh item.**

| Threshold | Spoiled Recall | Fresh Precision | Accuracy |
|---|---|---|---|
| 0.5 | 0.79 | 0.79 | 0.82 |
| **0.4** | **0.85** | **0.82** | **0.82** |
| 0.35 | 0.87 | 0.84 | 0.82 |
| 0.3 | 0.88 | 0.84 | 0.81 |

**0.4 was selected** as the operating threshold: it improves spoiled-item recall from 79% to 85% (catching more genuinely spoiled food) while also slightly improving fresh precision, with no loss in overall accuracy — a deliberate precision-recall trade-off aligned with the real-world cost of errors in this domain.

## Interpretability (Grad-CAM)

To verify the model was learning genuine spoilage indicators rather than incidental correlations, I applied Grad-CAM to visualize which image regions most influenced each prediction.

**Findings:**
- On clear spoilage cases (e.g., a moldy/discolored carrot, a heavily spotted banana), the model's attention concentrated tightly on the actual decayed regions, with high confidence (0.99–1.00) — consistent with real visual spoilage cues.
- On a low-confidence, ambiguous case (a pepper still attached to its plant, showing early color change rather than clear spoilage), attention was diffuse and spread across background leaves rather than the fruit — suggesting a genuine labeling/dataset ambiguity rather than a model failure.
- On confidently correct fresh predictions (e.g., a bunch of bananas, confidence 0.00 for spoilage), attention was diffuse with no fixation point — consistent with the absence of any spoilage signal to key on.
- On a correct but lower-confidence fresh prediction (packaged paneer), attention concentrated on packaging/branding text rather than the food product itself — suggesting the model may be partially learning a contextual shortcut ("packaged = fresh") rather than relying purely on visual decay cues. This is flagged as a limitation and a target for improvement with more diverse training data that separates packaging context from food appearance.

## Limitations & Future Work

- Some dataset labels appear ambiguous or possibly mislabeled; a cleaned/re-verified dataset could improve reported accuracy.
- The model shows early signs of relying on contextual shortcuts (e.g., packaging) in some cases rather than purely visual spoilage indicators — addressing this would require more varied training examples.
- Current scope is binary (fresh/spoiled) across all food types combined; a natural extension is multi-class classification (food type × freshness) for more granular, food-specific spoilage detection.
- Fine-tuning deeper layers of MobileNetV2 (rather than keeping it fully frozen) could further improve accuracy, at the cost of longer training time and higher overfitting risk on this dataset size.

## Why This Project

Food spoilage detection sits within a broader theme I'm interested in: applying interpretable machine learning to health- and safety-relevant biological problems, informed by a microbiology background. This complements other work I've done in antimicrobial resistance prediction from genomic data, applying the same principle — that domain knowledge should guide not just model building, but model *evaluation*, especially in contexts where errors carry real consequences.
