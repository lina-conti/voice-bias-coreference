# voice-bias-coreference
This repository contains the code associated with the LREC 2026 paper "Voice, Bias, and Coreference: An Interpretability Study of Gender in Speech Translation". 

The notebooks in this directory reproduce the analyses in Sections 6–8 of the paper and the example figure (Figure 1).

---

## Prerequisites

### Data and models

- **MuST-SHE** (evaluation benchmark) — [Bentivogli et al., 2020](https://aclanthology.org/2020.acl-main.619/)
- **MuST-SHE POS extension** — [Savoldi et al., 2022](https://aclanthology.org/2022.acl-long.127/)
- **MuST-C** (training corpus, target-language side) — [Di Gangi et al., 2019](https://aclanthology.org/N19-1202/)
- Model checkpoints:
  - Transformer ([Wang et al., 2020](https://aclanthology.org/2020.aacl-demo.6/)) — [download](https://github.com/facebookresearch/fairseq/blob/main/examples/speech_to_text/docs/mustc_example.md)
  - Conformer ([Papi et al., 2024](https://aclanthology.org/2024.acl-long.200/)) — [download](https://github.com/hlt-mt/FBK-fairseq/blob/master/fbk_works/BUGFREE_CONFORMER.md)

### Environment

Install [FBK-fairseq](https://github.com/hlt-mt/FBK-fairseq) and its dependencies. All scripts below are run from within that repository.

---

## Generating the Required Files

The notebooks depend on several files that must be generated before running them. Steps 0–4 follow [`CONTRASTIVE_SPES.md`](https://github.com/hlt-mt/FBK-fairseq/blob/master/fbk_works/CONTRASTIVE_SPES.md) (in the FBK-fairseq repository). Steps 5–6 are described here.

### Steps 0–4: Preprocessing and explanation pipeline

Follow [`CONTRASTIVE_SPES.md`](https://github.com/hlt-mt/FBK-fairseq/blob/master/fbk_works/CONTRASTIVE_SPES.md) to generate:

| Variable | Description |
|---|---|
| `$gender_explanation_tsv` | Model hypotheses annotated with gender term information (step 1) |
| `$orig_probs` | Token-level probabilities from the full ST model (step 2) |
| `$explanations_path` | Raw feature attribution heatmaps — `gender_explanations.h5` (step 3) |
| `$deletion_output` | Hypotheses at each spectrogram deletion step, with `--perc-interval 1 --max-percent 20` (step 4) |

The fbank feature arrays in `fbank/` (one `.npy` file per sample) are also produced by step 0.

### Step 5: ILM probabilities

The internal language model (ILM) is approximated by replacing the encoder output with a zero vector ([Variani et al., 2020](https://ieeexplore.ieee.org/document/9054689); [Meng et al., 2021](https://ieeexplore.ieee.org/document/9688093)). Run the following to compute token-level ILM probabilities:

```bash
# Create the dummy encoder output: a zero array of size equal to the
# encoder hidden dimension (512 for the Transformer and Conformers used here)
python3 -c "import numpy as np; np.save('dummy_encoder_outs.npy', np.zeros(512))"

# Run the ILM script (same arguments as step 2 of CONTRASTIVE_SPES.md)
python examples/speech_to_text/scripts/xai/get_probs_from_ilm.py ${data_dir} \
    --gen-subset ${gender_explanation_tsv} \
    --user-dir examples/speech_to_text \
    --max-tokens 10000 \
    --model-overrides "{'batch_unsafe_relative_shift':False, 'load_pretrained_encoder_from':None}" \
    --max-source-positions 7000 \
    --config-yaml ${explain_yaml_config} \
    --task speech_to_text_genderxai \
    --explanation-task gender \
    --prefix-size 1 \
    --path ${model_path} \
    --save-file ${ilm_probs} \
    --dummy-encoder-outs dummy_encoder_outs.npy
```

For the Conformer, add `--criterion ctc_multi_loss --underlying-criterion label_smoothed_cross_entropy --label-smoothing 0.1`, replace the task with `speech_to_text_genderxai_ctc`, and omit `--prefix-size`.

### Step 6: Forced word alignment (Gentle)

The time-dimension analysis in `spectrogram_analysis.ipynb` requires word-level alignments. Run [Gentle](https://github.com/lowerquality/gentle) on the MuST-SHE audio files to produce one JSON file per utterance, named `<base_id>` (e.g. `it-0742`), and collect them in a single directory (`JSON_FOLDER` in the notebook).

---

## Notebooks

Run the notebooks in the order listed. All analysis notebooks (§6–8) read from `summary_dataframe.tsv`, which is the output of `prepare_df_for_analysis.ipynb`.

| Notebook | Paper section | Description | Key inputs |
|---|---|---|---|
| `prepare_df_for_analysis.ipynb` | Prerequisite | Collects all per-sample information into `summary_dataframe.tsv` | `$gender_explanation_tsv`, `$orig_probs`, `$ilm_probs`, `$deletion_output`, MuST-SHE POS extension, MuST-C training data, SPM vocabulary |
| `training_data.ipynb` | §6, Tables 1 & 7 | Training data gender prevalence analysis | `summary_dataframe.tsv` |
| `ilm_analysis.ipynb` | §7, Tables 2 & 8–10 | ILM masculine preference and model–ILM correlation | `summary_dataframe.tsv` |
| `spectrogram_analysis.ipynb` | §8, Tables 3 & 4, Figures 2 & 3 | Frequency- and time-dimension attribution analysis | `summary_dataframe.tsv`, `gender_explanations.h5`, `normalized_explanations.h5`, Gentle alignment JSONs |
| `example_explanation.ipynb` | Figure 1 | Normalises explanations and renders the three-panel example figure | `summary_dataframe.tsv`, `gender_explanations.h5`, fbank features, Gentle alignment JSONs |

> **Note on `normalized_explanations.h5`:** `example_explanation.ipynb` normalises the raw heatmaps and saves them to `normalized_explanations.h5`. This file is then read by `spectrogram_analysis.ipynb`. Run at least the normalisation section of `example_explanation.ipynb` before running `spectrogram_analysis.ipynb`.

Each analysis notebook is configured for a single language (`LANG`) and model (`MODEL`). To reproduce all results in the paper, re-run for each of the three language pairs (es, fr, it) and both model architectures (transformer, conformer).

---

## Citation

```bibtex
@article{conti2025voice,
  title={Voice, Bias, and Coreference: An Interpretability Study of Gender in Speech Translation},
  author={Conti, Lina and Fucci, Dennis and Gaido, Marco and Negri, Matteo and Wisniewski, Guillaume and Bentivogli, Luisa},
  journal={arXiv preprint arXiv:2511.21517},
  year={2025}
}
```
