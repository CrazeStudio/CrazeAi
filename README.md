# CrazeAI

CrazeAI is a small decoder-only Transformer that can be trained with PyTorch and exported to a standalone native Android runtime. The Android app does **not** need Python or PyTorch after the model has been exported.

## Project layout

- `model.py` — Transformer architecture
- `tokenizer.py` — UTF-8 byte tokenizer (258 IDs: 256 bytes + BOS/EOS)
- `dataset.py` — instruction dataset loader with shifted next-token targets
- `train.py` — training/checkpointing
- `generate.py` — PyTorch reference inference
- `export.py` — converts `crazeai.pt` to native `.craze`
- `android/` — native C++ Android inference app
- `config.json` — model configuration
- `data.json` — training data

## 1. Train

Use Python/PyTorch only for training:

```bash
pip install -r requirements.txt
python train.py
```

This creates:

```text
crazeai.pt
```

For useful code generation, replace the small sample `data.json` with a large, clean dataset. The model architecture is only as useful as its training data.

## 2. Test the PyTorch model

```bash
python generate.py --prompt "Write a Python function that reverses a string."
```

## 3. Export the model

```bash
python export.py --checkpoint crazeai.pt --output crazeai.craze
```

The exporter writes a versioned `CRAZEAI1` binary containing the model configuration and FP32 tensors. The exported file is the only model file required by the Android runtime.

Do **not** commit `crazeai.pt` or `crazeai.craze` to Git if they are large; keep them as local release artifacts.

## 4. Build the Android runtime

Open the `android/` directory as an Android Studio project, make sure Android SDK 35 and the NDK/CMake components are installed, then build the debug APK.

The native library is implemented in:

```text
android/app/src/main/cpp/crazeai.cpp
```

It loads `.craze`, performs the same embedding → causal attention → MLP → layer norm → tied output projection used by the Python model, and samples the next token.

## 5. Run on Android

1. Install the debug APK.
2. Open CrazeAI.
3. Tap **Load .craze Model**.
4. Select the exported `crazeai.craze` file.
5. Wait for **Model loaded • Ready**.
6. Enter a prompt.
7. Tap **Generate**.

The app copies the selected model into its private app storage and runs inference through JNI/C++.

## Important

The current runtime is a correctness-first reference implementation. It recomputes the full context for every generated token, so it can be slow on phones. A KV-cache and FP16/INT8 quantized format should be added for a production-speed release.

The sample dataset is intentionally tiny. A tiny dataset will not make a large Transformer knowledgeable. For serious results, train on a large, deduplicated, high-quality dataset and validate it before export.

This is a new CrazeAI implementation and does not contain Meta model weights.
