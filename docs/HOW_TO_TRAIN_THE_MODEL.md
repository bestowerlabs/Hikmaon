# How to Train the Hikmaon Deepfake Model — Step by Step

**Who this guide is for:** anyone on the team, including non-programmers. If you
can copy-and-paste commands into a terminal and wait, you can train the model.
Every step is explained in plain language, with the exact command to run and
what you should see.

**What you are doing, in one sentence:** you show the AI thousands of examples
of *real* faces and *fake* (deepfake) faces, it learns the difference, and you
end up with a single file (`hikmaonnet.onnx`) that Hikmaon uses to detect
deepfakes.

**How long it takes:** roughly a day of downloading data, a few hours of the
computer preparing it, and several hours to ~1 day of the computer training
(this part is hands-off — you start it and walk away).

---

## Answer first: the FaceForensics++ question

**Yes — FaceForensics++ is exactly the right dataset. But an important warning:**

The **GitHub page** for FaceForensics++ (github.com/ondyari/FaceForensics) does
**NOT** contain the actual videos. It only contains a *download script* and
instructions. This trips everyone up. If your team downloaded the GitHub
repository, they got the **scripts, not the data** — the real videos are much
larger (hundreds of GB) and are only released after you request access.

**The correct way to get the data:**

1. Go to the FaceForensics++ GitHub page and open the access request form it
   links to (a Google Form / e-mail request agreeing to their terms — the data
   is for research use).
2. The authors e-mail you back a **download script** (`download-FaceForensics.py`)
   with an access token, usually within a day or two.
3. You **run that script** to download the videos. For example:
   ```
   python download-FaceForensics.py /data/ffpp -d all -c c23 -t videos
   ```
   This downloads the real videos and the deepfake videos into `/data/ffpp`.

So: downloading the GitHub repo was a reasonable first move, but it is only the
starting point — the team still needs to **request access and run the download
script** to get the actual videos. Everything below assumes you have the videos
on disk.

> **You do not have to use only FaceForensics++.** It is the best starting
> dataset. For a stronger model, later add DFDC, Celeb-DF, and some
> AI-generated (diffusion) images. The steps are identical — just more folders.

---

## The big picture (5 stages)

```
  1. SET UP        install the software (once)
        ↓
  2. GET DATA      download real + fake videos (FaceForensics++)
        ↓
  3. PREPARE       turn videos into picture "frames"  ← one command per folder
        ↓
  4. TRAIN         the computer learns (start it, walk away)
        ↓
  5. DEPLOY        copy one file into Hikmaon — done
```

Keep this picture in mind. Each stage below is one section.

---

## What kind of computer you need

Training an AI needs a computer with a **GPU** (a graphics card made by NVIDIA).
A normal laptop will technically work but could take *weeks*; a GPU finishes in
hours.

- **Recommended:** a cloud GPU machine (rent by the hour) or an office
  workstation with an NVIDIA GPU (e.g. RTX 3090/4090, A100).
- **Cloud options (rent one, no purchase):** Google Cloud, AWS, Lambda Labs,
  RunPod, Paperspace. Pick a machine described as having an "NVIDIA GPU" with at
  least **16 GB of GPU memory**. Ask their support "I need a GPU machine for
  PyTorch deep learning" if unsure.
- **Disk space:** at least **500 GB** free (the videos and frames are large).

You do **not** need a GPU for stages 1–3 (setup, download, prepare) — only for
stage 4 (training). But it's simplest to do everything on the GPU machine.

---

## Stage 1 — Set up the software (do this once)

Open a terminal on the training machine and run these commands one block at a
time. Lines starting with `#` are explanations — you don't type those.

```bash
# 1. Get the Hikmaon code
git clone https://github.com/bestowerlabs/Hikmaon.git
cd Hikmaon/backend

# 2. Create an isolated Python environment (keeps things tidy)
python -m venv .venv
source .venv/bin/activate          # on Windows: .venv\Scripts\activate

# 3. Install the training software
pip install -r ml/requirements.txt
```

**How to know it worked:** run this check —
```bash
python -c "import torch; print('GPU available:', torch.cuda.is_available())"
```
- If it prints `GPU available: True` → perfect, the GPU is ready.
- If it prints `False` → training will still run but very slowly. On a cloud GPU
  machine it should say `True`; if not, the GPU drivers aren't installed — ask
  the cloud provider's support to enable "CUDA / NVIDIA drivers", or pick an
  image labelled "Deep Learning" / "PyTorch".

---

## Stage 2 — Get the data (real + fake videos)

Follow the FaceForensics++ access process described at the top. When finished,
you should have a folder structure roughly like this:

```
/data/ffpp/
    original_sequences/       ← REAL videos
        ...
    manipulated_sequences/    ← FAKE (deepfake) videos
        Deepfakes/
        Face2Face/
        FaceSwap/
        NeuralTextures/
```

The exact folder names depend on the download options, but the key idea is: you
have **one set of folders with real videos** and **several folders with different
kinds of fakes**. Note down these folder paths — you'll use them next.

> **Tip:** Keep each *type* of fake separate (Deepfakes, Face2Face, …). Later
> this lets us prove the model works on kinds of fakes it has never seen — the
> single most important quality check.

---

## Stage 3 — Prepare the data (turn videos into frames)

The AI learns from still pictures, not videos. This stage chops each video into
a handful of picture "frames." **We built a tool that does this in one command.**

Run it **once per folder** — once for the real videos, once for each fake type:

```bash
# REAL videos → frames
python -m ml.prepare_dataset --videos /data/ffpp/original_sequences --out /data/frames/real

# FAKE videos → frames (one command per fake type)
python -m ml.prepare_dataset --videos /data/ffpp/manipulated_sequences/Deepfakes      --out /data/frames/deepfakes
python -m ml.prepare_dataset --videos /data/ffpp/manipulated_sequences/Face2Face      --out /data/frames/face2face
python -m ml.prepare_dataset --videos /data/ffpp/manipulated_sequences/FaceSwap       --out /data/frames/faceswap
python -m ml.prepare_dataset --videos /data/ffpp/manipulated_sequences/NeuralTextures --out /data/frames/neuraltextures
```

**What you'll see:** a line per video, e.g. `[12/1000] 033.mp4: 40 frames`, then a
summary like `Done: 40000 frames from 1000 videos`. Each command may take a
while (it's processing every video). It's safe to stop and re-run — already-done
videos are skipped.

**How to know it worked:** open one of the `--out` folders (e.g.
`/data/frames/real`) in a file browser. You should see many subfolders, each
containing `frame_0001.png`, `frame_0002.png`, … — actual face pictures.

> **Optional (advanced, better accuracy):** the frames above are whole video
> frames. Cropping tightly to the face improves accuracy. This needs a face
> detector and is an enhancement — **skip it for your first model**. Train on
> whole frames first to get a working detector, then revisit face-cropping.

---

## Stage 4 — Train the model

Two commands: build a "manifest" (a checklist of all your pictures), then start
training.

### 4a. Build the manifest (30 seconds)

The manifest is a spreadsheet-like file that lists every picture, whether it's
real or fake, and which "study group" (train / validate / test) it belongs to.
Our tool builds it correctly — including the crucial rule that pictures from the
same video never get split across groups (which would make the model look better
than it really is).

```bash
python -m ml.make_manifest \
    --real /data/frames/real \
    --fake deepfakes=/data/frames/deepfakes \
    --fake face2face=/data/frames/face2face \
    --fake faceswap=/data/frames/faceswap \
    --fake neuraltextures=/data/frames/neuraltextures \
    --holdout neuraltextures \
    --out /data/manifest.csv
```

Read that as: "real pictures are here; here are four kinds of fakes; **hold one
kind (neuraltextures) completely out of training** so we can later test on a fake
type the model never saw; save the checklist to `/data/manifest.csv`."

**What you'll see:** a table showing how many pictures went to train/val/test per
category, then `wrote /data/manifest.csv`. If it complains "No images found,"
double-check the folder paths.

### 4b. Start training (hours — walk away)

```bash
python -m ml.train --manifest /data/manifest.csv --out runs/v1 --epochs 30
```

**What happens:** the computer looks at all the pictures 30 times over (each pass
is one "epoch"), gradually learning. After each epoch it prints a line like:

```
{"epoch": 0, "train_loss": 0.42, "val_auc": 0.91, "val_acc": 0.86, ...}
```

The number that matters is **`val_auc`** — how well it tells real from fake on
pictures it isn't training on. **1.0 is perfect; 0.5 is random guessing.** You
want to see it climb toward **0.95+** over the epochs.

- Let it run. It automatically **saves the best version** to `runs/v1/best.pt`
  whenever it improves. You'll see `saved best.pt (val_auc=0.96)`.
- You can close the terminal only if you started it so it keeps running (see the
  "let it run overnight" tip below). Otherwise leave the terminal open.
- When it finishes you'll see `done. best val AUC: 0.9x`.

> **Let it run overnight without staying logged in:** prefix the command with
> `nohup` and add `&`, and it keeps running after you disconnect:
> ```bash
> nohup python -m ml.train --manifest /data/manifest.csv --out runs/v1 --epochs 30 > train.log 2>&1 &
> ```
> Check progress anytime with: `tail -f train.log` (press Ctrl-C to stop
> watching — this does **not** stop the training).

> **If it stops with "out of memory":** your GPU is smaller. Make the batches
> smaller by adding `--batch-size 32` (or `16`) to the train command.

### 4c. Check quality and calibrate (a few minutes)

```bash
python -m ml.evaluate --manifest /data/manifest.csv --checkpoint runs/v1/best.pt --split test --fit-temperature
```

This prints a report card. Look at:
- **`auc`** overall — aim for **0.95+**.
- **`per_generator`** — the AUC for *each* fake type, **including the held-out one
  (`neuraltextures`)**. This is the honest test: a good model scores high even on
  the fake type it never trained on. If the held-out score is much lower (say
  below 0.8), the model memorized rather than learned — add more variety of fakes
  and train again.
- `--fit-temperature` also tunes the model so its confidence numbers are
  trustworthy (a "90% fake" really means about 90%).

---

## Stage 5 — Deploy (put the model into Hikmaon)

Convert the trained model into the single file Hikmaon uses, then point Hikmaon
at it.

```bash
# 1. Export to the deployment file
python -m ml.export --checkpoint runs/v1/best.pt --out hikmaonnet.onnx

# 2. Tell Hikmaon where it is, and start the server
HIKMAON_MODEL_PATH=hikmaonnet.onnx uvicorn app.main:app
```

**How to know it's live:** open `http://your-server:8000/api/model/status` in a
browser (or ask a teammate to). It should say `"neural_detector": "loaded"`.
That's it — Hikmaon is now using **your** trained model to detect deepfakes, in
both images and video.

> Copy `hikmaonnet.onnx` to your production server and set the same
> `HIKMAON_MODEL_PATH` there. You do **not** need the GPU or the training
> software in production — just this one file.

---

## Keeping the model strong over time

Deepfake methods keep improving, so a model trained today slowly becomes less
accurate against next year's fakes. Plan to **re-train every few months** with
newer fake examples added to the folders. The process is identical — add new
folders to the `make_manifest` command and run stages 3–5 again. Always keep at
least one fake type held out (`--holdout`) so you can honestly measure whether
the new model generalizes.

---

## Quick reference (all commands in order)

```bash
# Setup (once)
cd Hikmaon/backend && python -m venv .venv && source .venv/bin/activate
pip install -r ml/requirements.txt

# Prepare (once per folder)
python -m ml.prepare_dataset --videos <videos_folder> --out <frames_folder>

# Manifest
python -m ml.make_manifest --real <real_frames> --fake NAME=<fake_frames> [--fake ...] --holdout NAME --out /data/manifest.csv

# Train
python -m ml.train --manifest /data/manifest.csv --out runs/v1 --epochs 30

# Evaluate
python -m ml.evaluate --manifest /data/manifest.csv --checkpoint runs/v1/best.pt --split test --fit-temperature

# Export + deploy
python -m ml.export --checkpoint runs/v1/best.pt --out hikmaonnet.onnx
HIKMAON_MODEL_PATH=hikmaonnet.onnx uvicorn app.main:app
```

---

## Troubleshooting (plain-language)

| You see… | What it means | What to do |
|---|---|---|
| `GPU available: False` | Training will be very slow | Use a GPU machine / "Deep Learning" cloud image |
| `ffmpeg not found` | The video tool is missing | `pip install imageio-ffmpeg` |
| `No videos found under …` | Wrong folder path | Check the path points at the folder that actually contains the videos |
| `No images found` (manifest) | Frames weren't extracted there | Re-run Stage 3 for that folder; check `--out` path matches |
| `CUDA out of memory` | GPU too small for the batch | Add `--batch-size 32` (or `16`) to the train command |
| `val_auc` stuck near 0.5 | Model isn't learning | Check real vs fake folders aren't swapped; ensure both have frames |
| held-out generator AUC is low | Model doesn't generalize | Add more variety of fake types and re-train |

If you get stuck, save the exact command you ran and the full error message —
that's what an engineer needs to help quickly.
