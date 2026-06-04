# RC handoff — train the within-subcat ("proper") embeddings

This is everything you need to train the 6 proper-split embeddings on the Northeastern RC,
then bring the results back so I can finish the pipeline. The part3a–d notebooks are already
LoRA-enabled and **resumable** (per-epoch checkpoints, skip-completed, graceful stop before the
2-hour cap), so a session getting cut off is a non-event: just relaunch and it continues.

**Files in this folder (`~/rc_handoff/`):**
- `men-proper.tar.gz` (335M) — men-8-subcat-split-proper-embedding (loafers / fashion-sneakers / oxfords)
- `women-proper.tar.gz` (503M) — women-8-subcat-split-proper-embedding (pumps / flats / fashion-sneakers)

Each archive contains, per subcat: the filtered train/val parquets, the product images, and the
4 LoRA part3 notebooks. The image counts are baked in; nothing else is needed to train.

---

## Step 1 — Transfer to the RC (run LOCALLY; you'll be prompted for your RC password)

```fish
scp ~/rc_handoff/men-proper.tar.gz   <your-rc-username>@login.explorer.northeastern.edu:~/
scp ~/rc_handoff/women-proper.tar.gz <your-rc-username>@login.explorer.northeastern.edu:~/
```
(One archive each, so scp won't choke the way thousands of loose image files would.)

## Step 2 — Set up on the RC

```bash
ssh <your-rc-username>@login.explorer.northeastern.edu
cd ~
tar xzf men-proper.tar.gz
tar xzf women-proper.tar.gz
source activate demand_modeling      # NOT `conda activate`
pip install peft                     # REQUIRED for the LoRA adapters (one-time)
```

## Step 3 — Train the embeddings (OnDemand Jupyter, `demand_modeling` kernel)

Open Jupyter via OnDemand (rc.northeastern.edu) on a **gpu-interactive** session (2h, no
approval, pick H200 if the queue is short). For **each of the 6 subcats**, run all 4 notebooks
in `…/<subcat>/data_preparation/`:

```
00_part3a_train_txtimg_lag1.ipynb
00_part3b_train_txtimg_time_ind.ipynb
00_part3c_train_txt_lag1.ipynb
00_part3d_train_txt_time_ind.ipynb
```

- Run a notebook top-to-bottom. If the 2-hour cap kills the session, just **relaunch and re-run
  the same notebook** — it loads the last checkpoint, skips any model whose output zip already
  exists, and continues. The men subcats are small (130–180 products) so they're fast; the women
  subcats (430–626) take longer.
- You can run **multiple sessions in parallel** (one notebook per session) to go faster.
- Each notebook writes its 6 prediction zips into `…/<subcat>/data/predictions/<variant>/`.

When a subcat is done it should have **24 zips** total under `data/predictions/`
(6 per variant × 4 variants).

## Step 4 — Bring the predictions back (small files, safe to tar)

On the RC, from your home dir:
```bash
tar czf men-preds.tar.gz   men-8-subcat-split-proper-embedding/*/data/predictions
tar czf women-preds.tar.gz women-8-subcat-split-proper-embedding/*/data/predictions
```
Then LOCALLY (paths are preserved, so this drops each subcat's zips back into place):
```fish
scp '<your-rc-username>@login.explorer.northeastern.edu:~/men-preds.tar.gz'   ~/rc_handoff/
scp '<your-rc-username>@login.explorer.northeastern.edu:~/women-preds.tar.gz' ~/rc_handoff/
tar xzf ~/rc_handoff/men-preds.tar.gz   -C ~/claude_code/demand_modeling/
tar xzf ~/rc_handoff/women-preds.tar.gz -C ~/claude_code/demand_modeling/
```

## Step 5 — Tell me

Once the zips are back in the `*-proper-embedding/<subcat>/data/predictions/` folders, tell me
and I'll run the rest locally per subcat: `part4_verify` (rebuilds paths_config.yaml from the new
zips) → `part5` → `01_1 → 01_2 → 02 → 03 → 04_evaluation`. That produces the proper-embedding
elasticities and completes the lazy-vs-proper comparison.

---

### Notes
- **Why all 4 part3 notebooks per subcat:** the dual-variant `04` needs both the text-only and
  text+image datasets, and each is built from the lag1 + time-independent predictions, so all
  four variants are required.
- **gpu partitions:** `gpu-interactive` = 2h, no approval (use this). `gpu` = 8h but needs
  approval if you want fewer relaunches.
- **peft** must be installed in the `demand_modeling` env or the notebooks raise a clear error.
