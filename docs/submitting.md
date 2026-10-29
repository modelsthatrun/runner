# Submitting results

`mtr submit` sends run records to the ModelsThatRun inbox. This page says what it checks, who reviews them, and how they show.

## What a submission is

One run record from `mtr run`, plus:
- **its raw timings,** `runs/<size>.times.json`: every timed call of each of the three runs, or every timed request of a language model
- **your note** (`-m`), shown to the reviewer
- **the license.** Records are dedicated to the public domain under CC0 1.0.

The format is `tools/measure/src/mtr_measure/schema/community-submission.schema.json`. In v1 only run records (speed and VRAM) are taken. Accuracy hardly depends on the GPU, so `mtr eval` results stay local for now.

## What is checked

**On your machine, before sending (`mtr submit`):**
- The record passes its schema and was taken under the current protocol version.
- It carries provenance:
  - the tool's git commit
  - whether `tools/measure/`, `timing/` or the model directory had uncommitted changes (they must not have)
  - the sha256 of `adapter.py`, `uv.lock` and the input
- Its timings file is there.
- An image model was timed on its task's timing image, not on another `--input`.
- It warns if your commit isn't on a public branch.

**By the inbox, on arrival:**
- The same schema and provenance checks.
- **The timings must give the record's numbers.** Each run's median and p90 are recomputed from the raw calls, then the median across runs, exactly as the tool builds a record. VRAM can't be recomputed: it comes from the GPU's memory counter.
- The model and size must have an adapter in this repository, and an image model must have been timed on its task's timing image.
- **Limits:** 256 KB per submission and 50 submissions a day per account. The same record sent twice stays one submission.

**At review, on the maintainers' side:**
- **The commit and the adapter.** The commit exists in this repository, and the adapter and lock at that commit are the ones that ran.
- **The weights.** Their hashes match the published checkpoint.
- **VRAM.** A peak more than 15% away from the reference machine's, at the same precision, gets a closer look. VRAM hardly depends on the GPU model.
- **Then a person decides:** accept, hide or reject, with a note you can read in `mtr submissions`.

None of this proves a record is honest, because nobody can see your machine. It makes records reproducible and odd ones easy to spot. Then labels and other contributors do the rest.

## How it shows

**Where.** An accepted record appears in the model page's **Community results** table:
- size
- GPU (marked *power-limited* if it ran below its default power limit)
- precision or quantization
- speed: milliseconds per call, or tokens per second
- VRAM peak
- your GitHub login, linked
- the date

**Labels.**
- **Unverified** until another contributor with the same GPU, size and precision gets within 10% of it. 10% is the spread the protocol allows between runs.
- **Reproduced** after that.
- An RTX 4090 result within 10% of the reference measurement adds to "Reproduced by N contributors" on the reference card.

**Where it never goes.** Community results never replace the reference measurement, and they never appear in task charts or rankings.

## After you submit

- **Review is by hand.** It may take days, and nothing notifies you. `mtr submissions` shows each submission's state (pending, accepted, hidden or rejected) and the reviewer's note.
- **The site isn't public yet.** Accepted results appear on modelsthatrun.dev once it opens.
- **A sign-in lasts 90 days.** After that, `mtr login` again.

## Removing a result

Open an issue and name the submission id (`mtr submissions` lists them). It is hidden from the page.
