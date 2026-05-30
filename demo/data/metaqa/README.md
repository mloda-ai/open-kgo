# Full MetaQA dataset (not included)

This folder is the designated drop point for the full MetaQA dataset when you
want to run the demos at full scale instead of against the committed sample.

**Nothing in this folder except this README is tracked by git.** The
`.gitignore` rule (`metaqa/*`, `!metaqa/README.md`) makes it structurally
impossible to commit the dataset by accident.

## What goes here

Download MetaQA from the upstream project (Google Drive only, no stable
direct-download URL): open https://github.com/yuyuz/MetaQA and follow the
Google Drive link in its README. Place the files here:

- `kb.txt` -- the knowledge base
- `qa_test.txt` -- the 1-hop test split (`1-hop/vanilla/qa_test.txt` upstream)
- `LICENSE.txt` -- the CC BY 3.0 license text that ships with the dataset

Keep `LICENSE.txt` next to the data. MetaQA is licensed under
[CC BY 3.0](https://creativecommons.org/licenses/by/3.0/legalcode); the license
must travel with the data and any derivative you redistribute, with attribution
to Zhang et al., 2018.

## License boundary

Everything in this folder is third-party data under CC BY 3.0. The hand-authored
sample one level up (`../sample_kb.txt`, `../sample_qa.txt`) is ours and is not
derived from these files. Keeping the two physically separate keeps the license
boundary unambiguous.

See [`../README.md`](../README.md) for the step-by-step build instructions.
