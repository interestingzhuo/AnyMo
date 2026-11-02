# Release candidate status

Prepared on 2026-11-01. This is a local publication candidate; creating this directory did not publish a repository, model or project-page update.

Completed in this candidate:

- English/Chinese README, verified paper/homepage links, citation metadata and data conventions.
- Explicit text/trajectory CLI; no hard-coded checkpoint or interactive debugger.
- Autograd-safe joint recovery and a regression test for the original in-place backward failure.
- Minimal inference dependencies separated from optional evaluation dependencies and the legacy environment snapshot.
- Inference module imports separated from evaluator initialization; the original tokenizer variants remain registered.
- Original research code retained, active metric debugger blocks removed, portable launchers, relative split paths.
- Package exclusion rules for checkpoints, datasets, caches, logs, credentials and local infrastructure helpers.

Validation completed: four CPU unit tests, Python syntax checks, shell syntax checks, citation metadata and local Markdown-link checks. The new standalone model import / GPU integration check remains pending because the automatic permission review timed out twice; no new GPU workload was launched during that blocked step. See `validation/checks.json`.

Before making the candidate public:

- Select the code license. The authors deliberately left it pending in `LICENSE_STATUS.md`.
- Resolve applicable copied/adapted third-party license texts and model/data/media release terms.
- Choose the final public code repository URL and set the project page's `codeUrl`. The current homepage has no configured public code URL; no URL has been guessed.
- Confirm anonymous access or the intended access-request process for the model and processed-test repositories. The asset download helper does not grant access.
- Run full training, audio feature preparation and benchmark evaluation if those capabilities will be advertised as reproducible on release day. The current release highlights the validated text/trajectory path.

The separate `release_materials/homepage/` directory contains a proposed update for the existing static homepage. The paper is already on arXiv; no submission or revision has been performed by this preparation step.
