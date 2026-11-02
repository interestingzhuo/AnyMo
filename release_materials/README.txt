AnyMo publication preparation

The code candidate is at this repository root. The original official/ snapshot and reproduction experiments remain unchanged.

homepage/
  index.original.html: the existing public project's index.html used as the patch baseline.
  index.html: proposed content update for the same page.
  config.js: the existing link/media configuration, preserved for reference.
  homepage-release.patch: additive resource/inference section for the current page.

To review in an existing interestingzhuo/anymo-page checkout:
  git apply --check /path/to/homepage-release.patch
  git apply /path/to/homepage-release.patch

This does not publish the page. The final code repository URL remains unset rather than guessed. Set codeUrl in the real config.js once the public code location is decided. Model and processed-test URLs point to the author's existing repositories; confirm their intended access settings before announcing availability.

publication_copy.txt contains proposed arXiv Comments and announcement text. The arXiv paper already exists as2605.29488v3 (2026-10-29); no arXiv submission, metadata update, GitHub push or Hugging Face upload has been performed here.

The code license is pending the authors' decision. Do not add a standard-license badge or describe this candidate as having a chosen open-source license until that decision is made. The package also inventories existing third-party attributions for review.
