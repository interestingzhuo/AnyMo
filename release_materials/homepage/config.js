/* AnyMo project page — the only file needed for routine link/media updates.
   Empty URLs are deliberately hidden, never replaced with guessed links. */
window.ANYMO_CONFIG = {
  venue: "NeurIPS 2026", // Based on the author's acceptance statement; confirm before publication.
  paperUrl: "https://arxiv.org/pdf/2605.29488",
  arxivUrl: "https://arxiv.org/abs/2605.29488",
  datasetUrl: "https://huggingface.co/datasets/L-yiheng/OmniHuMo",
  codeUrl: "",          // Add the actual public repository URL to reveal the Code links.
  projectUrl: "https://interestingzhuo.github.io/anymo-page/",       // Set to the final public project URL for canonical / social metadata.
  contactEmail: "",     // Optional project contact; hidden until configured.
  authorUrls: {},       // e.g. {"Yiheng Li": "https://your-verified-homepage.example"}
  preferLocalFigures: true, // fetch_assets.py sets this true after all figures are downloaded.
  allowRemoteFigures: false,
  media: {
    // MP4 / WebM paths or direct video URLs, not YouTube watch-page URLs.
    // These slots are hidden or use real paper figures when empty.
    teaser: "",
    text: "",
    speech: "",
    music: "",
    trajectory: ""
  }
};

