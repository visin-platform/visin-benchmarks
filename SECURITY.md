# Security reporting

Report vulnerabilities or exposed credentials privately through this repository's
GitHub **Security → Report a vulnerability**, when enabled. If private reporting
is unavailable, contact the maintainer at toomastahves@hotmail.com. Do not include
secrets or vulnerability details in a public issue.

Keep Visin pipeline keys in the ignored `.env` file or environment variables.
Use a key scoped to the intended project. Training uploads metrics, visualizations,
and model artifacts to the configured Visin service. Review configs and endpoint
settings before running them. Offline mode queues reports locally for later sync.

Security fixes are made on the current default branch. No maintenance guarantee
is provided for older revisions. Dataset archives and pretrained checkpoints come
from external sources and retain their own terms and provenance.
