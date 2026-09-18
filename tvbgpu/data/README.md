# External scientific assets

The frozen Larter–Breakspear training path requires these exact files here:

- `connectivity_zerlaut_68_newcentres.zip`: contains
  `QL_20120814_Connectivity/weights.txt` and
  `QL_20120814_Connectivity/tract_lengths.txt`.
- `QL_20120814_ProjectionMatrix.mat`: contains `ProjectionMatrix`.
- `QL_20120814_RegionMapping.txt`.

Obtain the verified versions from the workshop organiser and place them in this
directory. No download URL or redistribution license has yet been verified.
These files are deliberately excluded from Git pending provenance review.
Do not substitute `connectivity_68.zip` or a different projection silently.

Empirical EEG, posterior checkpoints and container images are also external.
The inference CLI accepts prepared EEG by path. Projection channel order is
still awaiting confirmation; do not infer it from arbitrary EEG channel names.
