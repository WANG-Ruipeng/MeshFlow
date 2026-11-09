# Pre-outcome arithmetic clarification

Recorded during head training, before terminal evaluation. The immutable PROTOCOL.md says that each DEV parent's owner recall >=90% implies all20 owners must be recalled. That sentence is arithmetically wrong: 2d has13 owners, so12/13=92.31% passes;610 has4 and608 has3, so those require4/4 and3/3. The actual frozen code implements the stated per-parent >=90% rule, with minimum19/20 combined. Neither code, numerical threshold nor checkpoint budget is changed.

This is a disclosure of the original text error, not a threshold relaxation chosen after observing predictions. DEV still supplies no normalization, gradient, threshold fitting or model selection.
