# Adaptive-operator ablation configs

Each config composes an adaptive baseline and fixes exactly one operator's
application rate at its configured base rate.  All other adaptive operators keep their normal update
rule.  Dataset selection is inherited from `datasets/common.yaml`, so every
ablation run always covers the complete benchmark suite.

Run a variant, for example, with:

```bash
python3 scripts/run.py --config-name=ablation/improved/algea_1_rc
```

The `improved` and `standard` directories contain the same ablation matrix for
their respective algorithm families.  Each output file is unique, so variants
can run independently or in parallel without overwriting one another.
