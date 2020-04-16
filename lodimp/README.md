# LoDimP/lodimp

This directory defines the `lodimp` Python module. All model architectures,
utilities, training scripts, and so forth live here.

To invoke a script, call the lodimp module directly and pass the script
as the first argument. Example:

```bash
cd ~/path/to/LoDimP
python lodimp lodimp/train.py ~/path/to/data real 64
```

The module entrypoint sets `PYTHONPATH` and also offers some utilities,
e.g., automatically detaching to screen or redirecting stdout to a file.
Run `python lodimp --help` for a detailed list.
