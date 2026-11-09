"""Prefer installed Hugging Face files without network metadata requests."""


def load_pretrained(loader, model_name, revision="main", **kwargs):
    """Download only when a complete local load is unavailable."""
    try:
        return loader.from_pretrained(
            model_name, revision=revision, local_files_only=True, **kwargs
        )
    except OSError:
        return loader.from_pretrained(model_name, revision=revision, **kwargs)
