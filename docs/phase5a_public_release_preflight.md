# Phase 5A Public Release Preflight

## Scope

Phase 5A prepares the first public GitHub publication from the final local
release audit state.

Repository URL:

```text
https://github.com/megastell/HunyuanOCR-to-ncnn
```

## Actions

- Add the GitHub repository as `origin`.
- Update the GitHub Discussion draft with the public repository URL.
- Push the final release commit to the public `main` branch.
- Push all local milestone tags.

## Explicit Non-Actions

- Do not publish the GitHub Discussion.
- Do not create an upstream ncnn_llm pull request.
- Do not attach binary packages to a GitHub Release in this phase.

## Final Manual Checklist

- Review `docs/github_discussion_hunyuanocr_ncnn_draft.md`.
- Confirm model-license wording before publishing the Discussion.
- Decide whether to create a GitHub Release and attach the Linux/Windows
  packages from the Phase 4L recovery directories.
- Decide whether to prepare an optional upstream ncnn_llm pull request.
