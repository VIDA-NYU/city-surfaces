# Changelog

## Unreleased
- Restructured to `citysurfaces` package with eval-only entrypoint.
- Hardened dataset loading (path validation, opt-in black fallback) and sanitized dump filenames.
- Added safe checkpoint helper and AMP support for inference.
