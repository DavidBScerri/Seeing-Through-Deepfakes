# Hash Module — SHA-256 byte-exact identification

This module gives the pipeline a **byte-exact fingerprint** for every
uploaded image and, when a small text-only registry is provided, looks
that fingerprint up against a closed set of previously registered
digests.

## What SHA-256 CAN and CANNOT do here

- **CAN**: uniquely identify a file whose bytes are identical to a file
  someone recorded earlier.
- **CANNOT**: decide, on its own, whether an image is AI-generated,
  edited, or real. That interpretation only exists because a trusted
  registry record carries an `origin_label`, and even then it is only
  as trustworthy as whoever wrote the record.

Concretely:

- SHA-256 is **byte-exact, not perceptual**. Recompression, format
  conversion, EXIF stripping, a single-pixel edit, or any other byte
  alteration normally produces a **different** digest — the same
  visible image will not match its previously registered self.
- A **no-match** result is **inconclusive**. It means only "no
  byte-identical file has been registered". It NEVER means
  "not AI-generated", "not manipulated", or "real". Downstream UI must
  not rebrand it as anything of the sort.
- A **match** only carries meaning through its record's `origin_label`.
  Never render "AI-generated" for an EXACT_MATCH unless the matched
  record explicitly has an AI-related label.
- **Perceptual hashing** (pHash / dHash / DINOv2 similarity indexes) and
  **external provenance registries** (public C2PA claim ledgers,
  vendor-hosted registries) are **future work**; this module does not
  cover either.

## The registry (text-only, tiny by construction)

Each entry is only a digest and a small descriptive metadata block —
**no image bytes, thumbnails, embeddings, or perceptual hashes are ever
stored**. Fields:

```
sha256           : 64-character lowercase hex digest (required)
origin_label     : ai_generated | ai_modified | camera_or_human | unknown  (required)
provider         : optional short string (e.g. "openai")
model            : optional short string
source_reference : optional non-sensitive reference (ticket / dataset id / URL)
notes            : optional short free-text notes
```

Every text field is capped at 512 characters so a caller cannot try to
stash base64-encoded image bytes in `notes`. The schema is closed
(`extra="forbid"`): unknown fields are rejected on load.

### Runtime location and configuration

The registry file lives outside the source tree. The library picks the
path with this precedence:

1. explicit `path=` on `HashRegistry(...)` / `load_registry(...)`;
2. the `SEEING_THROUGH_DEEPFAKES_HASH_REGISTRY` environment variable;
3. the default `data/hash_registry/registry.json` under the repo root
   (`data/` is gitignored — any file at this location stays out of
   version control).

The committed **`registry.example.json`** is a schema reference only —
it is never used as a live registry. Real registries stay outside the
package.

### Storage guarantees

- **Text only.** No binary payload of any kind — the schema forbids it.
- **Tiny.** A registry file grows linearly in the number of entries;
  each entry is a few hundred bytes at most. Loading refuses anything
  over 10 MiB as a safety cap; nothing that fits the schema will come
  close.
- **Atomic writes.** Every `register` / `remove` writes to a
  same-directory `.registry.*.tmp` file, `fsync`s, then `os.replace`s
  on top of the live file. A crash mid-write leaves the previous valid
  registry untouched.
- **Never overwritten on parse/validation failure.** A malformed
  registry surfaces as `INVALID_REGISTRY` — the loader refuses to
  reconstruct or replace it. Fix the file first.
- **Duplicates rejected.** Two records for the same digest are refused
  at load time; identical re-`register` calls are idempotent no-ops.

## Statuses the module reports

`HashLookupStatus` distinguishes five outcomes so the UI can render
each honestly:

- `exact_match` — a byte-identical file has been registered; the
  matched record carries whatever origin claim was recorded.
- `no_match` — the registry loaded fine, but the digest is not
  registered. **Inconclusive.**
- `registry_unavailable` — no registry was configured, or the
  configured file does not exist. Not the same as `no_match`.
- `invalid_registry` — the file exists but is malformed JSON or
  violates the schema. Not the same as `no_match`.
- `error` — an unexpected exception during lookup.

## CLI

```bash
# Print the streaming SHA-256 digest of one file
python -m src.genai_detection.hash_module.cli hash path/to/image.png

# Record the digest in the registry (image is NOT copied or modified)
python -m src.genai_detection.hash_module.cli register path/to/image.png \
    --label ai_generated --provider example --model example-v1

# Hash and look up
python -m src.genai_detection.hash_module.cli lookup path/to/image.png
```

Exit codes: 0 = match / hash success, 1 = no-match, 2 = user error,
3 = registry unavailable, 4 = registry invalid.

## Integration with the web app

The demo server always returns a `hash` block from `POST /api/analyse`
carrying the SHA-256 digest of the **original uploaded bytes** (before
any PIL decode/re-encode), together with the lookup status, matched
record (if any), a plain-language rationale, and a fixed scope
statement. The result is exposed as its own **Hash Evidence** card in
the UI and is deliberately **not fed into the fusion formula** in this
iteration — adding a fusion weight would require separate evaluation
and David's sign-off per the fusion-formula rule in `CLAUDE.md`.
