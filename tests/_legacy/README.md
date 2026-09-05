# Frozen numerical regression references

These two source files preserve the implementation used by DWD
`1.0.5+compat2`, based on upstream DWD 1.0.5 by Iain Carmichael and maintained
upstream by Kitware. They are test fixtures, not alternate production modules.
The original MIT license is retained in this directory and at repository root.

Source: [cheng338/dwd, compatibility commit 8bd3185](https://github.com/cheng338/dwd/tree/8bd318585c1a4aec1b56ae5ccca1b2c0f24ca9ce).
Upstream: [slicersalt/dwd](https://github.com/slicersalt/dwd), commit
`b564db19193674d967a9dc9327709869d5b67078`.

The copies normalize line endings to LF; their algorithm text is unchanged.
SHA-256 checksums of the fixture files are:

- `gen_dwd.py`: `f98161b6723e06df7f366fcab06192e274230359736bdb464e8059049474ae58`
- `gen_kern_dwd.py`: `f9a4a82c8a424cab2ddcfd1be1a25a2956321cf4bfbecda3d13093842d741438`

The original captured CRLF source hashes were
`bdf8129b8741e951c5db984fbbf2ec68c783b64f8c4f91c136ca21f035f5acb9`
and `576f294a2f8ac4d71ff852ff392a63b4b44dbbc71c1c09810e01c388ed509b1a`,
respectively. The separate independent tests verify the corrected solver using
augmented linear systems, objective finite differences, and singular-kernel
identities. Agreement with these legacy fixtures alone does not establish that
the legacy solver's mathematics is correct.
