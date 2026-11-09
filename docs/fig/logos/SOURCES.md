# Logo sources

The software stack figure (`docs/fig/stack.svg`, `stack-dark.svg`; script `docs/fig/src/stack.py`)
shows the official logos of three programs it names: llama.cpp, PyTorch and Hugging Face. The
files in this directory are those logos, sanitised and otherwise unmodified. The rules for using
them are in `docs/fig/src/STYLE.md` ("Logos").

Each logo is used only to refer to the project it belongs to. It does not mean that the project, its
authors or its trademark holders endorse Tesseract, and Tesseract is not affiliated with them.
Trademarks and copyrights stay with their owners.

Every download goes into its own empty directory outside the repository and is read only as data.
`logoclean.py` (in `docs/fig/src`) turns it into the stored file. The stored file renders pixel for
pixel like the download (`logoclean.py --compare`: cairosvg, 512 px high, 0 pixels differ), so
nothing is recoloured, stretched or cropped. `logoclean.py` removes the `<style>` element (its
classes are folded into `fill` attributes, so two logos in one figure cannot collide), `<metadata>`,
titles, comments, editor namespaces (`inkscape:`, `sodipodi:`), editor state and unused ids.
Remaining ids are prefixed with the logo name. It refuses a file that has a script, an event
handler, an external or non-fragment link, `<foreignObject>`, `<image>`, `<text>` (and so remote
fonts), a DOCTYPE or an ENTITY, or an element or attribute it does not know. The stored files
contain only `<svg>`, `<g>`, `<defs>` and basic shapes.

| File | Logo | Used in | Size (bytes) | sha256 of the stored file |
| --- | --- | --- | --- | --- |
| `llamacpp.svg` | llama.cpp icon (`llama1-icon.svg`) | `stack`: LLAMA.CPP tile, own dark tile, no backing | 1311 | `097fa4e27e5f016d2545432c8e6b0aaa555b54a8152f5a3cc3421b263aecbd28` |
| `pytorch.svg` | PyTorch lockup (flame and wordmark) | `stack`: PyTorch tile, on a white backing tile; the lockup carries the name | 1771 | `7973d0cbfa1065e5480b5153015ccab36f8848c0782d114e5b224f4e7c503f62` |
| `huggingface.svg` | Hugging Face logo (the hugging face) | `stack`: line "Hugging Face models" in the PyTorch tile, on a white backing tile | 35306 | `c3a5e196066609fbc9cbf7a6c0d096f03e8cd7a6aae35df9a0c9be674b015237` |

## llama.cpp: `llamacpp.svg`

- Source: `third_party/llama.cpp/media/llama1-icon.svg` in this repository's llama.cpp submodule at
  commit `e3797b184d428c7a019362612b76971f60af6670`; upstream
  <https://github.com/ggml-org/llama.cpp/blob/master/media/llama1-icon.svg>.
- sha256 of the file as retrieved: `3613677483c2bd701c3205fafe1b93345b7c66f79c829b733a68137d35e142d9`
- Licence: the file is part of the llama.cpp repository, which is MIT licensed
  (`third_party/llama.cpp/LICENSE`, Copyright (c) 2023-2026 The ggml authors). The MIT notice is in
  that file; the logo is used to name the project.
- Why this variant: the icon is square and tight, so it stays legible at 80 units (about 44 px at
  README width). The full lockup `llama1-logo.svg` is a 3:1 banner with wide dark margins; its mark
  would be much smaller at the same tile height. The icon has its own dark tile, so it needs no
  backing on either theme.
- Changed: sanitised (the three CSS classes folded into `fill`, editor metadata removed).

## PyTorch: `pytorch.svg`

- Source: the logo that the PyTorch repository shows at the top of its README,
  <https://raw.githubusercontent.com/pytorch/pytorch/main/docs/source/_static/img/pytorch-logo-dark.svg>.
  The last commit that touched the file is `19040583702377c6f5f60e88477424a113a27565`, so the
  permanent URL is
  <https://raw.githubusercontent.com/pytorch/pytorch/19040583702377c6f5f60e88477424a113a27565/docs/source/_static/img/pytorch-logo-dark.svg>.
  The file at `main` and at that commit are byte-identical.
- sha256 of the file as retrieved: `af2d52668f5b4cbe3343e1a46b46cd1a9e782efb7aade9e8d2012dd21d3cf6b4`
- Licence and trademark: PyTorch's source code has a BSD-style licence (its `LICENSE`); this note is
  not about the code. PyTorch, the PyTorch logo and any related marks are trademarks of The Linux
  Foundation. The footer of <https://pytorch.org/> says "Copyright (c) The Linux Foundation. All
  rights reserved. The Linux Foundation has registered trademarks and uses trademarks" and points to
  its trademark usage policy, <https://www.linuxfoundation.org/trademark-usage>. The logo is used to
  refer to the project.
- Why the lockup: the file has a flame and the wordmark in one piece (200 by 40 units, ratio 5:1).
  It is used whole. The wordmark is dark grey (`#252525`), so it needs the white backing tile on the
  dark theme. The repository also has `pytorch-logo-flame.svg`, but that is the older purple and
  orange flame, not the current logo, so it is not used.
- Changed: sanitised (two CSS classes folded into `fill`, the `<style>` element and the
  `enable-background` property removed).

## Hugging Face: `huggingface.svg`

- Source: the logo-only file on Hugging Face's brand page <https://huggingface.co/brand>, which
  says "Download the official Hugging Face brand assets to use in your projects":
  <https://huggingface.co/datasets/huggingface/brand-assets/resolve/main/hf-logo.svg>
  (the server redirects to a signed CDN address; the stable address is the one above).
- sha256 of the file as retrieved: `942cad1ccda905ac5a659dfd2d78b344fccfb84a8a3ac3721e08f488205638a0`
- Licence and trademark: the brand page gives no licence text or usage rules beyond offering the
  assets for download. Hugging Face and its logo are trademarks of Hugging Face, Inc. The logo names
  the model hub whose models run through PyTorch (the diagram line "Hugging Face models"). If
  Hugging Face asks for it to go, delete `huggingface.svg`, the badge line in `stack.py` and this
  entry.
- Changed: re-serialised by `logoclean.py` (an XML declaration and an explicit `xmlns` first; the
  original has no style, metadata or ids to remove).

## Adding or replacing a logo

1. Download the official file into a new, empty directory outside the repository. Never open it with
   anything that runs scripts, and never run anything from it.
2. `build/tools/svgvenv/bin/python -I docs/fig/src/logoclean.py <download.svg> <key>` writes
   `docs/fig/logos/<key>.svg`, or refuses and says why.
3. `build/tools/svgvenv/bin/python -I docs/fig/src/logoclean.py --compare <download.svg> docs/fig/logos/<key>.svg`
   must print `differing pixels: 0`.
4. Add a section here: source URL (a pinned one where there is one), sha256 of the download,
   licence or trademark note, and what was changed. Check the owner's brand rules first.
5. Draw it with `Fig.logo("<key>", ...)` and rebuild; look at both previews.
