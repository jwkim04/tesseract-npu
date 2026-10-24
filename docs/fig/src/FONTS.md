# Fonts

The figures in `docs/fig/*.svg` set all text in **Noto Sans Mono**, version 2.006, Regular and
Bold. The figures use the slashed-zero alternate (`zero.slash`).

| File | SHA-256 |
| --- | --- |
| `NotoSansMono-Regular.ttf` | `6b692c4b6d15ccf59f1c1fe8d11cb8a92f51960f3e9f1f523781755a3af7e29f` |
| `NotoSansMono-Bold.ttf` | `b22563d5f03b345a62115e78d8789c9ed99c2cfa8059fad6e7a5c4ced288e21e` |

- **License.** The SIL Open Font License, version 1.1 (<https://openfontlicense.org>).
  Copyright 2015 Google LLC. "Noto" is a trademark of Google LLC.
- **Source.** The Ubuntu package `fonts-noto-mono`, installed at
  `/usr/share/fonts/truetype/noto/`. Upstream is <https://github.com/notofonts/latin-greek-cyrillic>.
- **How the font is used.** `tsrfig.py` reads the glyph outlines with fontTools and writes them
  into each SVG as `<path>` elements, one definition per glyph used, placed with `<use>`. The SVG
  files contain no font files and no `<text>`, and viewers need no font installed.
- **License terms for the figures.** Under the OFL, documents made with the font are not covered
  by the license. The figures can therefore be distributed under the repository's license.
- **Changing the font.** `tsrfig.py` warns when the installed files differ from the hashes above,
  because changed outlines change the SVG bytes. Two environment variables point the library at
  other copies of the same version: `TSR_FIG_FONT_REGULAR` and `TSR_FIG_FONT_BOLD`.
