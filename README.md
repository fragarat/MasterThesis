# Master Thesis

This repository contains all the source files, data, and documentation for my Master’s thesis.

## Repository Structure
```text
MasterThesis/
├── chapters/ # Individual thesis chapters (e.g. introduction, methods)
├── figures/ # Images and figures used in the thesis
├── tables/ # Tables and data tables
├── style/ # Custom LaTeX style files (e.g. thesis.cls)
├── appendices/ # Appendices and supplementary materials
├── main.tex # Main LaTeX file that compiles the thesis
├── references.bib # Bibliography (BibTeX)
├── data/ # Raw and processed data used for analysis
└── README.md # This file
```


## Requirements & Compilation

1. **Requirements**  
   You will need a LaTeX distribution that supports modern font usage (e.g. [TeX Live](https://www.tug.org/texlive/) or [MiKTeX](https://miktex.org/)).  
   If you use system or custom fonts, compile with **XeLaTeX** (or optionally **LuaLaTeX**).

2. **Compile the thesis**  
   In the terminal, run:

   ```bash
   xelatex main.tex
   ```
   You may need to run the command twice (or more) to resolve cross-references and generate the bibliography.

3. **Bibliography**
   After running LaTeX, run:
   
   ```bash
   bibtex main
   ```
   Then re-run xelatex main.tex until everything is resolved.

## Customization & Usage

- **Fonts & Style**: To change fonts or styling, edit the files in the `style/` directory (e.g. `thesis.cls` or custom .sty files).

- **Content**: Modify or add chapter files under `chapters/` to update your thesis content.

- **Figures & Tables**: Place all images in `figures/` and tables in `tables/`.

- **Data & Analysis**: Keep your data and analysis scripts in the `data/` directory (or a subfolder), separate from the LaTeX source.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
