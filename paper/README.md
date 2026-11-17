# SoftwareX manuscript

`main.tex` uses the Elsevier `elsarticle` class (from CTAN, not included) and
reads its tables and quoted numbers from `gen/`, which is produced from
`results/` by

    python3 tools/make_tables.py paper/gen
    cd paper && latexmk -pdf main.tex
