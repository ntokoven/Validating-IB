#!/bin/sh
pdflatex -synctex=1 -interaction=nonstopmode $1.tex
biber $1
makeindex $1.nlo -s nomencl.ist -o $1.nls -t $1.nlg
pdflatex -synctex=1 -interaction=nonstopmode $1.tex
