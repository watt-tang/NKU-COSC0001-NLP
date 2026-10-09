$ErrorActionPreference = 'Stop'
Push-Location $PSScriptRoot
try {
    & xelatex -interaction=nonstopmode -halt-on-error -jobname=report main.tex
    if ($LASTEXITCODE -ne 0) { throw 'XeLaTeX first pass failed.' }
    & bibtex report
    if ($LASTEXITCODE -ne 0) { throw 'BibTeX failed.' }
    & xelatex -interaction=nonstopmode -halt-on-error -jobname=report main.tex
    if ($LASTEXITCODE -ne 0) { throw 'XeLaTeX second pass failed.' }
    & xelatex -interaction=nonstopmode -halt-on-error -jobname=report main.tex
    if ($LASTEXITCODE -ne 0) { throw 'XeLaTeX final pass failed.' }
} finally {
    Pop-Location
}
