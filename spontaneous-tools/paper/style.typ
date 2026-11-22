#let data = json("numbers.json")
#let arms = ("base", "explicit", "factual", "neutral", "continuation")
#let names = ("Unmodified", "Explicit", "Factual", "Neutral", "Continuation")
#let rule = 0.6pt + black
#let table-style = table.with(stroke: none, align: (x, _) => if x == 0 { left } else { center }, inset: (x: 6pt, y: 2.3pt))
#let top-rule = table.hline(stroke: rule)
#let mid-rule = table.hline(stroke: 0.35pt)
#let head(..args) = table.header(..args.pos().map(it => strong(it)))
#let artwork(file, caption) = figure(image(file, width: 100%), caption: caption, placement: none)
#let signed(n) = {
  let rounded = calc.round(calc.abs(n), digits: 1)
  let raw = str(rounded)
  let val = if raw.contains(".") { raw } else { raw + ".0" }
  (if rounded == 0 { "" } else if n < 0 { "−" } else { "+" }) + val
}
#let interval-cell(xs) = [#signed(xs.at(0)) (#signed(xs.at(1)), #signed(xs.at(2)))]
#let t = data.arms
// The 2015 submission style prints a fixed ruler, independent of typeset lines.
#let review-ruler() = context {
  let first = (counter(page).get().first() - 1) * 54
  for i in range(54) {
    let n = str(first + i)
    place(top + left, dx: 0.75in, dy: 1in + 2pt + i * 12pt,
      box(width: 0.45in, align(right, text(font: "TeX Gyre Heros", size: 8pt,
        fill: luma(70%), "0" * calc.max(0, 3 - n.len()) + n))))
  }
}

#let title-block(title, abstract) = block(width: 100%, above: 0pt, below: 11pt)[
  #set align(center)
  #set block(spacing: 0pt)
  #rect(width: 100%, height: 4pt, fill: black, stroke: none)
  #v(0.25in)
  #block[
    #set text(size: 17pt, weight: "bold")
    #set par(leading: 8.75pt)
    #title
  ]
  #v(0.25in)
  #rect(width: 100%, height: 1pt, fill: black, stroke: none)
  #v(24pt)
  #text(weight: "bold")[Anonymous Author(s)]
  #v(22pt)
  #text(size: 12pt, weight: "bold")[Abstract]
  #v(11pt)
  #pad(x: 0.5in, align(left, abstract))
]

#let paper-style(doc) = {
  set page(paper: "us-letter", margin: (x: 1.5in, y: 1in), columns: 1,
    numbering: "1", number-align: center, background: review-ruler())
  set text(font: "TeX Gyre Termes", size: 10pt, lang: "en", overhang: false)
  show math.equation: set text(font: "TeX Gyre Termes Math")
  show math.equation.where(block: true): set block(above: 7pt, below: 7pt)
  // Termes cap height at 10 pt is 6.62 pt. This gives an 11 pt baseline grid.
  set par(justify: true, first-line-indent: 0pt, leading: 4.38pt, spacing: 10.38pt)
  set heading(numbering: (..nums) => numbering("1.1", ..nums) + h(.65em))
  show heading: set text(weight: "bold", hyphenate: false)
  show heading: set align(left)
  show heading: set block(above: 11pt, below: 5.5pt)
  show heading.where(level: 1): set text(size: 12pt)
  show heading.where(level: 2): set text(size: 10pt)
  show raw: set text(size: 9pt)
  show table.cell: set par(justify: false)
  set figure(gap: 11pt)
  show figure: set block(above: 11pt, below: 11pt)
  show figure: set place(clearance: 11pt)
  show figure.caption: it => layout(area => {
    if measure(it).width <= area.width { align(center, it) }
    else { block(width: 100%, align(left, it)) }
  })
  show figure.where(kind: table): set figure.caption(position: top)
  show link: set text(fill: black)
  doc
}
