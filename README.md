# Market-implied stock price PDFs: risk-neutral and physical

Option prices embed information about the markets expectation of future performance of the underlying asset.
The set of European option prices across strikes for a given maturity $T$ implies a risk-neutral probability density function of the price $S_T$ of the underlying asset at the maturity.

This project uses several years of daily SPX option chain data to extract the market-implied risk-neutral pdfs for options with 1 day maturities, 7 day maturities, and 28 day maturities.
Using the Breeden-Litzenberger relation, which says that the implied pdf is given by the second partial derivative of price at maturity with respect to strike:
$f_Q(K) = e^{rT} \frac{\partial^2 C(K, T)}{\partial K^2},$
we numerically recover the pdf.

For the full rendered notebook (with tables and plots), see:

[My project on nbviewer](https://nbviewer.org/github/swbuchanan/implied-stock-distributions/blob/main/project.ipynb)

or look at project.ipynb, project.html, or project.pdf in the github repository.
