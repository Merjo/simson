## SIMSON - an introduction

Back to [overview](../README.md).

In order to model the global steel cycle (and potentially other materials in the future),
the Potsdam Institute for Climate Impact Research (PIK) has set up their model SIMSON: 'Simulation 
In-Use Material Stocks with the ODYM Network'. 

It is based on dynamic stock modelling (DSM) and material flow analysis (MFA) as well as various prediction techniques.
In this article, we are presenting SIMSON's:

1. [MFA Structure](#structure)
2. [Index Table](#index-table)
3. [Flows & Stocks](#flows--stocks)

### Structure

Here is an overview of the SIMSON MFA structure:

![Structure of SIMSON](simson_fin_structure.png)

SIMSON uses a total of 10 process with process IDs 0-9, 
where '0' represents all processes outside of its system boundary. 
In this section we briefly summarise the steel cycle as conceptualised by SIMSON, 
mentioning the process IDs along each step.

It distinguishes between two production routes: from flast furnace to basic
oxygen furnace (BOF, 1) and the electric arc furnace (EAF, 2) via direct reduction. The iron ore necessary 
for steel production is seen as exogenous to the system (0). Steel is formed into intermediate products
like sheets and rods in the forming process (3), and subsequently 'fabricated' (4) into end use products
like cars and machinery. 

After the 'In-Use' phase (5), steel reaches its end of live, 
is collected as scrap (6) or left in the landscape ('Dissipative', 8) and then recycled (7)
or ends up in landfill waste (9).

Additionally, steel trade occurs between the various world regions. We distinguish between crude steel trade,
indirect trade (trade of steel products in the use-phase, like the export of second-hand cars) and scrap trade.

### Index Table

The *Open Dynamic Material Systems Model (ODYM)* is a Python framework for
*DSM* and *MFA*. It realises flows and stocks via *NumPy* arrays, and the array's dimensions represent
various aspects of the MFA/DSM systems that are summarised by an *index table* in *ODYM*. 

These index tables assign every dimension (row) of an MFA a name and letter, which then can be used swiftly in 
'*Einstein sums*' in *NumPy*. 

The index table of SIMSON, along with the values that the indices of every dimension 
represent is shown here:

| Name   | Letter | Values                                                                                                                                                                               |
|--------| --- |--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Time | t | 1900-2100 (total of 201 years)                                                                                                                                                       |
| Element | e | *Fe (Iron)* for now, *Cu (Copper)* planned in future                                                                                                                                 |
| Region | r | 12 *REMIND* regions or 10 *Pauliuk* regions sorted alphabetically                                                                                                                    |
| In-use goods | g | *Construction*, *Machinery*, *Products*, *Transport*                                                                                                                                 |
| Waste | w | 10 Waste categories: 6 according to *Wittig* (*construction & development*, *municipal solid waste*, etc.), *forming* & *fabrication* scrap, *dissipative* & *not collectable* waste |
| Scenario | s | 5 SSP scenarios (SSP1, SSP2, etc.)                                                                                                                                                   |


### Flows & Stocks

Test $5+1$
tst2 $$5+1$$


$$F_{Env-Use} = T_{indirect}^{I}(g)$$
\$$F_{Use-Env} = T_{indirect}^{E}(g)$$
\end{equation}
\begin{equation}
    F_{Fbr-Use}(g)=\const{I(g)}-\const{T_{indirect}^{I}(g)}+ \const{T_{indirect}^{E}(g)}
\end{equation}
\begin{equation}
    F_{Fbr-Scr}(g) = F_{Fbr-Use}(g) (\frac{1}{\const{Y_{Fbr}(g)}}-1)
\end{equation}
\begin{equation}
    F_{Frm-Fbr} = \sum_g(F_{Fbr-Use}(g) + F_{Fbr-Scr}(g))
\end{equation}
\begin{equation}
    F_{Frm-Scr} = \frac{F_{Frm-Fbr}}{\const{Y_{Frm}}}-F_{Frm-Fbr}
\end{equation}
\begin{equation}
    F_{Env-Frm} = \const{T_{crude}^{I}}
\end{equation}
\begin{equation}
    F_{Frm-Env} = \const{T_{crude}^{E}}
\end{equation}
\begin{equation}
    P = F_{Frm-Fbr} + F_{Frm-Scr} + \const{T_{crude}^{E}} - \const{T_{crude}^{I}}
\end{equation}
\begin{equation}
    F_{Use-Scr}(g,w) = (\const{O(g)}-\const{T_{indirect}^{E}(g)})\const{D_{Use-Scr}(g,w)}
\end{equation}
\begin{equation}
    F_{Env-Scr}(w)=\const{T_{scrap}^{I}(w)}
\end{equation}
\begin{equation}
    F_{Scr-Env}(w)=\const{T_{scrap}^{E}(w)}
\end{equation}
\begin{equation}
    S_{available}(w) = \sum_gF_{Use-Scr}(g,w) + \const{T_{scrap}(w)}+ F_{Fbr-Scr} + F_{Frm-Scr}
\end{equation}
\begin{equation}
    S_{recyclable} = \sum_wS_{available}(w)\const{D_{Scr-Rcy}}
\end{equation}
\begin{equation}
    S_{prod-usable} = min(S_{recyclable},P\const{V_{maxScrapShareProduction}})
\end{equation}
\begin{equation}
    F_{EAF-Frm} = min(0, \frac{\frac{S_{prod-usable}}{P}-\const{V_{scrapShareBOF}}}{1-\const{V_{scrapShareBOF}}})P
\end{equation}
\begin{equation}
    F_{Rcy-EAF} = F_{EAF-Frm}
\end{equation}
\begin{equation}
    F_{BOF-Frm} = P - F_{EAF-Frm}
\end{equation}
\begin{equation}
    F_{Rcy-BOF} = F_{BOF-Frm}\const{V_{scrapShareBOF}}
\end{equation}
\begin{equation}
    F_{Env-BOF} = F_{BOF-Frm} - F_{Rcy-BOF}
\end{equation}
\begin{equation}
    F_{Scr-Rcy} = F_{Rcy-EAF} + F_{Rcy-BOF}
\end{equation}
\begin{equation}
    F_{Scr-Wst} = \sum_wS_{available} - F_{Scr-Rcy}
\end{equation}
$$