## The Project Structure

Back to [overview](../README.md).

SIMSON is structured into different subsections via a folder structure that is described here in detail.
Firstly, the contents of the [**main folder**](#main-folder) are described, followed by an in-depth description
of the [**source code**](#source-code) and [**data structure**](#data-structure).

### Main Folder

The project's main folder consists of five folders and a couple of extra files which are desribed here.

1. #### data

Here, naturally, the data used in the project is stored. It is desribed in detail below 
in the section 'Data Structure'.

2. #### doc

The doc folder consists of the markdown and png files and of this documentation. In the future, 
potentially a full HTML-documentation might be ammended in the future.

3. #### ODYM

Here, the ODYM framework is loaded as a submodule. ODYM is widely used in the source code
to create the material flow analysis structure as well as using its dynamic stock modelling functionality. 

Some ODYM classes are amended in the source code (s. below). More information on ODYM can be found 
[here](https://github.com/IndEcol/ODYM).

4. #### simulation

This folder contains the simulation environment, discussed in detail
in [this document](../Simulation.md).

6. #### src

The *src* folder contains the source code, further described in the section below.

#### Project Files

Further, in the main folder some project configuration files are stored.

The *requirements.txt* and *poetry.lock* files serve to help install the necessary libraries.

The git configurations are stored in *.gitignore* (to specify folders that should not be 
pushed during commits like the original data)  and *.gitmodules* (to ensure swift installation of the used submodules).

Lastly, '*README.md*' provides the startpoint of the documentation
and the '*CITATION.cff*' serves enables easy citation of SIMSON on Github (via the 'cite this repository' functionality).

### Source Code

### Data Structure

The data used is divided into the four sub-categories 
*models*, *original*, *output* and *processed*.

In *models*, the ODYM DynamicStockModel and MFASystem instances are stored for various configurations 
in order to be able to reload them quickly. Additionally, the LSTM Networks used as a prediction option are stored in
a subfolder 

The *original* folder contains the original data (stored on Gitlab, s. [Installation](Installation.md)) 
via submodules, ordered by data source.

'*Output*' is the folder storing the results of various tests and computations that are not part of the 
[simulation](Simulation.md) environment.

Lastly, in the *processed* folder SIMSON stores the cleaned and restructured original data once 
it has been loaded for the first time in a format that is easy to read. This accelerates the combutations.