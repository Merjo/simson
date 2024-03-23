## The Project Structure

- Back to [overview](../README.md).

SIMSON is structured into different subsections via a folder structure that is described here in detail.
Firstly, the contents of the [**main folder**](#main-folder) are described, followed by an in-depth description
of the [**source code**](#source-code) and [**data structure**](#data-structure).

### Main Folder

The project's main folder consists of five folders and a couple of extra files which are desribed here.

1. #### data

Here, naturally, the data used in the project is stored. It is desribed in detail below 
in the section 'Data Structure'.

2. #### doc

The doc folder consists of the markdown and other files of this documentation.
This includes a [bachelor's thesis on SIMSON](bachelor.hosak.pdf) with an in-depth description
of SIMSON's calculations and data sources.
In the future, a full HTML-documentation might be amended.

3. #### ODYM

Here, the ODYM framework is loaded as a submodule. ODYM is widely used in the source code
to create the material flow analysis structure as well as using its dynamic stock modelling functionality. 

Some ODYM classes are amended in the source code (s. below). More information on ODYM can be found 
[here](https://github.com/IndEcol/ODYM).

4. #### simulation

This folder contains the simulation environment, discussed in detail
in [this document](../Simulation.md).

5. #### src

The *src* folder contains the source code, further described in the section below.

#### Project Files

Further, in the main folder some project configuration files are stored.

The *requirements.txt* and *poetry.lock* files serve to help install the necessary libraries.

The git configurations are stored in *.gitignore* (to specify folders that should not be 
pushed during commits like the original data)  and *.gitmodules* (to ensure swift installation of the used submodules).

Lastly, '*README.md*' provides the startpoint of the documentation
and the '*CITATION.cff*' serves enables easy citation of SIMSON on Github (via the 'cite this repository' functionality).

### Source Code

The project's source code, and hence most of its Python files, 
is in the *src* folder. It is structured into these five sub-directories:

1. #### base_model

In this directory, all aspects of the realisation of the main MFA structure of 
SIMSON are stored. This includes the implementation of the MFA system as well as
the dynamic stock modelling, including helper and load functions.

The most import file is '*simson_base_model.py*'. From here various routines are called,
and hence it is a good point to start understanding the code.

2. #### calc_trade

Here, the various routines to load and scale the different types of trade
are stored.

3. #### economic_model

This folder contains the realisation of the economic model. In SIMSON, first
the base-model is run and then afterwards the assumtions on price elasticities are
used to adapt demand, scrap share in production and the scrap recovery rate.
This is explained in detail [here](Economic_Module.md).

The folder contains scripts that realise the optimization and load 
and store new dynamic stock models using an inflow-driven approach as the demand
is changed due to the elasticities.

4. #### modelling_approaches

This directory contains the realisation of the three modelling approaches
*inflow-*, *stock-* and *change-driven* dynamic stock modelling for the years 1900-2008
that are described in [this Bachelor's thesis](bachelor.hosak.pdf).

Additionally, functions used by all approaches are summarised in helper scripts, 
such as data loading, and the preparation of the results for the SIMSON 
material flow analysis system.

5. #### odym_extension

Here, two classes inheriting from ODYM are stored to ammend
the functionalities in ODYM with some practical functions. 

One is an extension of the ODYM dynamic stock model, allowing it to realise
multiple dimensions (have one model for all regions, scenarios and in-use goods),
the other inherits from the ODYM MFASystem, adding some handy functions for
more understandable and concise code.

6. #### predict

In the *predict* folder, the realisation of the three stock prediction techniques
discussed in [this Bachelor's thesis](bachelor.hosak.pdf), *Pauliuk*, *Duerrwaechter* and
the machine learning based *LSTM* prediction are implemented.

Further, some scripts containing tools and tests for all three techniques are included.

7. #### read_data

In *read_data* naturally the data loading functions are stored. The most important 
script is *load_data.py* which is used in many other files. 

From here, all other data loading and cleaning routines that read the original data 
are accessed, and then the cleaned and normalised data is stored in a handy *.csv* format
in the folder *data/processed*. 

When this has been initialised, in the future the data loader will automatically use the 
processed csv files, unless a specific recalculate flag is set to ``True``.

8. #### tools

The *tools* folder contains some common functionalities used all over 
the project like the *REMIND* or *Pauliuk* region mappings to aggregate country data
as well as functions to split data into subregions. 

Additionaly, it contains the *config.py* script where all project
configurations are stored. The configurations are realised as a instance
of the ``Config`` class, and this instance is accessed in most scripts to ensure 
that all files are working with the same configurations.

The configurations can either be changed directly in the *config.py* file, 
otherwise they can be initiated via the [simulation module](Simulation.md).

9. #### visualisation

Lastly, in the *visualisation* folder, several useful scripts to create
relevant graphics of the model or aspects of the model are stored.

The most useful one is *visualise_flows_and_stocks.py* where all flows 
and stocks of the MFA can be visualised via intuitive configuration variables.

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

Lastly, in the *processed* folder, SIMSON stores the cleaned and restructured original data once 
it has been loaded for the first time in a format that is easy to read. This accelerates the combutations.