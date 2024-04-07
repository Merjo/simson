# Installation of SIMSON

Back to [overview](../README.md).

*Note: In its current version, SIMSON can only be fully used
by members of the Potsdam Institute for Climate Impact Reserach (PIK).*

SIMSON can be used for a multitude of purposes - creating
own simulations, connecting it to other models, understanding
the proposed economic module and ammending it for futher functionality.

The basic functionality of SIMSON can be analysed by skimming this documentation,
the [Bachelor's thesis on SIMSON](bachelor.hosak.pdf) and this github repository.

This section describes how to install SIMSON on your own device.

1. ***Install Python 3***

You can download Python here: https://www.python.org/downloads/ .
For this project, Python 3 is required.

2. ***Clone repository***

Make sure that you have git installed, have generated a public ssh key and added it to your github account.

Clone the repository using `$ git clone git@github.com:Merjo/simson.git` from the parent directory of your choice.

3. ***Install required dependencies***

You can install all required dependencies using the command 
`$ pip install -r requirements.txt`

In some cases you might have to write `pip3` instead of `pip`.

4. ***Load data via submodules***

The original data is stored on the Potsdam Institute
of PIK's Gitlab server. It can not
be shared with members of the public. Hence, active utilisation
of SIMSON is restricted to PIK members. 

They can be added to the 
Gitlab repository internally. Afterwards they can load
the data in the submodules with the command 
`$ git submodule update --recursive --remote`.

5. ***Set default working directory***

The relative paths in the source code are adressed assuming the working
directory to be the main SIMSON folder. Hence you need to set it as 
the default working directory. How this can be established varies on 
your integrated development environment (IDE) like PyCharm or VSCode.

For example, in PyCharm, this can be declared via clicking on the main project folder
and choosing`Run` -> `Edit Configurations..` -> `Edit Configuration Templates...`
-> `Working Directory`.

6. ***Running the code***

Finally, the code can be tested by running some scripts.
Most scripts in the `src` folder have a default test function
that is called when running the script.

A great way to test SIMSON is running `simson_base_model.py`
in `src/base_model` which should result in a *success* output on the
console.

Another possibility is to run `visualise_flows_and_stocks.py`
in `src/visualisation`. In the code, specific parameters can be 
set to get a *matplotlib* plot of predictions for 
specific flows and stocks in the SIMSON MFA (s. Fig. in the [Introduction](Introduction.md)).

*Go to section 3:* [Simulation](Simulation.md)