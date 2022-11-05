# **Neural Network Reconstruction of higher-dimensional Plasma Space-Time**
This repository allows recovering full higher-dimensional (2D space + 1D time) Plasma State Vectors from partial samples along linear trajectories, mimicking spacecraft observations in space. This is approached by injecting the ideal Magnetohydrodynamics Equations into different kinds of Neural Networks to create innovative Physics-Informed Neural Networks while parallelly applying novel curriculum training approaches.

## **Data**
The data has to be of Hierarchical Data Format Version 5 (HDF5, .h5).<br />
It must include four subsets: _U, x, y, t_ where _U_ is of form _[8, Len(x), Len(y), Len(t)]_ and contains the full eight-dimensional Magnetohydrodynamics (Plasma) States.<br />

Example Dataset (LW3 Benchmark): [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.7116590.svg)](https://doi.org/10.5281/zenodo.7116590)



## **Installation**
Even when it is not required, it is beneficial to use virtual python environments. Hence, feel free to set up an environment and activate it prior to installing all necessary packages.

```bash
(base) user: ./$ python3.8 -m venv ./env_name
(base) user: ./$ source ./env_name/bin/activate
(env_name) user: ./$ 
```

To use the "Neural Network Reconstruction of higher-dimensional Plasma Space-Time" framework, one has to prepare and set up the execution environment by installing the necessary dependencies. Please install all needed packages by the following command.
```bash
(env_name) user: ./$ pip3 install -r requirements.txt
```

## **Tracking**

### **MLFlow**
In order to track experiment results, we use an MLFlow server.<br />
Start it by navigating to the *src/* directory and running the command:

`mlflow server`<br />
For the following examples, we assume that the server is running at:

`http://127.0.0.1:5000`

If the MLFlow server is running at another address one can pass it later as an [additional parameter](#optional-arguments) as follows:

``` bash
(env_name) user: [...] --tracking-uri AddressOfMLFLowServer
```

**For tracking artifacts to the mlflow server, one needs to have an authorized SSH-connection to the machine on which the server is running!**

_Note: For parallelly storing results of __many__ configurations, we recommend to setup a database backend following the [instructions](https://mlflow.org/docs/latest/tracking.html)_. 

### **HPO Configuration**
We run the HPO using the [optuna framework](https://optuna.org/).<br />
To keep track of the progress of single studies and trials, we recommend setting up a database backend that can handle multiple simultaneous read and write operations. In our experiments, we use [MySQL](https://www.mysql.com/de/) for this.<br />
More information on how to integrate a database backend can be found at the _storage parameter_ in the [Optional Arguments section](#optional-arguments).

### **Search Spaces**
All defined hyperparameters can be found in the `./src/hpo/hpo*.py` files.<br />
Individual search spaces can be set [there](./src/hpo/).

## **Running HPO for Reconstruction**
For all experiments, the computed metrics are logged to the running MLFlow instance. Furthermore, the best-trained model(s) of every experiment run is tracked by the MLFlow instance as _artifact_.

``` bash
(env_name) user: ./src$ python3 main.py [...]
```

_Note: One can abort the script at any time, and inspect the current results via the web interface of MLFlow. It is always possible to resume an already started HPO study!_

### **Optional Arguments**
* **-v, --version**<br />
Show the program's version number and exit

* **-c, --config []** <br />
Path to config file

* **-p, --problem [../data/problem.h5]**<br /> 
MHD problem/benchmarks that should be reconstructed<br />
Path to 2D MHD-datafile (.h5)<br />

* **-m, --model [pinn]**<br />
Model/architecture to run the HPO for<br />
Options: 'cgan', 'cwgangp', 'knn', 'mlp', 'pinn'

* **-am, --augmentation-model [None]**<br />
Path to augmentation model<br />
Must be stored as .pth (pytorch model) or .joblib (no-pytorch model) file<br />
No-pytorch models must have a method "forward" to map spacetime -> MHD state<br />
If augmentation-model = None, training data will not be augmented

* **--curr-method [None]**<br />
Method of the curriculum learning<br />
Options: None, 'colloc_inc_points', 'colloc_cuboid', 'colloc_cylinder', 'phys', 'trade_off', 'coeff', 'num_diff', 'hpo'<br />
None: no curriculum learning<br />
'colloc_inc_points': stepwise increase of the number of sampled collocation points<br />
'colloc_cuboid': stepwise shift or expansion of the spacetimes for the collocation point sampling along either x, y, or t axis<br />
'colloc_cylinder': stepwise extension of the spacetimes for the collocation point sampling in concentric circles around one or more spacecraft trajectories<br />
'phys': stepwise addition of MHD equations<br />
'trade_off': schedules the trade-off parameter weighting the physical loss<br />
'coeff': schedules the viscosity and resistivity coefficients<br />
'num_diff': schedules the deltas dx, dy, dt for calculating the derivatives<br />
'hpo': HPO decides which method to use

* **--curr-steps [30]**<br />
Number of curriculum steps

* **--curr-fraction-of-total-epochs [0.3]**<br />
Percentage of the overall epochs that is used for curriculum learning

* **--curr-factor-total-points [1]**<br />
Only relevant if curr-method = 'colloc_inc_points'<br />
Mltiplier determining how many collocation points are sampled in the last curriculum step<br />
Basis: number of points in the training data

* **--curr-axis [hpo]**<br />
Only relevant if curr-method = 'colloc_cuboid'<br />
Axis along which the spacetimes for the collocation point sampling will be shifted or expanded<br />
Options: 'x', 'y', 't' 'hpo' (HPO decides which axis to use)

* **-ssh-alias [master]**<br />
SSH alias<br />
Connection must be authorized!

* **--tracking-uri [http://localhost:5000]**<br />
Tracking URI for mlflow experiments and runs

* **--artifact-uri []**<br />
URI to track artifact/models in mlflow

* **--storage [mysql+pymysql://root:password@ip_address:3306/database]**<br />
Database URL<br />
If storage = None, in-memory storage is used, and the study will not be persistent<br />
For a very lightweight storage one can use SQLite _(NOT RECOMMENDED!)_:<br />
    &nbsp;&nbsp;&nbsp;&nbsp; Advantage: no need for an additionally installed backend database<br />
    &nbsp;&nbsp;&nbsp;&nbsp; Disadvantage: could cause blocking errors when working with multiple processes or distributed machines in general<br />
    &nbsp;&nbsp;&nbsp;&nbsp; Example value: "sqlite:///{}.db".format(studies/hpo)

* **--exp-name [Default]**<br />
Name of mlflow experiment

* **--study-name [Default]**<br />
Name of optuna study (HPO)

* **--eval-metric [mse]**<br />
Metric to decide which model is the best used for HPO<br />
Options: 'mse', 'mae', 'pc', 'mare'

* **--intermediate-steps [1000]**<br />
Number of epochs after which an intermediate evaluation of pytorch models is executed

* **-f, --fraction [0.2]**<br />
Percentage of the data that should be kept used for reducing the resolution of the evaluation/validation data<br />
Range of values: ]0; 1]

* **--fraction_lbfgs [0]**<br />
Fraction of epochs for which the LBFGS optimizer will be used (in the end of the training process)<br />
Range of values: [0; 1]

* **--plane []**<br />
Determines if the training data should be a 2D plane

* **--x-bounds []**<br />
Only used if argument "plane" is set.<br />
Two boundaries for the plane (training data) along x axis<br />
First parameter: start value (float)<br />
Second parameter: end value (float)<br />
Both parameters will be set to the closest value in x if they are not in x.

* **--y-bounds []**<br />
Only used if argument "plane" is set
Two boundaries for the plane (training data) along y axis<br />
First parameter: start value (float)<br />
Second parameter: end value (float)<br />
Both parameters will be set to the closest value in y if they are not in y.

* **--t-bounds []**<br />
Only used if argument "plane" is set<br />
Two boundaries for the plane (training data) along t axis<br />
First parameter: start value (float)<br />
Second parameter: end value (float)<br />
Both parameters will be set to the closest value in t if they are not in t.

* **--n-points [100]**<br />
If parameter "plane" is set: number of points in the training data: n_points^2<br />
If parameter "plane" is not set: number of points in the training data: n_points _x_ n_trajs

* **--random-trajs []**<br />
Determines sampling strategy for trajectories for training data<br />
If True: Trajectories are randomly sampled<br />
If False: Trajectories include whole x, y, t domains<br />
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Every trajectory will then be the same<br />
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Recommendation: use only one trajectory<br />

* **--space_usage_non_random_traj [0.5]**<br />
Percentage of the space domain from which a non-random trajectory will be sampled from<br />
Only used if random_trajs = False

* **--n-trajs [4]**<br />
Only used if parameter "plane" is not set<br />
Number of trajectories to sample (each trajectory consists of n_points many points)

* **--noise []**<br />
Determines if Gaussian noise should be added to the training data

* **--no-cuda []**<br />
Do not use cuda even if one is available

* **--dx [0.001]**<br />
Delta along x axis used for the numerical differentiation to calculate the physical loss

* **--dy [0.001]**<br />
Delta along y axis used for the numerical differentiation to calculate the physical loss

* **--dt [0.001]**<br />
Delta along t axis used for the numerical differentiation to calculate the physical loss

* **-s, --seed [0]**<br />
Reproducibility seed for numpy and torch modules

* **--n-trials [10]**<br />
Number of HPO trials


### **Sequential HPO**:
To run multiple successive reconstruction processes, one can use the shell script. To do so, first navigate to the *src/* directory and run the following command:

``` bash
(env_name) user: ./src$ chmod +x sequential_hpo.sh
```

One can then start multiple processes as follows:

``` bash
(env_name) user: ./src$ ./sequential_hpo.sh
```

By default, it will start one HPO process.

### Optional arguments
* **path/to/config.ini**<br />
Path to config file

* **n_processes**<br />
Number of processes that are started <br />

_Note: One cannot set the amount of processes without passing a config file first!_

### **Parallelized HPO**:
To run multiple parallel reconstruction processes, one can use the shell script. To do so, first navigate to the *src/* directory and run the following command:

``` bash
(env_name) user: ./src$ chmod +x parallel_hpo.sh
```

One can then start multiple processes as follows:

``` bash
(env_name) user: ./src$ ./parallel_hpo.sh
```

By default, it will start one HPO process for every available CPU.

### Optional Arguments
* **path/to/config.ini**<br />
Path to config file

* **max_cpus**<br />
Limit of processes that are started <br />

_Note: One cannot limit the amount of processes without passing a config file first!_

## **Creating Plots**
All plots can be reconstructed in the [visualization notebook](https://gitlab.lrz.de/marcusm/neural-network-reconstruction-of-higher-dimensional-plasma-space-time-2.0/-/blob/main/src/visualization.ipynb).
