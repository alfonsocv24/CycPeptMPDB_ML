# CycPeptMPDB_ML
In this repository you can find all the codes used in the experiments of the publication.
To run the code first make sure that you have the required packes.
You can install them in a python environment using:
```bash
  python3 -m venv env # Create virtual environment
  source env/bin/activate # Activate environment
  pip install -r requirements.txt # Install dependencies using pip
```
With the requirements ready, you can run any of the codes using:
```bash
  python Model_<Model_name>.py
```
Where <Model_name> relates to the folder in which the model is stored.
In the case of the SP model on the SP folder, there are three main flags:

  - ds: Provide a dataset name. AllPep of L67 are the options.
    ```bash
    python Model_SP.py -ds AllPep
    ```
  
  - cyc: When used, Cyclic Permutations will be used.
    ```bash
    python Model_SP.py -ds L67 -cyc
    ```

  - t: When used, trained weights will be loaded.
    ```bash
    python Model_SP.py -ds L67 -t
    ```
