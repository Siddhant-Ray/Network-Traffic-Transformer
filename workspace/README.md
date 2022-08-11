# Workspace

We provide an exhaustive guide here to reproduce all experiments to train and evaluate the NTT model. Most of the steps are automated but some will have to be done manually as there are multiple platforms used separately e.g. NS3 for simulations and PyTorch-Lightning for the NTT implementation.

## File descriptions:
The files inside the [TransformerModels](TransformerModels) directory is as follows:

<b> Core files - </b>
* [`encoder_delay.py`](TransformerModels/encoder_delay.py) : Pre-train the NTT by masking the <i> last delay </i> only.
* [`encoder_delay_varmask_chooseencodelem.py`](TransformerModels/encoder_delay_varmask_chooseencodelem.py) : Pre-train the NTT by masking delays after choosing equally from the NTT's output encoded elements.
* [`encoder_delay_varmask_chooseencodelem_multi.py`](TransformerModels/encoder_delay_varmask_chooseencodelem_multi.py) : Pre-train the NTT by masking delays after choosing equally from the NTT's output encoded elements and using multiple decoder instances.
* [`encoder_delay_varmask_chooseagglevel.py`](TransformerModels/encoder_delay_varmask_chooseagglevel.py) : Pre-train the NTT by masking delays after choosing equally from the 3 levels of aggregation for the NTT.
* [`finetune_encoder.py`](TransformerModels/finetune_encoder.py) : Fine-tune the NTT by masking the <i> last delay </i> only.
* [`finetune_encoder_multi.py`](TransformerModels/finetune_encoder_multi.py) : Fine-tune the NTT by masking the <i> last delay </i> but initialize with multiple decoders to match the architecture when pre-trained with multiple decoders.
* [`finetune_mct.py`](TransformerModels/finetune_mct.py) : Fine-tune the NTT to predict the MCT on the given data.
* [`finetune_mct_multi.py`](TransformerModels/finetune_mct_multi.py) : Fine-tune the NTT to predict the MCT on the given data but initialize with multiple decoders to match the architecture when pre-trained with multiple decoders.
* [`generate_sequences.py`](TransformerModels/generate_sequences.py) : Generate the sliding windows for the NTT from the processed NS3 simulations' packet data.
* [`utils.py`](TransformerModels/utils.py) : All utility functions for data pre-processing.
* [`arima.py`](TransformerModels/arima.py) : Train the ARIMA baselines.
* [`lstm.py`](TransformerModels/lstm.py) : Train the Bi-LSTM baselines.
* [`configs`](TransformerModels/configs) : Hyper-paramters for training the NTT model.


<b> Plot files - </b>
* [`plot_losses.py`](TransformerModels/plot_losses.py) : Plot MCT loss curves after fine-tuning the NTT pre-trained on masking the <i> last delay </i> only.
* [`mct_test_plots.py`](TransformerModels/mct_test_plots.py) : Plot MCT loss curves after fine-tuning the NTT pre-trained on masking on variable positions.
* [`plot_predictions.py`](TransformerModels/plot_predictions.py) : Plot historgrams of predictions after pre-training and fine-tuning the NTT.

<b> Others - </b>
* [`transformer_delay.py`](TransformerModels/transformer_delay.py) : A vanilla Transformer encoder-decoder architecture, naively trained on some packet data to predict delays. (this was only for initial insights)

The files inside the [PandasScripts](PandasScripts) directory is as follows:
* [`csvhelper_memento.py`](PandasScripts/csvhelper_memento.py) : Utility script to pre-process raw NS3 outputs to a format, which makes it easier to create the sliding windows and train the NTT.
* [`csv_gendelays.py`](PandasScripts/csv_gendelays.py) : Utility script to pre-process raw NS3 outputs to a format, which makes it easier to create the sliding windows and train the vanilla transformer.

The structure inside the [NetworkSimulators](NetworkSimulators) is as follows:
* [memento](NetworkSimulators/memento): Contains a working copy of ONLY the relevant code files for generating the pre-training and fine-tuning the NTT models. This cannot be run without the full setup, which is self contained in [`MementoDataforNTT`](https://gitlab.ethz.ch/sidray/memento-ns-3-for-ntt). Files inside this [memento](NetworkSimulators/memento) directory, should not be used anymore, except for quick reference.
* [ns3](NetworkSimulators/ns3): This was used for initial insights only, and no results from it have been included in the thesis. 
    - Contains a working copy of the relevant code files for generating the pre-training for the vanilla NTT model, which is authored in [`transformer_delay.py`](TransformerModels/transformer_delay.py). To generate this data, you must install ns3 from scratch as mentioned [here](https://www.nsnam.org/docs/release/3.35/tutorial/singlehtml/index.html#prerequisites).Following which, all the `.cc` files in [ns3](NetworkSimulators/ns3) must be put in the `scratch/` directory. This can be tricky, so we will provide a quicker alternative setup.
    - Alternatively, you can run the script [dockerns3.sh](NetworkSimulators/ns3/dockerns3.sh), and use the files [cptodocker.sh](NetworkSimulators/ns3/cptodocker.sh) and [cpfromdocker.sh](NetworkSimulators/ns3/cpfromdocker.sh) to move the code files and results, in and out of the ns3 container.
    - You can run the files inside the container with the commands: 
        * `export NS_LOG=congestion_1=info` 
        * and then `./waf --run scratch/congestion_1`. 
    - This generates a folder called `congesion_1` with the required data files.
    - For pre-processing, copy the `congesion_1` folder into [PandasScripts](PandasScripts) and run:
        * ```python csv_gendelays.py --model tcponly --numsenders 6```
    - The files can now be added to [`transformer_delay.py`](TransformerModels/transformer_delay.py) and the job can be run.


## To reproduce actual results in the thesis:

### Setup

To run on the ```TIK SLURM cluster```, you need to install ```pyenv```, details for which can be found here: [D-ITET Computing](https://computing.ee.ethz.ch/Programming/Languages/Python). On other clusters, it might be enough to just run 

    $ python -m venv venv

After the environment has been created (created name is `venv` for simplicity):

If it is a pyenv environment, run

    $ eval "$(pyenv init -)"
    $ eval "$(pyenv virtualenv-init -)"
    $ pyenv activate venv

Else run

    $ source venv/bin/acvtivate

Now, install the Python dependencies:

    $ pip install -r requirements.txt

The folder (submodule) [`MementoDataforNTT`](https://gitlab.ethz.ch/sidray/memento-ns-3-for-ntt) contains instructions to generate the training data using NS3 simulations. The module is self contained and will generate a folder called ```results/```, which will contain the required data. To preprocess, copy the ```results/``` folder into the directory [`PandasScripts`](PandasScripts) and run the script (modify the filesnames inside [`csvhelper_memento.py`](PandasScripts/csvhelper_memento.py) if needed):

    $ python csvhelper_memento.py --model memento

This will generate the pre-processed files. The files maybe different, depending on the kind of data generated but all of them will end with ```_final.csv```. Copy all files with this ending, into a folder named ```memento_data/``` and move this folder to the [`TransformerModels`](TransformerModels) directory. 

Copying ```results/``` and ```memento_data/```  to these destinations is needed, else the execution will fail. After copying the files, the training and fine-tuning phase is ready to be initiated.

### Training and fine-tuning:

We need GPUs to run the training and fine-tuning, and this documentation only covers the steps to run on the ```TIK SLURM cluster```. If running on other clusters, the setup might have to be modified a little. We provide a self-contained run script ```run.sh```, in which you can uncomment out the job you want to run. Now you can just execute:

    $ sbatch run.sh 

### Reproduce the plots:

The specific log folders generated after a pre-training or fine-tuning job, must be copied with the EXACT same names, into a ```logs``` directories relative to the [`plot_losses.py`](TransformerModels/plot_losses.py) or [`mct_test_plots.py`](TransformerModels/mct_test_plots.py) files, as displayed in the ```.py``` files. Following that, the plots can be generated as simply as:

    $ python mct_test_plots.py
    $ python plot_losses.py


## Comments:

* SBATCH commands in ```run.sh``` might need to be changed as per memory or GPU requirements.
* For running the ARIMA baselines, GPUs are not needed.
* On the TIK SLURM cluster, sometimes there is the following error ```OSError: [Errno 12] Cannot allocate memory```.
  To fix this:
    - Increase the amount of memory for the job to be run or 
    - Reduce the ```num_workers``` argument in the DataLoader inside the given ```.py``` file from 4 to 1.
* To switch to data from different topologies, you only need to change the ```NUM_BOTTLENECKS``` global varible in the ```.py``` you are running. Note that not all experiments are meant to be run on all topologies. For details on which topology is used for which experiment, refer to [`thesis.pdf`](../report/thesis.pdf) 
* Checkpoints will automatically be saved in the respective log folders for every job (refer to the ```.py``` to see specific names). It is advisable to copy the ```.ckpt``` files into a new folder named ```checkpoints/```, in order to initialise from the trained weights and not lose any work. This relative path ```checkpoints/*.ckpt``` can replaced in the appropriate ```.py``` file. Every fine-tuning ```.py``` file has a global variable ```PRETRAINED``` which can be set to ```True``` if you want to initialize from the saved weights, or ```False``` if fine-tuning must be done <i> from scratch </i>.
* The ```TRAIN``` global variable in the ```.py``` files is used to decide whether to train on the training data, or just test on the testing data.
* The ```trainer API``` from PyTorch lightning (present in the NTT ```.py``` files) is used to select multiple GPUs using the ```strategy``` argument. Possible options are 
    - `dp` : Data Parallel, this works always on the TIK SLURM cluster.
    - `ddp` : Distributed Data Parallel, this only works sometimes and we haven't used this. To run ddp jobs, modify the ```run.sh``` file, to include an `srun` command prior to the `python` command.
* To save files, sometimes you might have to modify the directory and file names in the code, as needed on your machine. As this is not an end-to-end software, somtimes it is not possible to create a generic file saving system across multiple experiments.



