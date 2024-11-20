# A Song of Privacy and Utility: A Comparative Evaluation of De-Identified and Synthetic Data
This repository contains the results and the implementation of the de-identified and synthetic dataset comparison framework proposed in the paper "A Song of Privacy and Utility: A Comparative Evaluation of De-Identified and Synthetic Data".

## Abstract
Practitioners are faced with two different strategies to reduce privacy risks in privacy-preserving data publishing. 
On the one hand, the tailored de-identification of datasets involves reducing the link to data subject identity and as such incurs less risk when sharing these de-identified datasets. 
While this approach is widely used and legally endorsed by data privacy regulations, it requires significant human expertise and manual effort to define and generalize identifiable attributes. 
On the other hand, the capabilities of Generative AI allow creating synthetic datasets that bear similarity to the original, but are in fact not representative of actual data subjects. 
Generating synthetic datasets relies more on automation and computational power, though it still requires AI expertise to set up hyper-parameters and optimization methods.

Given the distinct nature of both strategies, there is an ongoing and unresolved debate about which yields the best data privacy protection and utility results.
Resolving this debate requires a unified and scientific comparison framework that comprehensively examines the metrics of both approaches.
To address this gap, we present (i) a novel quantitative evaluation framework to facilitate privacy and utility comparisons between synthetic and de-identified datasets, and (ii) the results of a comparative study of both approaches applied in two distinct use cases, using the *Patients* and *ACSIncome* datasets. 

Our study results indicate that de-identified datasets outperform synthetic data in privacy protection, while synthetic data achieves comparable utility to the de-identified data. 
Thus, although synthetic data generation is a novel approach to data protection, it has yet to surpass de-identified data. 
The targeted de-identification of datasets continues to prove effective and private.

## Repository Content
### Datasets
The **Datasets** folder contains both the ACSIncome dataset and the Patients dataset used within the paper. Additionally, dataset required for the Patients optimization use-case are included.

### Hierarchies
The **Hierarchies** folder contains the hierarchies used for creating de-identified dataset and calculating categorical distances for both datasets.

### intellij
The **intellij** folder contains an intellij project implementing de-identification by utilizing the ARX anonymization library.
From this code a .jar file is constructed used in the generation of de-identified datasets.

### pycharm
The **pycharm** folder contains all the python code which can be used to reproduce the results. This folder is structure as follows:
*	**dataset_preparation**: contains code to obtain the ACSIncome dataset and preprocess both the ACSIncome and Patients datasets (results in ACSIncome.csv and Patients_cleaned.csv)
*	**privacy**: contains the privacy metric implementations and additional helper functions to perform the metric calculations
*	**utility**: contains the use-case specific and statistical utility metric implementations
*	**DistanceMetrics**: contains the custom distance calculations
*	**pipeline**: contains code to run all experiments in order
*	**results_condenser**: contains code to merge the obtained results in a limited set of files

### output
Contains the results obtained from running the pipeline and results_condenser.
