# <ins>Mol</ins>ecular-<ins>K</ins>ernel <ins>G</ins>raph <ins>N</ins>eural <ins>N</ins>etwork (MolKGNN)
By [Yunchao "Lance" Liu](www.LiuYunchao.com), [Yu Wang](https://yuwvandy.github.io/), [Oanh Vu](https://www.linkedin.com/in/oanhvu/), [Bobby Bodenheimer](http://www.vuse.vanderbilt.edu/~bobbyb/), [Jens Meiler](https://www.linkedin.com/in/jens-meiler-4b635339/), [Tyler Derr](https://tylersnetwork.github.io/)

This repository is the official implementation of MolKGNN in paper [*Interpretable Chirality-Aware Graph Neural Network for Quantitative Structure Activity Relationship Modeling in Drug Discovery*](https://www.biorxiv.org/content/10.1101/2022.08.24.505155v1).

Please cite our paper if you find MolKGNN useful in your work:

<pre>
@article {liu2022molkgnn,
	author = {Liu, Yunchao (Lance) and Wang, Yu and Vu, Oanh and Moretti, Rocco and Bodenheimer, Bobby and Meiler, Jens and Derr, Tyler},
	title = {Interpretable Chirality-Aware Graph Neural Network for Quantitative Structure Activity Relationship Modeling in Drug Discovery},
	year = {2022},
	doi = {10.1101/2022.08.24.505155},
	journal = {bioRxiv}
} 
</pre>

MolKGNN is a deep learning model based on Grah Neural Networks (GNNs) for molecular representation learning. It features in:
1. SE(3)-invariance
2. Conformation-invariance
3. Interpretability
![mol_conv](https://user-images.githubusercontent.com/5760199/186030531-6bd363d4-73da-414b-8cb7-d4b136dd3812.png)

<p align="center">
  <img src="https://user-images.githubusercontent.com/5760199/186030469-28661a4e-ff48-43b3-b707-885629791032.png" />
</p>

My [blog](https://medium.com/@YunchaoLanceLiu/molkgnn-extending-convolution-to-molecules-b94a4d51f39f) explaining this paper. 


# Acquire the Datasets

This repository does NOT include the datasets used in the experiment. Please download the datasets from [this link](https://figshare.com/articles/dataset/Well-curated_QSAR_datasets_for_diverse_protein_targets/20539893)

These are well-curated realistic datasets that removes false positves for a diverse important drug targets. The datasets also feature in its high imbalance nature (much more inactive molecules than active ones). Original papers of the datasets: see references [1,2]. 

**Introduction of the Datasets**

High-throughput screening (HTS) is the use of automated equipment to rapidly screen thousands to millions of molecules for the biological activity of interest in the early drug discovery process [3]. However, this brute-force approach has low hit rates, typically around 0.05\%-0.5\% [4]. Meanwhile, PubChem [5] is a database supported by the National Institute of Health (NIH) that contains biological activities for millions of drug-like molecules, often from HTS experiments. However, the raw primary screening data from the PubChem have a high false positive rate [6]. A series of secondary experimental screens on putative actives is used to remove these. While all relevant screens are linked, the datasets of molecules are often not curated to list all inactive molecules from the primary HTS and only confirmed actives after secondary screening. Thus, we identified nine high-quality HTS experiments in PubChem covering all important target protein classes for drug discovery. We carefully curated these datasets to have lists of inactive and confirmed active molecules. 

**Statistics of the Datasets, specified by PubChem Assay ID (AID)**

<p align="center">
  <img src="https://user-images.githubusercontent.com/5760199/186287898-30e5d105-6d80-4580-af9f-3044d9b2c8f8.png" />
</p>

# Process the Datasets


Uncompress the downloaded file and you will see several .sdf files. Create a folders according to the diagram below. Place all .sdf files **<em>raw</em>** folder. 
You can use the `dataset_multigenerator.py` to process all of them in parallel into [PyG's InMemoryDataset](https://pytorch-geometric.readthedocs.io/en/latest/notes/create_dataset.html), as shown below.

`python dataset_multigenerator.py`

The processed data will appear in the **<em>processed</em>** folder

<pre>
root_dir

|--dataset

|  |--qsar
|  |  |--clean_sdf
|  |  |  |--processed
|  |  |  |  |--kgnn-based-{dataset_AID}-3D.pt
|  |  |  |--raw
|  |  |  |  |--{dataset_AID}_actives_new.sdf
|  |  |  |  |--{dataset_AID}_inactives_new.sdf
  
|--kgnn

  |--entry.py

  |--*.py

</pre>




# Run the Codes

Here is an exmaple for running the code:

`python entry.py --dataset_name 1798 --dataset_path ../dataset --num_workers 16 --enable_oversampling_with_replacement --warmup_iterations 200 --max_epochs 3 --peak_lr 5e-2 --end_lr 1e-9 --batch_size 16 --default_root_dir actual_training_checkpoints --num_layers 3 --num_kernel1_1hop 10 --num_kernel2_1hop 20 --num_kernel3_1hop 30 --num_kernel4_1hop 50 --num_kernel1_Nhop 10 --num_kernel2_Nhop 20 --num_kernel3_Nhop 30 --num_kernel4_Nhop 50 --node_feature_dim 27 --edge_feature_dim 7 --hidden_dim 32 --seed 1 --task_comment "this is a test"`

# Q&A

Feel free to drop questions in the **<em>Issues</em>** tab, or contact me at yunchao.liu@vanderbilt.edu
  

# References
[[1] Butkiewicz, Mariusz, et al. "Benchmarking ligand-based virtual High-Throughput Screening with the PubChem database." Molecules 18.1 (2013): 735-756.](https://www.mdpi.com/1420-3049/18/1/735))

[[2] Butkiewicz, Mariusz, et al. "High-throughput screening assay datasets from the pubchem database." Chemical informatics (Wilmington, Del.) 3.1 (2017).](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5962024/)

[[3] Bajorath, Jürgen. "Integration of virtual and high-throughput screening." Nature Reviews Drug Discovery 1.11 (2002): 882-894.](https://www.nature.com/articles/nrd941)

[[4] Mueller, Ralf, et al. "Identification of metabotropic glutamate receptor subtype 5 potentiators using virtual high-throughput screening." ACS chemical neuroscience 1.4 (2010): 288-305.](https://pubs.acs.org/doi/full/10.1021/cn9000389)

[[5] Kim, Sunghwan, et al. "PubChem in 2021: new data content and improved web interfaces." Nucleic acids research 49.D1 (2021): D1388-D1395.](https://academic.oup.com/nar/article-abstract/49/D1/D1388/5957164)

[[6] Baell, Jonathan B., and Georgina A. Holloway. "New substructure filters for removal of pan assay interference compounds (PAINS) from screening libraries and for their exclusion in bioassays." Journal of medicinal chemistry 53.7 (2010): 2719-2740.](https://pubs.acs.org/doi/abs/10.1021/jm901137j)
