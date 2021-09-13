This is the codes for the model selection experiments in Section 5.5
1, download data from https://zenodo.org/deposit/5059769
2, run modelSelection.py to generate intermediate files in folder stats/
    modelSelection.py $Explainer $Num_runs
    $Explainer can be "integratedGradient", "guidedGradCAM", "guidedBackProp", "saliency", "gradientShap"
    $Num_runs is the number of runs for statistics purposes.
3, use analysis.ipynb to calculate the statistics
