% Load data
data  = readcell('../../Data/labelled_Genomic_Data.xlsx');

genomic_data = cell2mat(data(2:end, 4:end));

RFS_times = cell2mat(data(2:end, 3));

RFS_med = median(RFS_times);

bin_RFS = RFS_times > RFS_med;


% PCA analysis
cov_genomic = cov(genomic_data);

[Vd, Ve] = eig(cov_genomic, 'vector');

plot(sort(Ve, 'descend'))

[coeff, score, latent] = pca(genomic_data);

Y = tsne(genomic_data);
gscatter(Y(:,1), Y(:,2), bin_RFS);


% Trying to bin data into 5 groups
[clusters, E] = discretize(RFS_labels, 5);

gscatter(Y(:,1), Y(:,2), clusters);