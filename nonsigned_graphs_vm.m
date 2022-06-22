% Parts of code adapted from Tooley et al. (2020, 2022)
% Graph-measure extraction for graphs with no negative edges or thresholded at top 25, 30 & 40% of edges
data_dir = "/home/gatavins/brain_age/data";
bct_dir = "/home/gatavins/BCT/2019_03_03_BCT";
addpath(data_dir,bct_dir);
subjects = readtable(strcat(data_dir,"/func_fcon-parcWITHAGE-320x55279_20220615.csv"));
atlas_parcID = fopen(strcat(data_dir,"/gordon333CommunityAffiliation.1D"));
atlas_parc = textscan(atlas_parcID,"%u");
atlas_communities = double(atlas_parc{1})';

% preparing table for results
cols = ["sub_method","method","age",...
    genvarname(repelem("degree",333),"degree"),...
    genvarname(repelem("strength",333),"strength"),...
    genvarname(repelem("clustering",333),"clustering"),...
    genvarname(repelem("part_coef_avg",333),"part_coef_avg"),...
    genvarname(repelem("wmod_zscore",333),"wmod_zscore"),...
    genvarname(repelem("betweenness_c",333),"betweenness_c"),...
    "avgweight","glob_clustering","glob_participation",...
    "glob_efficiency","modularity","optim_numcom",...
    "small_worldness"];
colcount = width(cols);
rowcount = 4 * height(subjects);
vartypes = ["string","string",string(repelem({'double'},2006))];
size = [rowcount colcount];
results = table('Size',size,'VariableTypes',vartypes,...
        'VariableNames',cols);
minsize=[4 colcount];
    
for m=1:height(subjects)    
% 2 rounds: (1) no negative edges, 
%           (2) thresholded edges (top 25, 30, 40%)
for i=1:4
n = 4*(m-1)+i;
% storing id, age, and FC vector
results(n,1) = subjects(m,1);
results(n,3) = subjects(m,end);
vect = table2array(subjects(m,2:end-1));

% reshaping vectors into 333 x 333 symmetric matrices
fcmat = zeros(333);
fcmat(triu(ones(333),1)==1) = vect;
fcmat(tril(ones(333),-1)==1) = fcmat(triu(ones(333),1)==1)';
zfcmat = atanh(fcmat); % Fisher Z-transform

% assigning method titles in 2nd column
switch i
    case 1
        results{n,2} = "non-neg";
        zfcmat = threshold_absolute(zfcmat,0);
    case 2
        results{n,2} = "25perc";
        zfcmat = threshold_proportional(zfcmat,0.25);
    case 3
        results{n,2} = "30perc";
        zfcmat = threshold_proportional(zfcmat,0.30);
    case 4
        results{n,2} = "40perc";
        zfcmat = threshold_proportional(zfcmat,0.40);
end

% node-level measures
    % degree
    results{n,4:336} = degrees_und(zfcmat);
    
    % strength
    results{n,337:669} = strengths_und(zfcmat);
    
    % clustering coefficient 
    clust_coef = clustering_coef_wu(zfcmat);
    results{n,670:1002} = clust_coef';
    
    % participation coefficient with Gordon atlas
    pcoef = participation_coef(zfcmat,atlas_communities)';
    results{n,1003:1335} = pcoef;
    
    % within-module degree z-score with Gordon atlas
    results{n,1336:1668} = module_degree_zscore(zfcmat,0)';
    
    % betweenness centrality
    results{n,1669:2001} = (betweenness_wei(zfcmat)/((332*331)/2))';
    
% global measures
    results{n,2002} = mean(mean(zfcmat~=0));
    
    glob_clust = mean(clust_coef); % avg clustering coef 
    results{n,2003} = glob_clust;
    
    results{n,2004} = mean(pcoef); % avg PC
    
    results{n,2005} = efficiency_wei(zfcmat,0); % global efficiency

    % modularity measures (Louvain with negative asym & quality index)
    % using 'negative_asym' as per Rubinov & Sporns (2011) recommendation
    modvector = zeros(2,100);
    for k=1:100
         [C, Q] = community_louvain(zfcmat,1,[],'negative_asym');
         modvector(1,k) = max(C);
         modvector(2,k) = Q;
    end
    
    mod_mean = mean(modvector,2);
    results{n,2006} = mod_mean(2,1);
    results{n,2007} = mod_mean(1,1);
    
    % small-worldness calculation
    results{n,2008} = glob_clust / charpath(zfcmat);
end
end

writetable(results,strcat(data_dir,"graph_measures_nonsigned.csv"));
