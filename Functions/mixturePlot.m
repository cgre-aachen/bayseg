function mixturePlot(MU,COV,field_value,latent_field,labels)
% this function is used for producing mixture plot

n_var = size(MU,2);
n_cluster = size(MU,1);
Combinations = nchoosek(1:n_var,2);
n_plot = size(Combinations,1);
%labels = {'F760','LAI','EMI'};
% labels = {'NDVI z-score','log(EMI) z-score'};

clr_idx = round((size(jet,1)-1)/(n_cluster-1)*((1:n_cluster)-1)+1);
clr_matrix = jet;
clr_matrix = clr_matrix(clr_idx,:);

%figure;
if n_plot>1
    for i = 1:n_plot
        subplot(1,n_plot,i);
        XYlim = [min(field_value(:,Combinations(i,1))) max(field_value(:,Combinations(i,1))) min(field_value(:,Combinations(i,2))) max(field_value(:,Combinations(i,2)))];
        gscatter(field_value(:,Combinations(i,1)),field_value(:,Combinations(i,2)),latent_field,clr_matrix);
        hold on;
        for j = 1:n_cluster
            obj = gmdistribution(MU(j,Combinations(i,:)),COV(Combinations(i,:),Combinations(i,:),j),1);
            ezcontour(@(x1,x2)pdf(obj,[x1 x2]),XYlim)%,get(gca,{'XLim','YLim'}));
        end
        hold off;
        title(['feature ',num2str(Combinations(i,1)),' vs. ','feature ',num2str(Combinations(i,2))]);
        xlabel(labels{Combinations(i,1)});
        ylabel(labels{Combinations(i,2)});
    end
else
    i = 1;
    XYlim = [min(field_value(:,Combinations(i,1))) max(field_value(:,Combinations(i,1))) min(field_value(:,Combinations(i,2))) max(field_value(:,Combinations(i,2)))];
    gscatter(field_value(:,Combinations(i,1)),field_value(:,Combinations(i,2)),latent_field,clr_matrix);
    hold on;
    for j = 1:n_cluster
        obj = gmdistribution(MU(j,Combinations(i,:)),COV(Combinations(i,:),Combinations(i,:),j),1);
        ezcontour(@(x1,x2)pdf(obj,[x1 x2]),XYlim)%,get(gca,{'XLim','YLim'}));
    end
    hold off;
    title(['feature ',num2str(Combinations(i,1)),' vs. ','feature ',num2str(Combinations(i,2))]);
    xlabel(labels{Combinations(i,1)});
    ylabel(labels{Combinations(i,2)});
end
    
end