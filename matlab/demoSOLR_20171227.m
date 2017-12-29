clear all
addpath(genpath('./'))
FontSize=10;

%set the situation of the classification problem in the demo.
numClass=5;
numSampleForEachClass=5;
centerForEachClass=[-2 -1 0 1 4];
numFeatureDims=100;

%make data for validation
feature=[];
label=[];
for index_class=1:numClass
    temporal_feature=zeros(numSampleForEachClass,numFeatureDims);
    temporal_feature(:,1)=centerForEachClass(index_class).*2;
    temporal_feature(:,2)=centerForEachClass(index_class).*1;
    feature=[feature;temporal_feature];
    label=[label;ones(numSampleForEachClass,1).*index_class];
end
feature4training=feature+randn(size(feature)).*1;
feature4test=feature+randn(size(feature)).*1;

%make the scatter plot in the feature space
%the first and second dimension have information for classifcation, other
%dimensions are irrelevant.
%Classifiers must ignore irrelevant dimensions for good classifications
figure
%subplot(2,1,1)
subplot('Position',[0.15 0.55 0.4 0.4])
hold on
for index_class=1:5
    switch index_class
        case 1
            str4color='m';
        case 2
            str4color='b';
        case 3
            str4color='g';
        case 4
            str4color='y';
        case 5
            str4color='r';
    end
    handle4legend(index_class)=scatter(feature4training(label==index_class,1),feature4training(label==index_class,2),10,str4color); 
end
hold off
handle4legend=legend(handle4legend,{'Rank 1','Rank 2','Rank 3','Rank 4','Rank 5'});
set(handle4legend,'Box','off','FontSize',FontSize,'Location','NorthEastOutside')
set(gca,'Box','off','TickDir','out','TickLength',[0.02 0.02])
xlim([-15 15])
ylim([-15 15])
xlabel('Dimension 1','FontSize',FontSize)
ylabel('Dimension 2','FontSize',FontSize)
set(gca,'XTick',[-100:10:100],'FontSize',FontSize)
set(gca,'YTick',[-100:10:100],'FontSize',FontSize)
axis square


%Training of sparse ordianl logistic regression
model=SOLRtrain(feature4training,label);
%Prediction for test data
[predictedLabel_solr P]=SOLRpredict(feature4test,model);
score4SOLR=(sum(predictedLabel_solr==label)./length(predictedLabel_solr)).*100;

%plot the weights obtained
subplot(2,2,3)
weight=zeros(1,size(feature,2));
weight(model.effectiveDim)=model.beta;
bar(weight)
xlabel('Feature dimension','FontSize',FontSize)
ylabel('Weight value','FontSize',FontSize)
set(gca,'XTick',[1 20:20:numFeatureDims],'FontSize',FontSize)
set(gca,'YTick',[0:1:10],'FontSize',FontSize)
xlim([0 numFeatureDims+1])
ylim([0 max(model.beta)+0.1])
set(gca,'Box','off','TickDir','out','TickLength',[0.02 0.02])



%plot the accuracy
subplot(2,2,4)
bar([score4SOLR])
ylabel('% correct','FontSize',FontSize)
xlabel('SOLR', 'FontSize', FontSize)
set(gca,'YTick',[0:25:100],'FontSize',FontSize)
set(gca,'XTick',[])
xlim([0 2])
ylim([0 100])
set(gca,'Box','off','TickDir','out','TickLength',[0.02 0.02])

saveas(gcf,'figDemoSOLR_20171227.png','png')
