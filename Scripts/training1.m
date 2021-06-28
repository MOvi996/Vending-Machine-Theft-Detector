
clc;
clear all;
training_directory = 'D:\CV Projects';

imds = imageDatastore(fullfile(training_directory),...
    'IncludeSubfolders',true,...
    'LabelSource','foldernames');

[imdsTrain,imdsValidation] = splitEachLabel(imds,0.70,'randomized');

total_numimages = numel(imdsTrain.Labels) + numel(imdsValidation.Labels);
num_Train_Images = numel(imdsTrain.Labels);
num_Test_Images = numel(imdsValidation.Labels);

idx = randperm(num_Train_Images,16);


figure
for i = 1:16
    subplot(4,4,i)
    I = readimage(imdsTrain,idx(i));
    imshow(I)
end

%  net = alexnet;
network=load('D:\NUST Sem 7\Computer Vision\Semester Project\vgg19.mat');
net=network.net;
  
%protofile = 'deploy_alexnet_places365.prototxt';
%datafile = 'alexnet_places365.caffemodel';
%net = importCaffeNetwork(protofile,datafile)
% analyzeNetwork(net)
input_Size = net.Layers(1).InputSize
num_Classes = numel(categories(imdsTrain.Labels))
layersTransfer = net.Layers(1:end-3);

layers = [
    layersTransfer
    fullyConnectedLayer(num_Classes,'WeightLearnRateFactor',20,'BiasLearnRateFactor',20)
    softmaxLayer
    classificationLayer];
pixelRange = [-30 30];
imageAugmenter = imageDataAugmenter( ...
    'RandXReflection',true, ...
    'RandXTranslation',pixelRange, ...
    'RandYTranslation',pixelRange);
augimdsTrain = augmentedImageDatastore(input_Size(1:2),imdsTrain, ...
    'DataAugmentation',imageAugmenter);
augimdsValidation = augmentedImageDatastore(input_Size(1:2),imdsValidation);
options = trainingOptions('sgdm', ...
    'MiniBatchSize',16, ...
    'MaxEpochs',2, ...
    'InitialLearnRate',1e-4, ...
    'Shuffle','every-epoch', ...
    'ValidationData',augimdsValidation, ...
    'ValidationFrequency',10, ...
    'Verbose',false, ...
    'Plots','training-progress',...
    'CheckpointPath',training_directory,...
    'ExecutionEnvironment','cpu');

 
netTransfer = trainNetwork(augimdsTrain,layers,options);

% save netTransfer
% any other formats other than .mat ? that the model can be saved as
matfile = fullfile(training_directory, 'mineTransfer');
%save(matfile,'v7.3');

%% Testing
[YPred,scores] = classify(netTransfer,augimdsValidation);

% idx = randperm(numel(imdsValidation.Files),4);
% figure
% for i = 1:4
%     subplot(2,2,i)
%     I = readimage(imdsValidation,idx(i));
%     imshow(I)
%     label = YPred(idx(i));
%     title(string(label));
% end

YValidation = imdsValidation.Labels;

% detect the wrong detected file names
dx=[];
v=1;

wrong_filename={};
for j = 1:num_Test_Images
    if YPred(j) ~= imdsValidation.Labels(j)
    wrong_filename{v}=imdsValidation.Files(j);
    v=v+1;
    dx = [dx j] ;
    else 
    end
end
num_of_wrong_detected = numel(wrong_filename);
figure
for ii = 1:num_of_wrong_detected
    for tt = ii:4
  subplot(2,2,tt)
    jj = readimage(imdsValidation,dx(ii));
    imshow(jj)
    label1 = YPred(dx(ii));
    label2 = YValidation(dx(ii));
     set(get(gca,'ylabel'),'rotation',0)
%    ylabel(['Predicted class' (string(label1)) 'Original class ' (string(label2))]  );
% xlabel(wrong_filename{ii} );
    end
end

diary info.txt
diary on
input_Size
num_Classes
countEachLabel(imds)
total_numimages
num_Train_Images
num_Test_Images
num_of_wrong_detected
accuracy = mean(YPred == YValidation)
idx = randperm(numel(imdsValidation.Files),4);
figure
diary off
