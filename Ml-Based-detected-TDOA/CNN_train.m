%% 1. Setup & Data Loading
dataFolder = 'Data'; 
ads = audioDatastore(dataFolder, 'IncludeSubfolders', true, 'LabelSource', 'foldernames');
ads = audioDatastore(ads.Files, 'IncludeSubfolders', true, 'LabelSource', 'foldernames');

disp('--- Initial Dataset Status ---');
disp(countEachLabel(ads));

[trainAds, testAds] = splitEachLabel(ads, 0.8, 'randomized');

%% 2. Feature Extraction Setup
fs = 44100;
segmentDuration = 1.0; 
targetSamples = round(segmentDuration * fs); 

afe = audioFeatureExtractor('SampleRate', fs, ...
    'Window', hann(1024, 'periodic'), ...
    'OverlapLength', 512, ...
    'melSpectrum', true); 

disp('Extracting Balanced Spectrograms for CNN...');
[XTrain, YTrain] = createBalancedCNNSpectrograms(trainAds, afe, targetSamples);
[XTest, YTest] = createBalancedCNNSpectrograms(testAds, afe, targetSamples);

%% 3. Improved CNN Architecture
inputSize = [size(XTrain,1), size(XTrain,2), size(XTrain,3)];

layers = [
    imageInputLayer(inputSize, 'Name', 'input')
    
    % Layer Block 1
    convolution2dLayer(3, 16, 'Padding', 'same')
    batchNormalizationLayer
    reluLayer
    maxPooling2dLayer(2, 'Stride', 2)
    
    % Layer Block 2 - Added for better feature learning
    convolution2dLayer(3, 32, 'Padding', 'same')
    batchNormalizationLayer
    reluLayer
    maxPooling2dLayer(2, 'Stride', 2)
    
    % Output
    fullyConnectedLayer(2)
    softmaxLayer
    classificationLayer];

options = trainingOptions('adam', ...
    'MaxEpochs', 20, ... % Increased epochs for better convergence
    'MiniBatchSize', 64, ...
    'Shuffle', 'every-epoch', ...
    'Plots', 'training-progress', ...
    'InitialLearnRate', 0.001, ...
    'Verbose', false);

disp('Training CNN Network...');
cnnModel = trainNetwork(XTrain, YTrain, layers, options);

%% 4. Testing & Results (ADDED SECTION)
disp('Calculating Results...');
YPred = classify(cnnModel, XTest);

% 1. Print Accuracy to Command Window
accuracy = sum(YPred == YTest) / numel(YTest) * 100;
fprintf('\n===========================================\n');
fprintf('  CNN TRAINING COMPLETE\n');
fprintf('  FINAL TEST ACCURACY: %.2f%%\n', accuracy);
fprintf('===========================================\n');

% 2. Display Simple Confusion Matrix
figure('Name', 'CNN Confusion Matrix');
confusionchart(YTest, YPred);
title(['CNN Balanced Spectrogram Results (Acc: ' num2str(accuracy, '%.1f') '%)']);

% Save for Simulink
save('gunshotCNNModel.mat', 'cnnModel');

%% --- HELPER: Balanced CNN Extraction ---
function [X, Y] = createBalancedCNNSpectrograms(ads, afe, targetSamples)
    X = []; Y = [];
    reset(ads);
    
    while hasdata(ads)
        [audioIn, info] = read(ads);
        if size(audioIn, 2) > 1, audioIn = mean(audioIn, 2); end
        
        label = string(info.Label);
        
        if label == "NonGunshot"
            % 1 segment per second
            for i = 1 : targetSamples : (length(audioIn) - targetSamples + 1)
                chunk = audioIn(i : i + targetSamples - 1);
                X = cat(4, X, extract(afe, chunk));
                Y = [Y; info.Label];
            end
        else
            % High-density extraction for Gunshots (80% overlap)
            hopSize = floor(targetSamples * 0.2); 
            for j = 1 : hopSize : (length(audioIn) - targetSamples + 1)
                chunk = audioIn(j : j + targetSamples - 1);
                X = cat(4, X, extract(afe, chunk));
                Y = [Y; info.Label];
            end
        end
    end
    Y = categorical(Y);
    fprintf('Extraction Balanced: Gunshots(%d) vs NonGunshots(%d)\n', ...
        sum(Y=="Gunshot"), sum(Y=="NonGunshot"));
end