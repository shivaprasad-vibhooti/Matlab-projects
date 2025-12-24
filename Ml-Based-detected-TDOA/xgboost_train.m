%% 1. Setup & Data Loading
dataFolder = 'Data'; 
ads = audioDatastore(dataFolder, 'IncludeSubfolders', true, 'LabelSource', 'foldernames');
% Refresh datastore to ensure paths are valid
ads = audioDatastore(ads.Files, 'IncludeSubfolders', true, 'LabelSource', 'foldernames');

[trainAds, testAds] = splitEachLabel(ads, 0.8, 'randomized');

%% 2. Feature Extraction Setup
fs = 44100;
targetDuration = 2; 
targetSamples = targetDuration * fs;

% Define the extractor (Matches Simulink MFCC block defaults)
afe = audioFeatureExtractor('SampleRate', fs, ...
    'Window', hamming(1024, 'periodic'), ...
    'OverlapLength', 512, ...
    'mfcc', true, ...
    'mfccDelta', true);

%% 3. Balanced Feature Extraction (Clip-Level)
disp('Extracting and Balancing Features...');
[XTrain, YTrain] = extractSimulinkFeatures(trainAds, afe, targetSamples);
[XTest, YTest] = extractSimulinkFeatures(testAds, afe, targetSamples);

%% 4. Train the Ensemble Model
disp('Training Model (Bagged Trees)...');
t = templateTree('MaxNumSplits', 100); 
trainedModel = fitcensemble(XTrain, YTrain, ...
    'Method', 'Bag', ...
    'NumLearningCycles', 100, ...
    'Learners', t);

%% 5. Command Window Results
predictedLabels = predict(trainedModel, XTest);
accuracy = sum(predictedLabels == YTest) / length(YTest) * 100;

fprintf('\n===========================================\n');
fprintf('  SIMULINK MODEL TRAINING COMPLETE\n');
fprintf('  FINAL TEST ACCURACY: %.2f%%\n', accuracy);
fprintf('===========================================\n');

% Confusion Matrix
figure;
confusionchart(YTest, predictedLabels);
title(['Balanced Simulink Model (Acc: ' num2str(accuracy, '%.1f') '%)']);

% Save for Simulink Predict Block
save('gunshotModel.mat', 'trainedModel');
disp('Success! Load "gunshotModel.mat" into the Simulink Predict block.');

%% --- HELPER: Simulink-Compatible Balanced Extraction ---
function [X, Y] = extractSimulinkFeatures(ads, afe, targetSamples)
    X = []; Y = [];
    reset(ads);
    
    % Group files to calculate balance
    tempGunshotAudio = {};
    tempNonGunshotFeat = [];
    
    while hasdata(ads)
        [audioIn, info] = read(ads);
        if size(audioIn, 2) > 1, audioIn = mean(audioIn, 2); end
        
        if string(info.Label) == "NonGunshot"
            % Process 2s clips, average MFCCs to create ONE row per clip
            for i = 1 : targetSamples : (length(audioIn) - targetSamples + 1)
                snippet = audioIn(i : i + targetSamples - 1);
                f = extract(afe, snippet);
                X = [X; mean(f, 1)]; % Average across time frames
                Y = [Y; info.Label];
            end
        else
            tempGunshotAudio{end+1} = audioIn; %#ok<AGROW>
        end
    end
    
    % Balance Gunshots to match NonGunshot count
    numNeeded = size(X, 1);
    numFiles = length(tempGunshotAudio);
    clipsPerFile = ceil(numNeeded / numFiles);
    
    for k = 1:numFiles
        audioIn = tempGunshotAudio{k};
        [~, peakIdx] = max(abs(audioIn));
        
        % Extract multiple shifted 2s windows around the gunshot peak
        halfWin = floor(targetSamples / 2);
        searchStart = max(1, peakIdx - targetSamples);
        searchEnd = min(length(audioIn) - targetSamples, peakIdx + targetSamples);
        
        hop = max(1, floor((searchEnd - searchStart) / clipsPerFile));
        
        count = 0;
        for j = searchStart : hop : searchEnd
            if count >= clipsPerFile, break; end
            snippet = audioIn(j : j + targetSamples - 1);
            f = extract(afe, snippet);
            X = [X; mean(f, 1)]; 
            Y = [Y; categorical("Gunshot")];
            count = count + 1;
        end
    end
    Y = categorical(Y);
end