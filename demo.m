clear;

rand('seed',1013);

BITs = [16 32 64 96 128 256 512];
bit_num = length(BITs);

%********************************************************************
% load database
% database should contain traindata, testdata, traingnd, testgnd
load('MNIST.mat');
trainnum      = 5000;
% trainnum      = 30000;
testnum       = 1000;
% crop data by trainnum and testnum
traindata    = traindata(1:trainnum,:);
traingnd     = traingnd(1:trainnum,:);
testdata     = testdata(1:testnum,:);
testgnd      = testgnd(1:testnum,:);
cateTrainTest = bsxfun(@eq, traingnd, testgnd');

%***************************************************************
% Feature transformation
%***************************************************************
method    = 'FSDH';
feature   = 'kernel';
n_anchors = 1000;
% Kernel trans
anchor = traindata(randperm(trainnum, n_anchors),:);
sigma  = 0.4; % for normalized data
X      = exp(-sqdist(traindata,anchor)/(2*sigma*sigma));
testX  = exp(-sqdist(testdata,anchor)/(2*sigma*sigma));
%*************************************************************

%*********************************
% iterating by code length L
for i=1:bit_num
    bit = BITs(i);    
    fprintf('bit =%d\n', bit);    
    %***************************************************************
    % run FSDH
    %***************************************************************
    tic;
    [~, R] = FSDH(X, traingnd, bit);
    COP = toc;    
    %****************************************************************
    % Evaluation
    %****************************************************************
    B  =     X*R > 0;
    tH = testX*R > 0;    
    H  = B;
    hammRadius = 2;
    B  = compactbit(H);
    tB = compactbit(tH);    
    hammTrainTest = hammingDist (tB, B)';
    Ret = (hammTrainTest <= hammRadius+1e-8);
    [Pre, Rec] = evaluate_macro(cateTrainTest, Ret);    
    [~, HammingRank]=sort(hammTrainTest,1);
    MAP = cat_apcal(traingnd,testgnd,HammingRank);
    fprintf('Pre=%.3g Rec = %.3g MAP=%.3g training time=%.3g \n', Pre, Rec, MAP, COP);    
end
