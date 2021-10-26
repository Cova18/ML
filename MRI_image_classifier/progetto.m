%% Project Group 3: Comotti F., Covali A., di Noia C., Rossini R., Zappa C.
% In this project we read a dataset containing 3762 segmented brain MRIs,
% connected with a table of radiomic features and a diagnosis (brain tumor
% or healthy). First we try to use a Convolutional Neural Network (CNN) on
% images, but we only get 56% accuracy. Then, we operate with features with
% a Support Vector Machine algorithm (SVM) and we get accuracy around 87%

%% 1) Import data from text file

% Set up the Import Options and import the data
opts = delimitedTextImportOptions("NumVariables", 15);

% Specify range and delimiter
opts.DataLines = [2, Inf];
opts.Delimiter = ",";

% Specify column names and types
opts.VariableNames = ["Image", "Class", "Mean", "Variance", ...
    "StandardDeviation", "Entropy", "Skewness", "Kurtosis", ...
    "Contrast", "Energy", "ASM", "Homogeneity", "Dissimilarity", ....
    "Correlation", "Coarseness"];
opts.VariableTypes = ["double", "double", "double", "double", ...
    "double", "double", "double", "double", "double", "double", ...
    "double", "double", "double", "double", "double"];

% Specify file level properties
opts.ExtraColumnsRule = "ignore";
opts.EmptyLineRule = "read";

% Specify variable properties
opts = setvaropts(opts, "Image", "TrimNonNumeric", true);
opts = setvaropts(opts, "Image", "ThousandsSeparator", ",");

% Import the data
BrainTumor = readtable("Brain Tumor.csv", opts);

% Clear temporary variables
clear opts

% Removal of 'Class' and 'image number' from the dataset
BrainTumor = removevars(BrainTumor, 'Image');
Class = table2array(BrainTumor, 'Class');
Class = Class(:,1);
BrainTumor = removevars(BrainTumor, 'Class');
BrainTumor = removevars(BrainTumor, 'Coarseness'); 
% Coarseness is removed because it's out of scale and equal for every entry

% This plot shows that the 2 classes are balanced
figure()
C = categorical(Class,[0 1],{'Negative','Positive'});
histogram(C)
ylabel('Patients');
title('Class balance');

%% 2) Image Import

% To import images the first time from original files & save in a .mat file:
% (uncomment to execute)

% cd './BrainTumorOrdered';
% files = dir('*.jpg');
% for ind = 1:size(files,1)
%     pathtemp = files(ind).name;
%     jpg__temp = imread(pathtemp);
%     jpg__temp = rgb2gray(jpg__temp);
%     stackimg(ind,:,:) = jpg__temp; % Create 3d matrix with all slices
%     disp(['The dimension of stackimg after ' num2str(ind)...
%         ' iterations is ' num2str(size(stackimg))]);
% end
% save('prova.mat', 'stackimg')

% To import images from the pre-saved MATLAB structure
stackimg1 = load('-mat', 'stackimg.mat')
stackimg = stackimg1.stackimg;
clear stackimg1

%% 3) Patient division (training & testing set)

% We keep 3000 patients as training set and 762 as testing set
Ntest = 762;
n = size(stackimg,1);
Y = randsample(n, Ntest); % extr. 762 numbers from 1 to n (corresp. to test set)
original_indices_0 = 1:n;
bool_data_train = true(1,n);

for j=1:Ntest
    bool_data_train(Y(j)) = false;
end

bool_data_test =~bool_data_train; 
indices_test = original_indices_0(bool_data_test);
indices_train = original_indices_0(bool_data_train);
labels_test = Class(indices_test);
labels_train = Class(indices_train);

image_test = stackimg(indices_test,:,:);
image_train = stackimg(indices_train,:,:);

% Permuting the datasets to match the SGD function requirements
image_test_x = permute(image_test, [2 3 1]);
image_train_x = permute(image_train, [2 3 1]);

%% 4) Convolutional Network on IMAGES - TRAINING
rng(1); % Reset the random number generator

% W1 size is 101x101x20 so that convoluted images are 140x140x20
W1 = 1e-2*randn([101 101 20]); 
W5 = (2*rand(100, 98000) - 1) * sqrt(6) / sqrt(360 + 98000);
Wo = (2*rand( 2,  100) - 1) * sqrt(6) / sqrt( 2 +  100);

for epoch =1:3
    epoch
    [W1, W5, Wo] = SGD_MnistConv(W1, W5, Wo, image_train_x, labels_train);
end

save('TrainedImage.mat');
%% 5) Convolutional Network on IMAGES - TESTING
load ("TrainedImage.mat")

X   = image_test_x;
D   = labels_test;
acc = 0;
N   = length(D);
results_test = zeros(N, 1);

for k =1:N % TESTING
    k 
    x  = X(:,:,k); 
    y1  = Conv(x, W1); %convolution filter
    y2  = ReLU(y1); %ReLU
    y3  = Pool(y2); %pooling
    y4  = reshape(y3, [], 1); %reshape
    v5  = W5*y4; % from here it's like a common deep learning network
    y5  = ReLU(v5);
    v   = Wo*y5;
    y   = Softmax(v); %final result
    [~, i] = max(y);
    result(k) = i;
    
    % NB: D contains 0,1 whereas i contains 1,2
    % We associated 0-->1 e 1-->2 and therefore we compare i with D(k)+1
    if i == D(k)+1 
        acc = acc + 1;
    end
end

% Calculate and draw the confusion matrix for CNN 
ConfMatCNN = confusionmat(D, result-1);
figure() 
confusionchart(ConfMatCNN, {'Negative', 'Positive'})
title('CNN confusion matrix')

errAcc = sqrt(acc*N)/N; % Poisson statistical error on counts
acc = acc/N;
fprintf('Image CNN accuracy is %f', acc);
fprintf(' +- %f\n', errAcc);

save('TestedImage.mat');

%% 6) PCA

X = table2array(BrainTumor); % raw data matrix (before PCA)
rho = corr(X); % correlation matrix of X
[n,q] = size(X);
[U,S,V] = svd (rho); % rho eigenvalues & eigenvectors

% coeff = pca(S);
eigenv = diag(S);
N = length(eigenv);
eigenv(N+1,1)=sum(eigenv); %last number in eigenv is the sum of eigenvalues
eigenv(:,2) = eigenv(:,1)/eigenv(N+1,1) % 2nd column percentage
eigenv(1,3) = eigenv(1,2);
eigenv([2:N],3) = eigenv([2:N],2)

for i=1:q
    eigenvgraph(i)=S(i,i);
end
figure()
plot([1:q],eigenvgraph, '.b', [1:q],eigenvgraph,'-b', 'Markersize', 10);
xlabel('Eigenvalue number');
ylabel('Eigenvalue');

% We keep the first 5 features
dataset = X*V(:,1:5);
[n, numvar]= size(dataset);

% Division of the new dataset in training & testing
data_test = dataset(indices_test,:);
data_train = dataset(indices_train,:);

% We divide positive & negative patients to plot them as a function of 
% the first 3 features in the ranking obtained after PCA
bool_datapos = true(1,n);
for j=1:n
    if Class(j) == 0
        bool_datapos(j) = false;
    end
end
bool_dataneg =~bool_datapos; 
indices_dataneg = original_indices_0(bool_dataneg);
indices_datapos = original_indices_0(bool_datapos);
datapos = dataset(indices_datapos,:);
dataneg = dataset(indices_dataneg,:);

figure()
scatter3(dataneg(:,1), dataneg(:,2), dataneg(:,3))
hold on
scatter3(datapos(:,1), datapos(:,2), datapos(:,3))
xlabel('Y_1')
ylabel('Y_2')
zlabel('Y_3')
legend('Negative', 'Positive')


% Graph of the entries of each principal component
values = {'Mean';'Variance'; 'Std Dev'; 'Entropy'; ...
    'Skewness'; 'Kurtosis'; 'Contrast'; 'Energy'; 'ASM'; 'Homogeneity'; ...
    'Dissimilarity'; 'Correlation'};
figure() 
plot(V(:,1).^2, 'LineWidth', 2) %Y1
hold on
set(gca,'xtick',[1:12],'xticklabel',values)
plot(V(:,2).^2, 'LineWidth', 2) %Y2
plot(V(:,3).^2, 'LineWidth', 2) %Y3
plot(V(:,4).^2, 'LineWidth', 2) %Y4
plot(V(:,5).^2, 'LineWidth', 2) %Y5
ylabel('Relative weight')
grid on
title('Relative weight on the original features on the 5 principal components')
legend('Y_1', 'Y_2', 'Y_3', 'Y_4', 'Y_5')

figure()
importance = sqrt(V(:,1).^2 + V(:,2).^2 + V(:,3).^2 + V(:,4).^2 + V(:,5).^2)
scatter([1:12],importance, 'LineWidth', 2)
set(gca,'xtick',[1:12],'xticklabel',values)
hold on 
grid on
ylabel('Importance')
title('Original feature importance in the 5 principal components')

%% 7) SVM on features - TRAINING (kfold) & TESTING
%save('PreparedData.mat');
%load('PreparedData.mat');
kernel = 'linear';
K = 5;
NNeg=0;
NPos=0;
NNegTest=0;
NPosTest=0;
original_indices_train = 1:length(labels_train);
index = crossvalind('Kfold', length(labels_train), K); 

for ind=1:size(data_train,1)
    if labels_train(ind) == 1 % Creates the two-group labels for train data
        NPos = NPos+1;
    else
        NNeg = NNeg+1;
    end
end

for ind=1:size(data_test,1)
    if labels_test(ind) == 1 % Creates the two-group labels for test data
        NPosTest = NPosTest+1;
    else
        NNegTest = NNegTest+1;
    end
end

%K-FOLD CROSS VALIDATION (Training+Testing)
accVec = zeros(5,1);
sensVec = zeros(5,1);
specVec = zeros(5,1);
tic
for ind = 1:K
        disp(['K = ' num2str(ind)]);
        temp_train_set = true(1,size(data_train,1));
        temp_train_set(index == ind) = false;
        temp_test_set =~temp_train_set;

        temp_train_indices = original_indices_train(temp_train_set);
        temp_test_indices = original_indices_train(temp_test_set);

        temp_train_labels = labels_train(temp_train_indices);
        temp_test_labels = labels_train(temp_test_indices);

        temp_train_data = data_train(temp_train_indices,:);
        temp_test_data = data_train(temp_test_indices,:); 

        tr_data = temp_train_data;

        try % Training phase
            pc_tr_data = squeeze(tr_data(:,1:numvar));
            svmStruct = fitcsvm(pc_tr_data, temp_train_labels,...
                'KernelFunction', kernel, 'Standardize', true, ...
                'BoxConstraint', 1);
            svmStruct = compact(svmStruct);
        catch exception
            msgString = getReport(exception)
        end

        
        for subject = 1:size(temp_test_data, 1) %Testing phase
            % Testing data
            temp_tdata = temp_test_data(subject,:);
            temp_tlabel = temp_test_labels(subject);
            t_NPos = 0;
            t_NNeg = 0;
            
            for count=1:length(temp_test_labels)
                if temp_test_labels(count) == 1
                    t_NPos = t_NPos+1;
                else
                    t_NNeg = t_NNeg+1;
                end
            end

            te_data = temp_tdata;

            if size(te_data,2) == 0
                disp('No data!');
            else

               try
                    pc_te_data = squeeze(te_data(:,1:numvar));
                    class = predict(svmStruct, pc_te_data)';
                    class = cast(class,'double');
                    clear pc_te_data;

                    % Case 0 -> Negative
                    if temp_tlabel == 0 && class == 0
                        accVec(ind) = accVec(ind) + (1/length(temp_test_labels));
                        specVec(ind) = specVec(ind) + (1/t_NNeg);
                    end

                    % Case 1 -> Positive
                    if temp_tlabel == 1 && class == 1
                        accVec(ind) = accVec(ind) + (1/length(temp_test_labels));
                        sensVec(ind) = sensVec(ind) + (1/t_NPos);
                    end
               catch exception
                   disp('Error testing');
                   msgString = getReport(exception)
               end
            end
            clear te_data
        end
end
toc
accK = mean(accVec);
specK = mean(specVec);
sensK = mean(sensVec);
errAccuracyK = std(accVec);
errSpecificityK = std(specVec);
errSensitivityK = std(sensVec);

disp('.');
disp('SVM k-fold Results');
fprintf(['k-fold accuracy SVM: ' num2str(accK)]);
fprintf(' +- %f\n', errAccuracyK);
fprintf(['k-fold sensitivity SVM: ' num2str(sensK)]);
fprintf(' +- %f\n', errSensitivityK);
fprintf(['k-fold specificity SVM: ' num2str(specK)]);
fprintf(' +- %f\n', errSpecificityK);


%EXTERNAL TESTING 
temp_test_data = data_test;
temp_test_labels = labels_test;
[n, numvar] = size(data_test);
resultSVM = zeros(n,1);
accuracyTest = 0;
sensitivityTest = 0;
specificityTest = 0;
tic
for subject = 1:size(temp_test_data, 1)
            % Testing data
            temp_tdata = temp_test_data(subject,:);
            temp_tlabel = temp_test_labels(subject);
            te_data = temp_tdata;

            if size(te_data,2) == 0
                disp('No data!');
            else
               try
                    pc_te_data = squeeze(te_data(:,1:numvar));
                    class = predict(svmStruct, pc_te_data)';
                    class = cast(class,'double');
                    clear pc_te_data;
                    resultSVM(subject) = class;

                    % Case 0 -> Negative
                    if temp_tlabel == 0 && class == 0
                        accuracyTest = accuracyTest + (1/size(data_test, 1));
                        specificityTest = specificityTest + (1/NNegTest);
                    end

                    % Case 1 -> Positive
                    if temp_tlabel == 1 && class == 1
                        accuracyTest = accuracyTest + (1/size(data_test, 1));
                        sensitivityTest = sensitivityTest + (1/NPosTest);
                    end
               catch exception
                   disp('Error testing');
                   msgString = getReport(exception)
               end
            end
            clear te_data
end
toc
errAccuracyTest = sqrt(accuracyTest*n)/n;
errSensitivityTest = sqrt(sensitivityTest*NPos)/NPos;
errSpecificityTest = sqrt(specificityTest*NNeg)/NNeg;

% Calculate and draw the confusion matrix for CNN 
ConfMatSVM = confusionmat(labels_test, resultSVM);
figure() 
confusionchart(ConfMatSVM, {'Negative', 'Positive'})
title('SVM confusion matrix')

disp('.');
disp('SVM Testing Results');
fprintf(['Testing accuracy SVM: ' num2str(accuracyTest)]);
fprintf(' +- %f\n', errAccuracyTest);
fprintf(['Testing sensitivity SVM: ' num2str(sensitivityTest)]);
fprintf(' +- %f\n', errSensitivityTest);
fprintf(['Testing specificity SVM: ' num2str(specificityTest)]);
fprintf(' +- %f\n', errSpecificityTest);
disp('.');
