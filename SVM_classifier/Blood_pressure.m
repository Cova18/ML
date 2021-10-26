% Trying to predict whether the systolic pressure is higher than a
% threshold given dataset hospital, removing the systolic pressure from the
% features
clear;
clc;
load hospital
clear Description

%% Phase 1: data preparation & plotting

% ungrouping pressure columns
hospital = [hospital(:,1:5),dataset(hospital.BloodPressure(:,1),...
    'VarNames','SystolicPressure'),dataset(hospital.BloodPressure(:,2),...
    'VarNames','DiastolicPressure'),hospital(:,7:end)]; 
hospital.Properties.Units{6} = 'mm Hg';hospital.Properties.Units{7} = ...
    'mm Hg';hospital.Properties.VarDescription{6} = 'Systolic/Diastolic';
hospital.Properties.VarDescription{7} = 'Systolic/Diastolic';
Systolic = hospital(:,6);
hospital(:,{'LastName','Trials','SystolicPressure'}) = [];

% Creating vector with sex: 0 = male & 1 = female
for ind=1:size(hospital,1);
    if hospital.Sex(ind) == 'Male';
        aux(ind,1)=0;
    else
        aux(ind,1)=1;
    end
end

figure(1)
histogram(hospital.Sex)
ylabel('Counts');

figure(2)
histogram(hospital.Age)
xlabel('Age');
ylabel('Counts');

figure(3)
histogram(hospital.DiastolicPressure)
xlabel('Diastolic Pressure (mmHg)');
ylabel('Counts');

features = hospital(1,:);
hospital(:,'Sex') = []; %removing sex from hospital dataset

data = double(hospital); % convert hospital dataset to matrix
SysPressure = double(Systolic); % convert systolic dataset to vector
data = [aux, data]; % attaches sex to the matrix data
data(:,3) = data(:,3)*0.453592;
[n, NUMVAR] = size(data);
clear Systolic;
clear aux; 

threshold = round(mean(SysPressure)); % Threshold for group assignment 
NNeg=0;
NPos=0;

for ind=1:n
    if SysPressure(ind)>=threshold % Creates the two-group labels
        labels(ind,1)=1;
        NPos = NPos+1;
    else
        labels(ind,1)=0;
        NNeg = NNeg+1;
    end
end

figure(4)
C = categorical(data(:,4),[0 1],{'No','Yes'});
histogram(C)
xlabel('Smokers');
ylabel('Counts');

figure(5)
histogram(data(:,3), 20)
xlabel('Weight (kg)');
ylabel('Counts');

figure(6)
histogram(SysPressure)
xline(threshold-0.5, 'r', 'LineWidth', 3)
legend('', 'Threshold')
xlabel('Systolic pressure (mmHg)');
ylabel('Counts');

figure(7)
D = categorical(labels,[0 1],{'Under','Over'});
histogram(D)
ylabel('Counts')
xlabel('Under/over threshold (123 mmHg) classification on systolic pressure')

%% Phase 2: SVM (copy & paste)

% Training and parameter tuning
kernel = 'linear';
K = 5; % K-fold parameter
index = crossvalind('Kfold', n, K); 
original_indices = 1:n;

figure(8); 
hold on;

for numvar = 1:NUMVAR
  
    % Metrics
    accuracy = 0;
    sensitivity = 0;
    specificity = 0;

    for ind = 1:K
        disp(['K = ' num2str(ind)]);

        temp_train_set = true(1,size(data,1));
        temp_train_set(index == ind) = false;
        temp_test_set =~temp_train_set;

        temp_train_indices = original_indices(temp_train_set);
        temp_test_indices = original_indices(temp_test_set);

        temp_train_labels = labels(temp_train_indices);
        temp_test_labels = labels(temp_test_indices);

        temp_train_data = data(temp_train_indices,:);
        temp_test_data = data(temp_test_indices,:); 

        tr_data = temp_train_data;

        try

            pc_tr_data = squeeze(tr_data(:,1:numvar));
            svmStruct = fitcsvm(pc_tr_data, temp_train_labels,...
                'KernelFunction', kernel, 'Standardize', true, ...
                'BoxConstraint', 1);
            svmStruct = compact(svmStruct);

        catch exception

            % disp('Error optimization preprocessing');
            msgString = getReport(exception)

        end

        % TESTING phase
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

                    % Caso 0 -> Negative
                    if temp_tlabel == 0 && class == 0
                        accuracy = accuracy + (1/size(data, 1));
                        specificity = specificity + (1/NNeg);
                    end

                    % Caso 1 -> Positive
                    if temp_tlabel == 1 && class == 1
                        accuracy = accuracy + (1/size(data, 1));
                        sensitivity = sensitivity + (1/NPos);
                    end

               catch exception

                   disp('Error testing');
                   msgString = getReport(exception)

               end

            end

            clear te_data

        end

    end % cv

    Accuracies(numvar)=accuracy;
    scatter(numvar,accuracy); 
    drawnow; 
    hold on;
    disp(['Accuracy(' numvar ') = '  num2str(accuracy)]);
    
    %experiment
    Sensitivities(numvar)=sensitivity;
    scatter(numvar,sensitivity); 
    drawnow; 
    hold on;
    disp(['Sensitivity(' numvar ') = '  num2str(sensitivity)]);
    Specificities(numvar)=specificity;
    scatter(numvar,specificity); 
    drawnow; 
    hold on;
    disp(['Specificity(' numvar ') = '  num2str(specificity)]);
end

figure(8)
axis([0.9 5.1 0.3 1])
plot([1:NUMVAR], Accuracies, '-r', 'Linewidth', 1.5)
hold on
plot([1:NUMVAR], Sensitivities, '-g', 'Linewidth', 1.5)
plot([1:NUMVAR], Specificities, '-b', 'Linewidth', 1.5)
legend('','','','','','','','','','','','','','','', ...
    'Accuracy', 'Sensitivity', 'Specificity');
xlabel('Number of features');
ylabel('Ratio (%)')

disp('.');
disp('.');
disp('.');
disp('Results');
disp('-----');
disp(['Accuracy: ' num2str(accuracy)]);
disp(['Sensitivity: ' num2str(sensitivity)]);
disp(['Specificity: ' num2str(specificity)]);
