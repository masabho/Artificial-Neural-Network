% Age Grading An gambiae and An.arabiensis Using NIRS and ANN models.
% Author: Masabho Peter Milali
% Created: September 2018
% Modified: March 2019 
% MATLAB R2016b

%% loading cleaned spectra 
   clear;
   clc;

%% Datasets used in the original analysis
   load IFA_GA.mat;
   load IFA_GA_AgeClassBalanced.mat;
   load IFA_ARA.mat;
   load IFA_ARA_AgeClassBalanced.mat;

   %% Datasets used to test reproducibility of our study
   load cleaned_DS1.mat % first generation gambaie from Soumousso Bukinafaso
   load cleaned_DS2.mat % first generation gambaie from Kodeni Bukinafaso
   load cleaned_DS3.mat % recent gambaie colony at CSU with original larvae collected in Bukinafaso
   load cleaned_DS4.mat % gambiae colony at CSU established in 1975
   load cleaned_DS5.mat % combination of DS1 and DS2
   load cleaned_DS6.mat % combination of DS1, DS2 and DS3
   load cleaned_DS7.mat % arabiensis reared at ifakara (Ntamatungiro data)
   load cleaned_DS8.mat % first generation arabiensis from Pemba (pyrthroid data
   
  
    
   % Regression_MayagayaCleaning_Data = IFA_GA.mat;
   
    
     
     Replicate = 10;
     for i = 1:Replicate
%% Unstratified randomization
     K = randperm(size(Regression_MayagayaCleaning_Data,1));
     RandomisedSpectra_MayagayaCleaning_Data = Regression_MayagayaCleaning_Data(K(1:size(Regression_MayagayaCleaning_Data,1)),:); 
     n = size(RandomisedSpectra_MayagayaCleaning_Data,1);
     TrainSize = randsample(n,(0.7*n));
     TestSample = RandomisedSpectra_MayagayaCleaning_Data(~ismember(1:n,TrainSize),:); % Testing data
     TrainSample = RandomisedSpectra_MayagayaCleaning_Data(ismember(1:n,TrainSize),:); %Training sample
     
     %Mayagaya_Xtrain = TrainSample(:,2:end);
     %Mayagaya_Ytrain = TrainSample(:,1);
      % Mayagaya_Xtest = TestSample(:,2:end); % Query data
       % Mayagaya_Ytest = TestSample(:,1);
       
     Mayagaya_Xtrain = TrainSample(:,6:end);
     Mayagaya_Ytrain = TrainSample(:,1:5);
     Mayagaya_Xtest = TestSample(:,6:end);
     Mayagaya_Ytest = TestSample(:,1:5);
     
     netRegressionMayagayaCleaning = feedforwardnet(10);
    % netRegressionMayagayaCleaning.layers{1}.transferFcn = 'purelin';
     %netParityStatus.layers{end}.transferFcn = 'logsig'; 
     
    
          %view(netRegressionMayagayaCleaning)
    % Customize dividing of training set into training, validation and test set

    %[trainInd,valInd,testInd] = dividerand(508,0.7,0.15,0.15);
  
    % Training the net
    [netRegressionMayagayaCleaning,tr] = train(netRegressionMayagayaCleaning,Mayagaya_Xtrain',Mayagaya_Ytrain');
     
    % saving trained net
%      save netRegressionMayagayaCleaning;
   
    % Testing the trained neural network model
    
%     load netRegressionMayagayaCleaning;
    MayagayaTrainPredictedAge = netRegressionMayagayaCleaning(Mayagaya_Xtrain');% training dataset
    MayagayaTestPredictedAge = netRegressionMayagayaCleaning(Mayagaya_Xtest');%out of the sample data set
    
    %Scoring the model accuracy as a regresser
    RMSE_ANN_Mayagaya(i) = sqrt((sum((Mayagaya_Ytest - MayagayaTestPredictedAge').^2))/length(Mayagaya_Ytest));
%     ErrorsANN_Mayagaya_RMSE = sqrt((Mayagaya_Ytest - MayagayaTestPredictedAge').^2);
%     save ErrorsANN_Mayagaya_RMSE
    MAE_ANN_Mayagaya(i) = (sum(abs(Mayagaya_Ytest - MayagayaTestPredictedAge')))/length(Mayagaya_Ytest);
%     ErrorsANN_Mayagaya_MAE(i) = abs(Mayagaya_Ytest - MayagayaTestPredictedAge');
%     save ErrorsANN_Mayagaya_MAE
    MAPE_ANN_Mayagaya(i) = sum(abs((Mayagaya_Ytest - MayagayaTestPredictedAge')./Mayagaya_Ytest))*(100/length(Mayagaya_Ytest));
%     ErrorsANN_Mayagaya_MAPE(i) = abs((Mayagaya_Ytest - MayagayaTestPredictedAge')./Mayagaya_Ytest)*(100/length(Mayagaya_Ytest));
%     save ErrorsANN_Mayagaya_MAPE
    
    
    % Scoring the model accuracy as a binary classifier
  
     NumberMayagaya_Ytest_Young(i) = sum(Mayagaya_Ytest<7); 
     NumberMayagaya_Ytest_Old(i) = sum(Mayagaya_Ytest>=7);
     
     % An. gambiae (IFA_GA)
     NumberCorrectlyPredicted_Young = sum((Mayagaya_Ytest == 1|Mayagaya_Ytest == 3|Mayagaya_Ytest == 5)&(MayagayaTestPredictedAge' < 6.5));
     NumberWronglyPredicted_Young = sum((Mayagaya_Ytest==7|Mayagaya_Ytest==9|Mayagaya_Ytest==11|Mayagaya_Ytest==15|Mayagaya_Ytest==20)&(MayagayaTestPredictedAge' < 6.5));
     NumberCorrectlyPredicted_Old = sum((Mayagaya_Ytest==7|Mayagaya_Ytest==9|Mayagaya_Ytest==11|Mayagaya_Ytest==15|Mayagaya_Ytest==20)&(MayagayaTestPredictedAge' >= 6.5));
     NumberWronglyPredicted_Old = sum((Mayagaya_Ytest == 1|Mayagaya_Ytest == 3|Mayagaya_Ytest == 5)&(MayagayaTestPredictedAge' >= 6.5));
     
     % An. arabiensis (IFA_ARA)
%      NumberCorrectlyPredictedLess7DaysOld = sum((Mayagaya_Ytest == 1|Mayagaya_Ytest == 3|Mayagaya_Ytest == 5)&(MayagayaTestPredictedAge' < 6.5));
%      NumberWronglyPredictedLess7DaysOld = sum((Mayagaya_Ytest==7|Mayagaya_Ytest==9|Mayagaya_Ytest==11|Mayagaya_Ytest==15|Mayagaya_Ytest==20|Mayagaya_Ytest==25)&(MayagayaTestPredictedAge' < 6.5));
%      NumberCorrectlyPredictedGreaterOrEqual7DaysOld = sum((Mayagaya_Ytest==7|Mayagaya_Ytest==9|Mayagaya_Ytest==11|Mayagaya_Ytest==15|Mayagaya_Ytest==20|Mayagaya_Ytest==25)&(MayagayaTestPredictedAge' >= 6.5));
%      NumberWronglyPredictedGreaterOrEqual7DaysOld = sum((Mayagaya_Ytest == 1|Mayagaya_Ytest == 3|Mayagaya_Ytest == 5)&(MayagayaTestPredictedAge' >= 6.5));
%     
     
      % Testing reproducibility
%      NumberCorrectlyPredicted_Young = sum((Mayagaya_Ytest == 1|Mayagaya_Ytest == 3|Mayagaya_Ytest == 5)&(MayagayaTestPredictedAge' < 6.5));
%      NumberWronglyPredicted_Young = sum((Mayagaya_Ytest==7|Mayagaya_Ytest==9|Mayagaya_Ytest==13|Mayagaya_Ytest==16|Mayagaya_Ytest==20|Mayagaya_Ytest==25)&(MayagayaTestPredictedAge' < 6.5));
%      NumberCorrectlyPredicted_Old = sum((Mayagaya_Ytest==7|Mayagaya_Ytest==9|Mayagaya_Ytest==13|Mayagaya_Ytest==16|Mayagaya_Ytest==20|Mayagaya_Ytest==25)&(MayagayaTestPredictedAge' >= 6.5));
%      NumberWronglyPredicted_Old = sum((Mayagaya_Ytest == 1|Mayagaya_Ytest == 3|Mayagaya_Ytest == 5)&(MayagayaTestPredictedAge' >= 6.5));
%         
%     
    
     AccuracyANN_Classifier_Mayagaya = (NumberCorrectlyPredicted_Young + NumberCorrectlyPredicted_Old)/length(Mayagaya_Ytest);
     
     TrueYoung(i) = NumberCorrectlyPredicted_Young;
     WrongYoung(i) = NumberWronglyPredicted_Young;
     TrueOld(i) =  NumberCorrectlyPredicted_Old;
     WrongOld(i) = NumberWronglyPredicted_Old;
     AccuracyANN_BinaryClassifier_Mayagaya(i) = AccuracyANN_Classifier_Mayagaya;
     
    
     end 
     
      save netRegressionMayagayaCleaning_IFA_ARA; % trained ANN An. arabiensis
      load netRegressionMayagayaCleaning_IFA_ARA;
      save netRegressionMayagayaCleaning_IFA_GA; % trained ANN An. gambiae
      load netRegressionMayagayaCleaning_IFA_GA;
      
     
     Average_RMSE_ANN_Mayagaya =  mean(RMSE_ANN_Mayagaya);
     Std_dev_RMSE_ANN_Mayagaya = std(RMSE_ANN_Mayagaya)
     Average_MAPE_ANN_Mayagaya = mean(MAPE_ANN_Mayagaya);
     Std_dev_MAPE_ANN_Mayagaya = std(MAPE_ANN_Mayagaya)
     Average_MAE_ANN_Mayagaya = mean(MAE_ANN_Mayagaya);
     Std_dev_MAE_ANN_Mayagaya = std(MAE_ANN_Mayagaya);
     
     AverageNumberMayagaya_Ytest_Young = mean(NumberMayagaya_Ytest_Young);
     NumberMayagaya_Predicted_Young = TrueYoung + WrongYoung;
     AverageNumberMayagaya_Predicted_Young = mean(NumberMayagaya_Predicted_Young);
     AverageNumberMayagaya_Ytest_Old = mean(NumberMayagaya_Ytest_Old);
     NumberMayagaya_Predicted_Old = TrueOld + WrongOld;
     AverageNumberMayagaya_Predicted_Old = mean(NumberMayagaya_Predicted_Old)
     Average_AccuracyANN_BinaryClassifier_Mayagaya = mean(AccuracyANN_BinaryClassifier_Mayagaya);
     Std_dev_AccuracyANN_BinaryClassifier_Mayagaya = std(AccuracyANN_BinaryClassifier_Mayagaya);
     Average_TrueYoung_ANN = mean(TrueYoung);
     Average_WrongYoung_ANN = mean(WrongYoung);
     Average_TrueOld_ANN =  mean(TrueOld);
     Average_WrongOld_ANN = mean(WrongOld);
     
     SensitivityANN_BinaryClassifier_Mayagaya = TrueOld./NumberMayagaya_Ytest_Old
     Average_SensitivityANN_BinaryClassifier_Mayagaya = mean(SensitivityANN_BinaryClassifier_Mayagaya)
     Std_dev_SensitivityANN_BinaryClassifier_Mayagaya = std(SensitivityANN_BinaryClassifier_Mayagaya)
     SpecificityANN_BinaryClassifier_Mayagaya = TrueYoung./NumberMayagaya_Ytest_Young
     Average_SpecificityANN_BinaryClassifier_Mayagaya = mean(SpecificityANN_BinaryClassifier_Mayagaya)
     Std_dev_SpecificityANN_BinaryClassifier_Mayagaya = std (SpecificityANN_BinaryClassifier_Mayagaya)
           
           %% Testing trained model on independent test set
%            load netRegressionMayagayaCleaning;
%            load cleaned_DS4.mat
%            ITS_Absorbances = cleaned_DS3(:,3:end);
%            ITS_ActualAgeLabels = cleaned_DS3(:,2);
%            PredictedAge_ITS = netRegressionMayagayaCleaning(ITS_Absorbances');
%            RMSE_ANN_ITS = sqrt((sum((ITS_ActualAgeLabels - PredictedAge_ITS').^2))/length(ITS_ActualAgeLabels));
%            
%            Number_ITS_Young = sum(ITS_ActualAgeLabels<7); 
%            NumberMayagaya_Ytest_Old(i) = sum(ITS_ActualAgeLabels>=7);
%            
%             % Independent test set (ITS)
%            NumberCorrectlyPredicted_Young = sum((ITS_ActualAgeLabels == 3|ITS_ActualAgeLabels == 6)&(PredictedAge_ITS' < 7.5));
%            NumberWronglyPredicted_Young = sum((ITS_ActualAgeLabels==9|ITS_ActualAgeLabels==12|ITS_ActualAgeLabels==15)&(PredictedAge_ITS' < 7.5));
%            NumberCorrectlyPredicted_Old = sum((ITS_ActualAgeLabels==9|ITS_ActualAgeLabels==12|ITS_ActualAgeLabels==15)&(PredictedAge_ITS' >= 7.5));
%            NumberWronglyPredicted_Old = sum((ITS_ActualAgeLabels == 3|ITS_ActualAgeLabels == 6)&(PredictedAge_ITS' >= 7.5));
%            AccuracyANN_ITS = (NumberCorrectlyPredicted_Young + NumberCorrectlyPredicted_Old)/length( ITS_ActualAgeLabels);
%      
%            
     
    figure()
    subplot(2,2,4);
    plot(Mayagaya_Ytest, MayagayaTestPredictedAge', 'bo',Mayagaya_Ytrain+0.8, MayagayaTrainPredictedAge', 'r^')
    xlabel('Actual age (days)', 'FontSize',16);
    ylabel('Estimated age (days)', 'FontSize', 16);
    ylim([-15 15])
    xlim([0 30])
    set(gca,'XTickLabel',[{'0'}, {'5'},{'10'},{'15'},{'20'},{'25'}, {''}],'FontSize', 16);
    set(gca,'YTickLabel',[{'-5'}, {'0'},{'5'},{'10'},{'15'},{'20'}, {'25'},{'30'},{''}],'FontSize', 16);
    legend({'Test data','Training data'},'FontSize',16, ...
              'location','SE');
    legend boxoff
%     set(gcf, 'Position',[10,10,900, 1500]);
    set(gca,'fontname','arial')
    
     % Plotting residuals
    subplot(2,2,4);
    set(gcf, 'Position',[10,10,850, 2000]);
    Errors_Mayagaya_Ytest_ANN = Mayagaya_Ytest - MayagayaTestPredictedAge';
    Errors_Mayagaya_Ytrain_ANN = Mayagaya_Ytrain - MayagayaTrainPredictedAge';
    plot(Mayagaya_Ytest, Errors_Mayagaya_Ytest_ANN, 'bo', Mayagaya_Ytrain+0.8,Errors_Mayagaya_Ytrain_ANN, 'r^')   
    xlabel('Actual age in days', 'FontSize',14);
    ylabel('Residuals in days', 'FontSize', 14);
    xlim([0 27])
    ylim([-15 15])
    set(gca,'XTickLabel',[[{'0'}, {'5'},{'10'},{'15'},{'20'}, {'25'}]],'FontSize', 14);
    %set(gca,'YTicklabel',[{'-0.5'},{'0'},{'0.5'}, {'1'},{'1.5'},{''}], 'FontSize',16);
    legend({'Test data','Training data'},'FontSize',14, ...
              'location','NW');
    legend boxoff
    set(gca,'fontname','arial')
    hold on
    
      % Box plot
    figure()
    subplot(2,2,4);
    h = boxplot(MayagayaTestPredictedAge',Mayagaya_Ytest, 'Colors','k');
    set(h,{'linew'},{1})
    xlabel('Actual age in days', 'FontSize',14);
    ylabel('Estimated age in days', 'FontSize', 14);
    set(gca,'XTickLabel',[{'1'}, {'3'},{'5'},{'7'},{'9'}, {'11'}, {'15'}, {'20'}, {'25'}],'FontSize', 14);
    %set(gca,'YTickLabel',[{'-5'}, {'0'},{'5'},{'10'},{'15'}, {'20'}, {'25'}, {''}],'FontSize', 16);
    set(gcf, 'Position',[10,10,850, 2000]);
    set(gca,'fontname','arial');
     
     
     subplot(2,2,4);
     xdata_MayagayaANN = [1 2];
     ydata_MayagayaANN = [Average_TrueLessSeven_ANN/AverageNumberMayagaya_PredictedLessThan_7d*100, Average_WrongLessSeven_ANN/AverageNumberMayagaya_PredictedLessThan_7d*100;  Average_TrueGreaterSeven_ANN/AverageNumberMayagaya_PredictedGreaterOrEqual_7d*100, Average_WrongGreaterSeven_ANN/AverageNumberMayagaya_PredictedGreaterOrEqual_7d*100];
     e = [std(TrueLessSeven./NumberMayagaya_PredictedLessThan_7d)*100, std(WrongLessSeven./NumberMayagaya_PredictedLessThan_7d)*100; std(TrueGreaterSeven./NumberMayagaya_PredictedGreaterOrEqual_7d)*100, std(WrongGreaterSeven./NumberMayagaya_PredictedGreaterOrEqual_7d)*100]
     hb = bar(xdata_MayagayaANN,ydata_MayagayaANN,1);
     hold on
     xe = [0.85 1.15; 1.85 2.15];
     errorbar(xe,ydata_MayagayaANN,e,'*','CapSize',18,'LineWidth',2)
    
     set(hb(2),'facecolor',[120/256 198/256 83/256])% got color value using color picker
     %set(hb(2),'facecolor',[244/256 134/256 66/256])
     ylabel('% Mosquitoes','FontSize', 16);
     xlabel('Model prediction','FontSize', 16);
     set(gca,'XTick',[1 2]);
     set(gca,'XTickLabel',[{'< 7 days (N = 139)'}, {'? 7 days (N = 223 )'}],'FontSize', 16);
     ylim([0 140])
     set(gca,'YTickLabel',[{'0'}, {'20'},{'40'},{'60'}, {'80'},{'100'},{''},{''}],'FontSize', 16);
     legend({'Correct prediction','False prediction'},'FontSize',16, ...
              'location','NW');
     legend boxoff
     set(gcf, 'Position',[10,10,900, 1500]);
     set(gca,'fontname','arial')
     
     hold on
               
     
      Replicate = 10;
     for i = 1:Replicate
 %% randomization
     K = randperm(size(Regression_MayagayaCleaning_Data,1));
     RandomisedSpectra_MayagayaCleaning_Data = Regression_MayagayaCleaning_Data(K(1:size(Regression_MayagayaCleaning_Data,1)),:); 
     n = size(RandomisedSpectra_MayagayaCleaning_Data,1);
     TrainSize = randsample(n,(0.7*n));
     TestSample = RandomisedSpectra_MayagayaCleaning_Data(~ismember(1:n,TrainSize),:); % Testing data
     TrainSample = RandomisedSpectra_MayagayaCleaning_Data(ismember(1:n,TrainSize),:); %Training sample
     Mayagaya_Xtrain_PLS = TrainSample(:,2:end);
     Mayagaya_Ytrain_PLS = TrainSample(:,1);
     Mayagaya_Xtest_PLS = TestSample(:,2:end); % Query data
     Mayagaya_Ytest_PLS = TestSample(:,1);


   % PLS regression model on spectra according to Mayagaya
   [Mayagaya_XtrainL, Mayagaya_Ytrainl, Mayagaya_XtrainS, Mayagaya_YtrainS, Mayagaya_beta_PLS] = plsregress(Mayagaya_Xtrain_PLS, Mayagaya_Ytrain_PLS, 10, 'CV', 10);
    A = [ones(size(Mayagaya_Xtest_PLS,1), 1), Mayagaya_Xtest_PLS];
    Mayagaya_PredictedAge_PLS = A*Mayagaya_beta_PLS;%Predicting age of test sample (out of sample testing)
    B = [ones(size(Mayagaya_Xtrain_PLS,1),1), Mayagaya_Xtrain_PLS];
    Mayagaya_trainPredicted_PLS = B*Mayagaya_beta_PLS;%Predicting age of training sample
    
    
    %Scoring the model accuracy as a regresser
    RMSE_PLS_Mayagaya(i) = sqrt((sum((Mayagaya_Ytest_PLS-Mayagaya_PredictedAge_PLS).^2))/length(Mayagaya_Ytest_PLS));
%     ErrorsPLS_Mayagaya_RMSE = sqrt((Mayagaya_Ytest-Mayagaya_PredictedAge).^2);
%     save ErrorsPLS_Mayagaya_RMSE
    MAE_PLS_Mayagaya(i) = (sum(abs(Mayagaya_Ytest_PLS-Mayagaya_PredictedAge_PLS)))/length(Mayagaya_Ytest_PLS);
%     ErrorsPLS_Mayagaya_MAE(i) = abs(Mayagaya_Ytest-Mayagaya_PredictedAge);
%     save ErrorsPLS_Mayagaya_MAE
    MAPE_PLS_Mayagaya(i) = sum(abs((Mayagaya_Ytest_PLS-Mayagaya_PredictedAge_PLS)./Mayagaya_Ytest_PLS))*(100/length(Mayagaya_Ytest_PLS));
%     ErrorsPLS_Mayagaya_MAPE(i) = abs((Mayagaya_Ytest-Mayagaya_PredictedAge)./Mayagaya_Ytest)*(100/length(Mayagaya_Ytest));
%     save ErrorsPLS_Mayagaya_MAPE
    
    
    %Scoring the model accuracy as a binary classifier
    
     NumberMayagaya_Ytest_Young_PLS(i) = sum(Mayagaya_Ytest_PLS<7);
     NumberMayagaya_Ytest_Old_PLS(i) = sum(Mayagaya_Ytest_PLS>=7);
     
     % An. gambiae (IFA_GA)
     NumberCorrectlyPredicted_Young_PLS = sum((Mayagaya_Ytest_PLS == 1|Mayagaya_Ytest_PLS == 3|Mayagaya_Ytest_PLS == 5)&( Mayagaya_PredictedAge_PLS < 6.5));
     NumberWronglyPredicted_Young_PLS = sum((Mayagaya_Ytest_PLS ==7|Mayagaya_Ytest_PLS==9|Mayagaya_Ytest_PLS==11|Mayagaya_Ytest_PLS==15|Mayagaya_Ytest_PLS==20)&(Mayagaya_PredictedAge_PLS < 6.5));
     NumberCorrectlyPredicted_Old_PLS = sum((Mayagaya_Ytest_PLS==7|Mayagaya_Ytest_PLS==9|Mayagaya_Ytest_PLS==11|Mayagaya_Ytest_PLS==15|Mayagaya_Ytest_PLS==20)&(Mayagaya_PredictedAge_PLS >= 6.5));
     NumberWronglyPredicted_Old_PLS = sum((Mayagaya_Ytest_PLS == 1|Mayagaya_Ytest_PLS == 3|Mayagaya_Ytest_PLS == 5)&(Mayagaya_PredictedAge_PLS >= 6.5));
     
     % An. arabiensis (IFA_ARA)
%      NumberCorrectlyPredictedLess7DaysOld_PLS = sum((Mayagaya_Ytest_PLS == 1|Mayagaya_Ytest_PLS == 3|Mayagaya_Ytest_PLS == 5)&(Mayagaya_PredictedAge_PLS < 6.5));
%      NumberWronglyPredictedLess7DaysOld_PLS = sum((Mayagaya_Ytest_PLS==7|Mayagaya_Ytest_PLS==9|Mayagaya_Ytest_PLS==11|Mayagaya_Ytest_PLS==15|Mayagaya_Ytest_PLS==20|Mayagaya_Ytest_PLS==25)&(Mayagaya_PredictedAge_PLS < 6.5));
%      NumberCorrectlyPredictedGreaterOrEqual7DaysOld_PLS = sum((Mayagaya_Ytest_PLS==7|Mayagaya_Ytest_PLS==9|Mayagaya_Ytest_PLS==11|Mayagaya_Ytest_PLS==15|Mayagaya_Ytest_PLS==20|Mayagaya_Ytest_PLS==25)&(Mayagaya_PredictedAge_PLS >= 6.5));
%      NumberWronglyPredictedGreaterOrEqual7DaysOld_PLS = sum((Mayagaya_Ytest_PLS == 1|Mayagaya_Ytest_PLS == 3|Mayagaya_Ytest_PLS == 5)&(Mayagaya_PredictedAge_PLS >= 6.5));
%     
     
     % Testing Reproducibility
%      NumberCorrectlyPredicted_Young_PLS = sum((Mayagaya_Ytest_PLS == 1|Mayagaya_Ytest_PLS == 3|Mayagaya_Ytest_PLS == 5)&(Mayagaya_PredictedAge_PLS < 8.5));
%      NumberWronglyPredicted_Young_PLS = sum((Mayagaya_Ytest_PLS==7|Mayagaya_Ytest_PLS==9|Mayagaya_Ytest_PLS==13|Mayagaya_Ytest_PLS==16|Mayagaya_Ytest_PLS==20|Mayagaya_Ytest_PLS==25)&(Mayagaya_PredictedAge_PLS < 8.5));
%      NumberCorrectlyPredicted_Old_PLS = sum((Mayagaya_Ytest_PLS==7|Mayagaya_Ytest_PLS==9|Mayagaya_Ytest_PLS==13|Mayagaya_Ytest_PLS==16|Mayagaya_Ytest_PLS==20|Mayagaya_Ytest_PLS==25)&(Mayagaya_PredictedAge_PLS >= 8.5));
%      NumberWronglyPredicted_Old_PLS = sum((Mayagaya_Ytest_PLS == 1|Mayagaya_Ytest_PLS == 3|Mayagaya_Ytest_PLS==5)&(Mayagaya_PredictedAge_PLS >= 8.5));
%      


    
     AccuracyPLS_Classifier_Mayagaya = (NumberCorrectlyPredicted_Young_PLS + NumberCorrectlyPredicted_Old_PLS)/length(Mayagaya_Ytest_PLS);
     
     TrueYoung_PLS(i) = NumberCorrectlyPredicted_Young_PLS;
     WrongYoung_PLS(i) = NumberWronglyPredicted_Young_PLS;
     TrueOld_PLS(i) =  NumberCorrectlyPredicted_Old_PLS;
     WrongOld_PLS(i) = NumberWronglyPredicted_Old_PLS;
     AccuracyPLS_BinaryClassifier_Mayagaya(i) = AccuracyPLS_Classifier_Mayagaya;
    
     end 
       
     
     Average_RMSE_PLS_Mayagaya =  mean(RMSE_PLS_Mayagaya);
     Std_dev_RMSE_PLS_Mayagaya = std(RMSE_PLS_Mayagaya);
     Average_MAPE_PLS_Mayagaya = mean(MAPE_PLS_Mayagaya);
     Std_dev_MAPE_PLS_Mayagaya = std(MAPE_PLS_Mayagaya)
     Average_MAE_PLS_Mayagaya = mean(MAE_PLS_Mayagaya);
     Std_dev_MAE_PLS_Mayagaya = std(MAE_PLS_Mayagaya);
     
     AverageNumberMayagaya_Ytest_Young_PLS = mean(NumberMayagaya_Ytest_Young_PLS);
     NumberMayagaya_Predicted_Young_PLS = TrueYoung_PLS + WrongYoung_PLS;
     AverageNumberMayagaya_Predicted_Young_PLS = mean(NumberMayagaya_Predicted_Young_PLS)
     AverageNumberMayagaya_Ytest_Old_PLS = mean(NumberMayagaya_Ytest_Old_PLS);
     NumberMayagaya_Predicted_Old_PLS = TrueOld_PLS + WrongOld_PLS;
     AverageNumberMayagaya_Predicted_Old_PLS = mean(NumberMayagaya_Predicted_Old_PLS)
     Average_AccuracyPLS_BinaryClassifier_Mayagaya = mean(AccuracyPLS_BinaryClassifier_Mayagaya);
     Std_dev_AccuracyPLS_BinaryClassifier_Mayagaya = std(AccuracyPLS_BinaryClassifier_Mayagaya);
     Average_TrueYoung_PLS = mean(TrueYoung_PLS);
     Average_WrongYoung_PLS = mean(WrongYoung_PLS);
     Average_TrueOld_PLS =  mean(TrueOld_PLS);
     Average_WrongOld_PLS = mean(WrongOld_PLS);
     
     SensitivityPLS_BinaryClassifier_Mayagaya = TrueOld_PLS./NumberMayagaya_Ytest_Old_PLS
     Average_SensitivityPLS_BinaryClassifier_Mayagaya = mean(SensitivityPLS_BinaryClassifier_Mayagaya)
     Std_dev_SensitivityPLS_BinaryClassifier_Mayagaya = std(SensitivityPLS_BinaryClassifier_Mayagaya)
     SpecificityPLS_BinaryClassifier_Mayagaya = TrueYoung_PLS./NumberMayagaya_Ytest_Young_PLS
     Average_SpecificityPLS_BinaryClassifier_Mayagaya = mean(SpecificityPLS_BinaryClassifier_Mayagaya)
     Std_dev_SpecificityPLS_BinaryClassifier_Mayagaya = std (SpecificityPLS_BinaryClassifier_Mayagaya)
     
     [h,p_RMSE_PLS_ANN_TwoTail,ci,stats] = ttest(RMSE_PLS_Mayagaya,RMSE_ANN_Mayagaya); % two tail t-test
     [h,p_RMSE_PLS_ANN_OneTail,ci,stats] = ttest(RMSE_PLS_Mayagaya,RMSE_ANN_Mayagaya,'Tail','left'); % One tail t-test
     [h,p_MAPE_PLS_ANN_TwoTail,ci,stats] = ttest(MAPE_PLS_Mayagaya,MAPE_ANN_Mayagaya); % two tail t-test
     [h,p_MAPE_PLS_ANN_OneTail,ci,stats] = ttest(MAPE_PLS_Mayagaya,MAPE_ANN_Mayagaya, 'Tail','left'); % One tail t-test
     [h,p_MAE_PLS_ANN_TwoTail,ci,stats] = ttest(MAE_PLS_Mayagaya,MAE_ANN_Mayagaya); % two tail t-test
     [h,p_MAE_PLS_ANN_OneTail,ci,stats] = ttest(MAE_PLS_Mayagaya,MAE_ANN_Mayagaya, 'Tail','left'); % One tail t-test
     [h,p_Accuracy_PLS_ANN_TwoTail,ci,stats] =ttest(AccuracyPLS_BinaryClassifier_Mayagaya,AccuracyANN_BinaryClassifier_Mayagaya);%two tail t-test
     [h,p_Accuracy_PLS_ANN_OneTail,ci,stats] =ttest(AccuracyPLS_BinaryClassifier_Mayagaya,AccuracyANN_BinaryClassifier_Mayagaya, 'Tail','left'); % one tail
     [h,p_Sensitivity_PLS_ANN_TwoTail,ci,stats] =ttest(SensitivityPLS_BinaryClassifier_Mayagaya,SensitivityANN_BinaryClassifier_Mayagaya);%two tail
     [h,p_Sensitivity_PLS_ANN_OneTail,ci,stats] =ttest(SensitivityPLS_BinaryClassifier_Mayagaya,SensitivityANN_BinaryClassifier_Mayagaya, 'Tail','left'); % one tail
     [h,p_Specificity_PLS_ANN_TwoTail,ci,stats] =ttest(SpecificityPLS_BinaryClassifier_Mayagaya,SpecificityANN_BinaryClassifier_Mayagaya);%two tail 
     [h,p_Specificity_PLS_ANN_OneTail,ci,stats] =ttest(SpecificityPLS_BinaryClassifier_Mayagaya,SpecificityANN_BinaryClassifier_Mayagaya, 'Tail','left');% one tail 
              
           %testing independent Testset
%            load cleaned_DS4.mat
%            ITS_Absorbances = cleaned_DS4(:,3:end);
%            ITS_ActualAgeLabels = cleaned_DS4(:,2);
%            A = [ones(size(ITS_Absorbances,1), 1), ITS_Absorbances];
%            PredictedAge_ITS_PLS = A*Mayagaya_beta_PLS;
%            RMSE_PLS_ITS = sqrt((sum((ITS_ActualAgeLabels - PredictedAge_ITS_PLS).^2))/length(ITS_ActualAgeLabels));
%            
           
%            Number_ITS_Young = sum(ITS_ActualAgeLabels<7); 
%            NumberMayagaya_Ytest_Old(i) = sum(ITS_ActualAgeLabels>=7);
%            
%             % Independent test set (ITS)
%            NumberCorrectlyPredicted_Young = sum((ITS_ActualAgeLabels == 3|ITS_ActualAgeLabels == 6)&(PredictedAge_ITS' < 7.5));
%            NumberWronglyPredicted_Young = sum((ITS_ActualAgeLabels==9|ITS_ActualAgeLabels==12|ITS_ActualAgeLabels==15)&(PredictedAge_ITS' < 7.5));
%            NumberCorrectlyPredicted_Old = sum((ITS_ActualAgeLabels==9|ITS_ActualAgeLabels==12|ITS_ActualAgeLabels==15)&(PredictedAge_ITS' >= 7.5));
%            NumberWronglyPredicted_Old = sum((ITS_ActualAgeLabels == 3|ITS_ActualAgeLabels == 6)&(PredictedAge_ITS' >= 7.5));
%            AccuracyANN_ITS = (NumberCorrectlyPredicted_Young + NumberCorrectlyPredicted_Old)/length( ITS_ActualAgeLabels);
%      
     
     
    
    hold on
    figure(2)
    subplot(2,2,3);
    plot(Mayagaya_Ytest_PLS, Mayagaya_PredictedAge_PLS, 'bo', Mayagaya_Ytrain_PLS+0.8, Mayagaya_trainPredicted_PLS, 'r^')
    xlabel('Actual age (days)', 'FontSize',16);
    ylabel('Estimated age (days)', 'FontSize', 16);
    ylim([-7 35])
    xlim([0 30])
    set(gca,'XTickLabel',[{'0'}, {'5'},{'10'},{'15'}, {'20'},{'25'},{''}],'FontSize', 16);
    set(gca,'YTickLabel',[{'-5'}, {'0'},{'5'},{'10'},{'15'}, {'20'}, {'25'}, {'30'},{''}],'FontSize', 16);
    legend({'Test data','Training data'},'FontSize',16, ...
              'location','SE');
    legend boxoff
%     ylim([-5 27])
    set(gcf, 'Position',[10,10,850, 2000]);
    set(gca,'fontname','arial')
    
     % Plotting residuals
    subplot(2,2,3);
    set(gcf, 'Position',[10,10,850, 2000]);
    Errors_Mayagaya_Ytest_PLS = Mayagaya_Ytest_PLS - Mayagaya_PredictedAge_PLS;
    Errors_Mayagaya_Ytrain_PLS = Mayagaya_Ytrain_PLS - Mayagaya_trainPredicted_PLS;
    plot(Mayagaya_Ytest_PLS, Errors_Mayagaya_Ytest_PLS, 'bo', Mayagaya_Ytrain_PLS+0.8,Errors_Mayagaya_Ytrain_PLS, 'r^')   
    xlabel('Actual age in days', 'FontSize',16);
    ylabel('Residuals in days', 'FontSize', 16);
    xlim([0 27])
    ylim([-17 17])
    set(gca,'XTickLabel',[[{'0'}, {'5'},{'10'},{'15'},{'20'}, {'25'}]],'FontSize', 14);
    %set(gca,'YTicklabel',[{'-0.5'},{'0'},{'0.5'}, {'1'},{'1.5'},{''}], 'FontSize',16);
    legend({'Test data','Training data'},'FontSize',14, ...
              'location','NW');
    legend boxoff
    set(gca,'fontname','arial')
    
      % Box plot
    figure()
    subplot(2,2,3);
    h = boxplot(Mayagaya_PredictedAge_PLS, Mayagaya_Ytest_PLS, 'Colors','k');
    set(h,{'linew'},{1})
    xlabel('Actual age in days', 'FontSize',14);
    ylabel('Estimated age in days', 'FontSize', 14);
    set(gca,'XTickLabel',[{'1'}, {'3'},{'5'},{'7'},{'9'}, {'11'}, {'15'}, {'20'}, {'25'}],'FontSize', 14);
    %set(gca,'YTickLabel',[{'-5'}, {'0'},{'5'},{'10'},{'15'}, {'20'}, {'25'}, {''}],'FontSize', 16);
    set(gcf, 'Position',[10,10,860, 2000]);
    set(gca,'fontname','arial');
    hold on 
     
     subplot(2,2,3);
     xdata_MayagayaPLS = [1 2];
     ydata_MayagayaPLS = [Average_TrueLessSeven_PLS/AverageNumberMayagaya_PredictedLessThan_7d_PLS*100 Average_WrongLessSeven_PLS/AverageNumberMayagaya_PredictedLessThan_7d_PLS*100;  Average_TrueGreaterSeven_PLS/AverageNumberMayagaya_PredictedGreaterOrEqual_7d_PLS*100 Average_WrongGreaterSeven_PLS/AverageNumberMayagaya_PredictedGreaterOrEqual_7d_PLS*100];
     E = [std(TrueLessSeven_PLS./NumberMayagaya_PredictedLessThan_7d_PLS)*100, std(WrongLessSeven_PLS./NumberMayagaya_PredictedLessThan_7d_PLS)*100; std(TrueGreaterSeven_PLS./NumberMayagaya_PredictedGreaterOrEqual_7d_PLS)*100, std(WrongGreaterSeven_PLS./NumberMayagaya_PredictedGreaterOrEqual_7d_PLS)*100]
     hb = bar(xdata_MayagayaPLS,ydata_MayagayaPLS,1);
     hold on
     xe = [0.85 1.15; 1.85 2.15];
     errorbar(xe,ydata_MayagayaPLS,E,'*','CapSize',18,'LineWidth',2)
      set(hb(2),'facecolor',[120/256 198/256 83/256])%got color value using color picker
%      set(hb(2),'facecolor',[244/256 134/256 66/256])
     ylabel('% Mosquitoes','FontSize', 14);
     xlabel('Model prediction','FontSize', 14);
     set(gca,'XTick',[1 2]);
     set(gca,'XTickLabel',[{'< 7 days (N = 139)'}, {'? 7 days (N = 223)'}],'FontSize', 14);
     ylim([0 140])
     set(gca,'YTickLabel',[{'0'}, {'20'},{'40'},{'60'}, {'80'},{'100'},{''},{''}],'FontSize', 14);
     
     legend({'Correct prediction','False prediction'},'FontSize',14, ...
              'location','NW');
     legend boxoff
     set(gca,'fontname','arial')
     
     hold off
%      print(gcf,'Fig2.tif','-dtiff','-r300');

     
        %% Direct training of binary classifiers  
     
     % Categorizing Mosquitoes into age classes (<7 or >=7) and assigning
     % zero label for those in age class < 7 and one for those in age class
     % >= 7
%      %% Anopheles
      Class_Mosquitoes = Regression_MayagayaCleaning_Data;
%      SortedMosquitoes = sortrows(Class_Mosquitoes, 1); % sorted according to age labels
%      NumberLessSeven = size((SortedMosquitoes(SortedMosquitoes ==3 |SortedMosquitoes == 5)),1) % size of Mosquitoes less seven days
% %      SizeGreaterOrEqualSeven = size((SortedMosquitoes(SortedMosquitoes==8|SortedMosquitoes==11)),1)
%      MosqLessSeven = SortedMosquitoes(1: NumberLessSeven,:);
%      MosqLessSeven_New = [zeros(size((MosqLessSeven),1),1),MosqLessSeven];
%      MosqGreaterOrEqualSeven = SortedMosquitoes((NumberLessSeven + 1): size((SortedMosquitoes),1),:);
%      MosqGreaterOrEqualSeven_New = [ones(size((MosqGreaterOrEqualSeven),1),1),MosqGreaterOrEqualSeven];
%      
%      Unbalanced_AgeClass_SevenCutOff = [MosqLessSeven_New ; MosqGreaterOrEqualSeven_New];
%      AgeCategorized_Mayagaya = Unbalanced_AgeClass_SevenCutOff;
     %% Age classes balanced
%        load Balanced_AgeClass_SevenCutOff_Ara.mat;
%        AgeCategorized_Mayagaya = Balanced_AgeClass_SevenCutOff_Ara;
    
    AgeCategorized_Mayagaya = Cleaned_Alex_data_Ara_Ifa;
%     Absorbances = AgeCategorized_Mayagaya(:,3:end);
%     Absorbances = Absorbances';
%     AgeLabels = AgeCategorized_Mayagaya(:,1);

     %% Model training 
     
     C_Replicate = 5; % setting number of replicates
   for iBinary = 1:C_Replicate
     K_Class = randperm(size(AgeCategorized_Mayagaya,1));
     Randomised_AgeCategorized_Mayagaya = AgeCategorized_Mayagaya(K_Class(1:size(AgeCategorized_Mayagaya,1)),:);
     n = size(Randomised_AgeCategorized_Mayagaya,1);
     TrainSize = randsample(n,(0.7*n));
     TestSample = Randomised_AgeCategorized_Mayagaya(~ismember(1:n,TrainSize),:); % Testing data
     TrainSample = Randomised_AgeCategorized_Mayagaya(ismember(1:n,TrainSize),:); %Training sample
     Binary_Xtrain_Mayagaya_ANN = TrainSample(:,3:end);
     Binary_Ytrain_Mayagaya_ANN = TrainSample(:,1);
     Binary_Xtest_Mayagaya_ANN = TestSample(:,3:end); % Query data
     Binary_Ytest_Mayagaya_ANN = TestSample(:,1);
       
     
     netClassificationMayagayaCleaning = feedforwardnet(10);
%      netClassificationMayagayaCleaning = feedforwardnet([10 10]);
%      netClassificationMayagayaCleaning.layers{end}.transferFcn = 'logsig';
%      netClassificationMayagayaCleaning.layers{2}.transferFcn = 'logsig';
%      netClassificationMayagayaCleaning.layers{3}.transferFcn = 'logsig';
     %netClassificationMayagayaCleaning.performFcn = 'crossentropy';
    % Customize dividing of training set into training, validation and test set

    % [trainInd,valInd,testInd] = dividerand(508,0.7,0.15,0.15);
  
    % Training the net
    [netClassificationMayagayaCleaning,tr] = train(netClassificationMayagayaCleaning,Binary_Xtrain_Mayagaya_ANN',Binary_Ytrain_Mayagaya_ANN');
    % saving trained net
%     save netRegressionMayagayaCleaning;
   
    % Testing the trained neural network model
%     load netRegressionMayagayaCleaning;
     MayagayaTrainPredictedAge_ClassANN = netClassificationMayagayaCleaning(Binary_Xtrain_Mayagaya_ANN');% testing training dataset
     MayagayaTestPredictedAge_ClassANN = netClassificationMayagayaCleaning(Binary_Xtest_Mayagaya_ANN');% testing on out of the sample data set
           
     
     %Scoring the model accuracy
     NumberBinaryMayagaya_Ytest_Young(iBinary) = sum(Binary_Ytest_Mayagaya_ANN == 0);
     NumberBinaryMayagaya_Ytest_Old(iBinary) = sum(Binary_Ytest_Mayagaya_ANN == 1);
     
     NumberCorrectlyPredictedYoung_BinaryMyg_ANN = sum((Binary_Ytest_Mayagaya_ANN == 0)&(MayagayaTestPredictedAge_ClassANN' < 0.5));
     NumberWronglyPredictedYoung_BinaryMyg_ANN = sum((Binary_Ytest_Mayagaya_ANN == 1)&(MayagayaTestPredictedAge_ClassANN' < 0.5));
     NumberCorrectlyPredictedOld_Binary_ANN = sum((Binary_Ytest_Mayagaya_ANN == 1)&(MayagayaTestPredictedAge_ClassANN' >= 0.5));
     NumberWronglyPredictedOld_Binary_ANN = sum((Binary_Ytest_Mayagaya_ANN == 0)&(MayagayaTestPredictedAge_ClassANN' >= 0.5));
     AccuracyANN_BinaryMyg = ((NumberCorrectlyPredictedYoung_BinaryMyg_ANN + NumberCorrectlyPredictedOld_Binary_ANN)/length(Binary_Ytest_Mayagaya_ANN))*100;
     
     TrueYoung_BinaryMygANN(iBinary) = NumberCorrectlyPredictedYoung_BinaryMyg_ANN;
     WrongYoung_BinaryMygANN(iBinary) = NumberWronglyPredictedYoung_BinaryMyg_ANN;
     TrueOld_BinaryMygANN(iBinary) =  NumberCorrectlyPredictedOld_Binary_ANN;
     WrongOld_BinaryMygANN(iBinary) = NumberWronglyPredictedOld_Binary_ANN;
     AccuracyANN_BinaryClassifier_BinaryMyg(iBinary) = AccuracyANN_BinaryMyg;
        
   end  
      save netClassificationMayagayaCleaning_Arabiensis;
      load netClassificationMayagayaCleaning_Arabiensis;
      save netClassificationMayagayaCleaning_DS1
      save netClassificationMayagayaCleaning_DS2
   
     AverageNumberBinaryMayagaya_Ytest_Young_ANN = mean(NumberBinaryMayagaya_Ytest_Young);
     NumberBinaryMayagaya_Predicted_Young_ANN = TrueYoung_BinaryMygANN + WrongYoung_BinaryMygANN;
     AverageNumberBinaryMayagaya_Predicted_Young_ANN = mean(NumberBinaryMayagaya_Predicted_Young_ANN);
     AverageNumberBinaryMayagaya_Ytest_Old_ANN = mean(NumberBinaryMayagaya_Ytest_Old);
     NumberBinaryMayagaya_Predicted_Old_ANN = TrueOld_BinaryMygANN + WrongOld_BinaryMygANN;
     AverageNumberBinary_Predicted_Old_ANN = mean(NumberBinaryMayagaya_Predicted_Old_ANN);
     Average_AccuracyANN_BinaryClassifier_BinaryMyg = mean(AccuracyANN_BinaryClassifier_BinaryMyg);
     Std_dev_AccuracyANN_BinaryClassifier_BinaryMyg = std(AccuracyANN_BinaryClassifier_BinaryMyg);
     Average_TrueYoung_BinaryMyg_ANN = mean(TrueYoung_BinaryMygANN);
     Average_WrongYoung_BinaryMyg_ANN = mean(WrongYoung_BinaryMygANN);
     Average_TrueOld_BinaryMyg_ANN =  mean(TrueOld_BinaryMygANN);
     Average_WrongOld_BinaryMyg_ANN = mean(WrongOld_BinaryMygANN);
     
     SensitivityANN_BinaryClassifier_BinaryMyg = TrueOld_BinaryMygANN./NumberBinaryMayagaya_Ytest_Old
     Average_SensitivityANN_BinaryClassifier_BinaryMyg = mean(SensitivityANN_BinaryClassifier_BinaryMyg)
     Std_dev_SensitivityANN_BinaryClassifier_BinaryMyg = std(SensitivityANN_BinaryClassifier_BinaryMyg)
     SpecificityANN_BinaryClassifier_BinaryMyg = TrueYoung_BinaryMygANN./NumberBinaryMayagaya_Ytest_Young
     Average_SpecificityANN_BinaryClassifier_BinaryMyg = mean(SpecificityANN_BinaryClassifier_BinaryMyg)
     Std_dev_SpecificityANN_BinaryClassifier_BinaryMyg = std (SpecificityANN_BinaryClassifier_BinaryMyg)
     
                 %% Testing trained binary classifier model on independent test set
%            load netClassificationMayagayaCleaning;
%            load cleaned_DS7.mat
%            ITS_Absorbances = Cleaned_Alex_data_Ara_Ifa(:,3:end);
%            ITS_AgeClassLabels = Cleaned_Alex_data_Ara_Ifa(:,1);
%            PredictedAgeClass_ITS = netClassificationMayagayaCleaning_Arabiensis(ITS_Absorbances');

%            
%            Number_ITS_Young = sum(ITS_AgeClassLabels==0); 
%            Number_ITS_Old = sum(ITS_AgeClassLabels==1);
%            
%             % Independent test set (ITS)
%            NumberCorrectlyPredicted_Young = sum((ITS_AgeClassLabels == 0)&(PredictedAgeClass_ITS' < 0.5));
%            NumberWronglyPredicted_Young = sum((ITS_AgeClassLabels==1)&(PredictedAgeClass_ITS' < 0.5));
%            NumberCorrectlyPredicted_Old = sum((ITS_AgeClassLabels==1)&(PredictedAgeClass_ITS' >= 0.5));
%            NumberWronglyPredicted_Old = sum((ITS_AgeClassLabels == 0)&(PredictedAgeClass_ITS' >= 0.5));
%            AccuracyANN_ITS = (NumberCorrectlyPredicted_Young + NumberCorrectlyPredicted_Old)/length( ITS_AgeClassLabels);

%            Sensitivity_ITS =  NumberCorrectlyPredicted_Old/Number_ITS_Old
%            Specificity_ITS = NumberCorrectlyPredicted_Young/Number_ITS_Young
%            
     
    figure() 
    subplot(2,2,4);
    plot(Binary_Ytest_Mayagaya_ANN,  MayagayaTestPredictedAge_ClassANN', 'bo', Binary_Ytrain_Mayagaya_ANN+0.05, MayagayaTrainPredictedAge_ClassANN', 'r^')
    xlabel('Actual age (days)', 'FontSize',16);
    ylabel('Estimated age (days)', 'FontSize', 16);
    xlim([-0.1 1.2])
    ylim([-0.5 2.0])
    set(gca,'XTickLabel',[{'< 7 days'}, {''},{''},{''},{''}, {'? 7 days'}, {''}],'FontSize', 16);
    set(gca,'YTicklabel',[{'-0.5'},{'0'},{'0.5'}, {'1'},{'1.5'},{''}], 'FontSize',16);
    legend({'Test data','Training data'},'FontSize',16, ...
              'location','NW');
    legend boxoff
    set(gcf, 'Position',[10,10,900, 1500]);
    set(gca,'fontname','arial')
    
     % Plotting residuals
    subplot(2,2,4);
    set(gcf, 'Position',[10,10,850, 2000]);
    Errors_Binary_Ytest_MayagayaANN = Binary_Ytest_Mayagaya_ANN - MayagayaTestPredictedAge_ClassANN';
    Errors_Binary_Ytrain_MayagayaANN = Binary_Ytrain_Mayagaya_ANN - MayagayaTrainPredictedAge_ClassANN';
    plot(Binary_Ytest_Mayagaya_ANN, Errors_Binary_Ytest_MayagayaANN, 'bo', Binary_Ytrain_Mayagaya_ANN+0.05,Errors_Binary_Ytrain_MayagayaANN, 'r^')
    xlabel('Actual age class in days', 'FontSize',16);
    ylabel('Residuals in days', 'FontSize', 16);
    xlim([-0.1 1.1])
    ylim([-1.2 1.2])
    set(gca,'XTickLabel',[{'0'}, {''},{''},{''},{''}, {'1'}],'FontSize', 16);
    %set(gca,'YTicklabel',[{'-0.5'},{'0'},{'0.5'}, {'1'},{'1.5'},{''}], 'FontSize',16);
    legend({'Test data','Training data'},'FontSize',16, ...
              'location','NW');
    legend boxoff
    set(gca,'fontname','arial')
    hold on
    
      %box plot 
    figure()
    subplot(2,2,4);
    h = boxplot(MayagayaTestPredictedAge_ClassANN', Binary_Ytest_Mayagaya_ANN, 'Colors','k');
    set(h,{'linew'},{1})
    xlabel('Actual age class', 'FontSize',14);
    ylabel('Estimated age class', 'FontSize', 14);
    set(gca,'XTickLabel',[{'0'}, {'1'}],'FontSize', 14);
    %set(gca,'YTickLabel',[{'-5'}, {'0'},{'5'},{'10'},{'15'}, {'20'}, {'25'}, {''}],'FontSize', 16);
    set(gcf, 'Position',[10,10,850, 2000]);
    set(gca,'fontname','arial');
     
    
     subplot(2,2,4);
     xdata_MayagayaBinaryANN = [1 2];
     ydata_MayagayaBinaryANN = [Average_TrueYoung_BinaryMyg_ANN/AverageNumberBinaryMayagaya_Predicted_Young_ANN*100 Average_WrongYoung_BinaryMyg_ANN/AverageNumberBinaryMayagaya_Predicted_Young_ANN*100;  Average_TrueOld_BinaryMyg_ANN/AverageNumberBinary_Predicted_Old_ANN*100 Average_WrongOld_BinaryMyg_ANN/AverageNumberBinary_Predicted_Old_ANN*100];
%      ydata_MayagayaBinaryANN = [Average_TrueGreaterSeven_BinaryMyg_ANN/AverageNumberBinaryMayagaya_PredictedGreaterOrEqual_7d_ANN*100 Average_WrongGreaterSeven_BinaryMyg_ANN/AverageNumberBinaryMayagaya_PredictedGreaterOrEqual_7d_ANN*100;Average_TrueLessSeven_BinaryMyg_ANN/AverageNumberBinaryMayagaya_PredictedLessThan_7d_ANN*100 Average_WrongLessSeven_BinaryMyg_ANN/AverageNumberBinaryMayagaya_PredictedLessThan_7d_ANN*100];
     E = [std(TrueYoung_BinaryMygANN./NumberBinaryMayagaya_Predicted_Young_ANN)*100, std(WrongYoung_BinaryMygANN./NumberBinaryMayagaya_Predicted_Young_ANN)*100; std(TrueOld_BinaryMygANN./NumberBinaryMayagaya_Predicted_Old_ANN)*100, std(WrongOld_BinaryMygANN./NumberBinaryMayagaya_Predicted_Old_ANN)*100];
     hb = bar(xdata_MayagayaBinaryANN,ydata_MayagayaBinaryANN,1);
     hold on
     xe = [0.85 1.15; 1.85 2.15];
     errorbar(xe,ydata_MayagayaBinaryANN,E,'*','CapSize',17,'LineWidth',2)
%      set(hb(2),'facecolor',[120/256 198/256 83/256])%got color value using color picker
     set(hb(2),'facecolor',[244/256 134/256 66/256])
     ylabel('% Mosquitoes','FontSize', 14);
     xlabel('Model prediction','FontSize', 14); 
     set(gca,'XTick',[1 2]);
     set(gca,'XTickLabel',[{'< 7 days (N = 139)'}, {'? 7 days (N = 223)'}],'FontSize', 14);
     ylim([0 140])
     set(gca,'YTickLabel',[{'0'}, {'20'},{'40'},{'60'}, {'80'},{'100'},{''},{''}],'FontSize', 14);
     
     legend({'Correct prediction','False prediction'},'FontSize',14, ...
              'location','NW');
     legend boxoff
     set(gca,'fontname','arial')
     
     hold on
     
%      AgeCategorized_Mayagaya = [BinaryAge_Mayagaya BinaryAbsorbances_Mayagaya];
     
     C_ReplicateBinaryPLS = 5; % setting number of replicates
  for iBinary = 1:C_ReplicateBinaryPLS
     K_ClassPLS = randperm(size(AgeCategorized_Mayagaya,1));
     Randomised_AgeCategorized = AgeCategorized_Mayagaya(K_ClassPLS(1:size(AgeCategorized_Mayagaya,1)),:);
     
     n = size(Randomised_AgeCategorized,1);
     TrainSize = randsample(n,(0.7*n));
     TestSample = Randomised_AgeCategorized(~ismember(1:n,TrainSize),:); % Testing data
     TrainSample = Randomised_AgeCategorized(ismember(1:n,TrainSize),:); %Training sample
     Binary_Xtrain_MayagayaPLS = TrainSample(:,3:end);
     Binary_Ytrain_MayagayaPLS = TrainSample(:,1);
     Binary_Xtest_MayagayaPLS = TestSample(:,3:end); % Query data
     Binary_Ytest_MayagayaPLS = TestSample(:,1);
     
       
     % Training PLS binary classification model on ten component using ten
     % fold cross validation
     [XtrainL, ytrainl, XtrainS, YtrainS, BinaryMayagaya_beta] = plsregress(Binary_Xtrain_MayagayaPLS , Binary_Ytrain_MayagayaPLS, 10, 'CV', 10);
   
   
     A_BinaryMayagaya = [ones(size(Binary_Xtest_MayagayaPLS,1), 1),  Binary_Xtest_MayagayaPLS];
     BinaryMayagaya_PredictedPLS = A_BinaryMayagaya * BinaryMayagaya_beta;
     B_BinaryMayagaya = [ones(size(Binary_Xtrain_MayagayaPLS,1),1), Binary_Xtrain_MayagayaPLS];
     BinaryMayagaya_YtrainPredictedPLS = B_BinaryMayagaya * BinaryMayagaya_beta;
    
     
     %Scoring the model accuracy
     NumberBinaryMayagaya_Ytest_Young_PLS(iBinary) = sum(Binary_Ytest_MayagayaPLS == 0);
     NumberBinaryMayagaya_Ytest_Old_PLS(iBinary) = sum(Binary_Ytest_MayagayaPLS == 1);
     
     NumberCorrectlyPredictedYoung_BinaryMygPLS = sum((Binary_Ytest_MayagayaPLS == 0)&(BinaryMayagaya_PredictedPLS < 0.5));
     NumberWronglyPredictedYoung_BinaryMygPLS = sum((Binary_Ytest_MayagayaPLS == 1)&(BinaryMayagaya_PredictedPLS < 0.5));
     NumberCorrectlyPredictedOld_BinaryMygPLS = sum((Binary_Ytest_MayagayaPLS == 1)&(BinaryMayagaya_PredictedPLS >= 0.5));
     NumberWronglyPredictedOld_BinaryMygPLS = sum((Binary_Ytest_MayagayaPLS == 0)&(BinaryMayagaya_PredictedPLS >= 0.5));
     AccuracyPLS_BinaryMyg = (NumberCorrectlyPredictedYoung_BinaryMygPLS + NumberCorrectlyPredictedOld_BinaryMygPLS)/length(Binary_Ytest_MayagayaPLS);
     
     TrueYoung_BinaryMygPLS(iBinary) = NumberCorrectlyPredictedYoung_BinaryMygPLS;
     WrongYoung_BinaryMygPLS(iBinary) = NumberWronglyPredictedYoung_BinaryMygPLS;
     TrueOld_BinaryMygPLS(iBinary) =  NumberCorrectlyPredictedOld_BinaryMygPLS;
     WrongOld_BinaryMygPLS(iBinary) = NumberWronglyPredictedOld_BinaryMygPLS;
     AccuracyPLS_BinaryClassifier_BinaryMyg(iBinary) = AccuracyPLS_BinaryMyg;
        
  end
  
%      save BinaryMayagaya_beta; % saving the whole space 
%      save('Trained_PLSClassificationModel_Arabiensis.mat', 'BinaryMayagaya_beta'); % saving only trained model
%      save('Trained_PLSClassificationModel_Ara_FiveCuttOff.mat', 'BinaryMayagaya_beta');
%      save('Trained_PLSClassificationModel_Ara_FiveCuttOff_Balanced.mat', 'BinaryMayagaya_beta');
%      load Trained_PLSRegressionModel_Arabiensis.mat;
%      
%      save('Trained_PLSClassBalanced_Ara.mat', 'BinaryMayagaya_beta');
     
  
     AverageNumberBinaryMayagaya_Ytest_Young_PLS = mean(NumberBinaryMayagaya_Ytest_Young_PLS);
     NumberBinaryMayagaya_Predicted_Young_PLS =   TrueYoung_BinaryMygPLS + WrongYoung_BinaryMygPLS;
     AverageNumberBinaryMayagaya_Predicted_Young_PLS = mean(NumberBinaryMayagaya_Predicted_Young_PLS);
     AverageNumberBinaryMayagaya_Ytest_Old_PLS = mean(NumberBinaryMayagaya_Ytest_Old_PLS);
     NumberBinaryMayagaya_Predicted_Old_PLS = TrueOld_BinaryMygPLS + WrongOld_BinaryMygPLS;
     AverageNumberBinaryMayagaya_Predicted_Old_PLS = mean( NumberBinaryMayagaya_Predicted_Old_PLS);
     Average_AccuracyPLS_BinaryClassifier_BinaryMyg_PLS = mean(AccuracyPLS_BinaryClassifier_BinaryMyg);
     Std_dev_AccuracyPLS_BinaryClassifier_BinaryMyg = std(AccuracyPLS_BinaryClassifier_BinaryMyg)
     Average_TrueYoung_BinaryMygPLS = mean(TrueYoung_BinaryMygPLS);
     Average_WrongYoung_BinaryMygPLS = mean(WrongYoung_BinaryMygPLS);
     Average_TrueOld_BinaryMygPLS =  mean(TrueOld_BinaryMygPLS);
     Average_WrongOld_BinaryMygPLS = mean(WrongOld_BinaryMygPLS);
     
     SensitivityPLS_BinaryClassifier_BinaryMyg = TrueOld_BinaryMygPLS./NumberBinaryMayagaya_Ytest_Old_PLS
     Average_SensitivityPLS_BinaryClassifier_BinaryMyg = mean(SensitivityPLS_BinaryClassifier_BinaryMyg)
     Std_dev_SensitivityPLS_BinaryClassifier_BinaryMyg = std(SensitivityPLS_BinaryClassifier_BinaryMyg)
     SpecificityPLS_BinaryClassifier_BinaryMyg = TrueYoung_BinaryMygPLS./NumberBinaryMayagaya_Ytest_Young_PLS
     Average_SpecificityPLS_BinaryClassifier_BinaryMyg = mean(SpecificityPLS_BinaryClassifier_BinaryMyg)
     Std_dev_SpecificityPLS_BinaryClassifier_BinaryMyg = std (SpecificityPLS_BinaryClassifier_BinaryMyg)
     
     
     [h,p_DirectClass_Accuracy_PLS_ANN_TwoTail,ci,stats] =ttest(AccuracyANN_BinaryClassifier_BinaryMyg,AccuracyPLS_BinaryClassifier_BinaryMyg);% Two tail
     [h,p_DirectClass_Accuracy_PLS_ANN_OneTail,ci,stats] =ttest(AccuracyANN_BinaryClassifier_BinaryMyg,AccuracyPLS_BinaryClassifier_BinaryMyg,'Tail','right');% One tail 
     [h,p_DirectClass_Sensitivity_PLS_ANN_TwoTail,ci,stats] =ttest(SensitivityANN_BinaryClassifier_BinaryMyg,SensitivityPLS_BinaryClassifier_BinaryMyg);% Two tail
     [h,p_DirectClass_Sensitivity_PLS_ANN_OneTail,ci,stats] =ttest(SensitivityANN_BinaryClassifier_BinaryMyg,SensitivityPLS_BinaryClassifier_BinaryMyg, 'Tail','right');% One tail
     [h,p_DirectClass_Specificity_PLS_ANN_TwoTail,ci,stats] =ttest(SpecificityANN_BinaryClassifier_BinaryMyg,SpecificityPLS_BinaryClassifier_BinaryMyg); % Two tail
     [h,p_DirectClass_Specificity_PLS_ANN_OneTail,ci,stats] =ttest(SpecificityANN_BinaryClassifier_BinaryMyg,SpecificityPLS_BinaryClassifier_BinaryMyg, 'Tail','right'); % One tail
    
                      %% Testing trained binary classifier model on independent test set

           load cleaned_DS7.mat
           ITS_Absorbances_PLS = Cleaned_Alex_data_Ara_Ifa(:,3:end);
           ITS_AgeClassLabels = Cleaned_Alex_data_Ara_Ifa(:,1);
           %A_BinaryMayagaya = [ones(size(ITS_Absorbances_PLS,1), 1),  ITS_Absorbances_PLS];
           PredictedAgeClass_ITS_PLS = A_BinaryMayagaya * Mayagaya_beta_PLS;

%            
%            Number_ITS_Young = sum(ITS_AgeClassLabels==0); 
%            Number_ITS_Old = sum(ITS_AgeClassLabels==1);
%            
%             % Independent test set (ITS)
%            NumberCorrectlyPredicted_Young = sum((ITS_AgeClassLabels == 0)&(PredictedAgeClass_ITS_PLS < 0.5));
%            NumberWronglyPredicted_Young = sum((ITS_AgeClassLabels==1)&(PredictedAgeClass_ITS_PLS < 0.5));
%            NumberCorrectlyPredicted_Old = sum((ITS_AgeClassLabels==1)&(PredictedAgeClass_ITS_PLS >= 0.5));
%            NumberWronglyPredicted_Old = sum((ITS_AgeClassLabels == 0)&(PredictedAgeClass_ITS_PLS >= 0.5));
%            Accuracy_ITS = (NumberCorrectlyPredicted_Young + NumberCorrectlyPredicted_Old)/length( ITS_AgeClassLabels);

%            Sensitivity_ITS =  NumberCorrectlyPredicted_Old/Number_ITS_Old
%            Specificity_ITS = NumberCorrectlyPredicted_Young/Number_ITS_Young
%            
     
     
     
    subplot(2,2,1);
    plot(Binary_Ytest_MayagayaPLS,  BinaryMayagaya_PredictedPLS, 'bo', Binary_Ytrain_MayagayaPLS+0.05, BinaryMayagaya_YtrainPredictedPLS, 'r^')
    xlabel('Actual age class', 'FontSize',16);
    ylabel('Estimated age class', 'FontSize', 16);
    xlim([-0.1 1.2])
    ylim([-0.5 2.0])
    set(gca,'XTickLabel',[{'< 7 days'}, {''},{''},{''},{''}, {'? 7 days'}, {''}],'FontSize', 16);
    set(gca,'YTicklabel',[{'-0.5'},{'0'},{'0.5'}, {'1'},{'1.5'},{''}], 'FontSize',16);
    legend({'Test data','Training data'},'FontSize',16, ...
              'location','NW');
    legend boxoff
    set(gcf, 'Position',[10,10,900, 1500]);
    set(gca,'fontname','arial')
     
    hold on
    
     % Plotting residuals
    subplot(2,2,3);
    set(gcf, 'Position',[10,10,850, 2000]);
    Errors_Binary_Ytest_MayagayaPLS = Binary_Ytest_MayagayaPLS - BinaryMayagaya_PredictedPLS;
    Errors_Binary_Ytrain_MayagayaPLS = Binary_Ytrain_MayagayaPLS - BinaryMayagaya_YtrainPredictedPLS;
    plot(Binary_Ytest_MayagayaPLS, Errors_Binary_Ytest_MayagayaPLS, 'bo', Binary_Ytrain_MayagayaPLS+0.05,Errors_Binary_Ytrain_MayagayaPLS, 'r^')
    xlabel('Actual age class in days', 'FontSize',16);
    ylabel('Residuals in days', 'FontSize', 16);
    xlim([-0.1 1.1])
    ylim([-1.2 1.2])
    set(gca,'XTickLabel',[{'0'}, {''},{''},{''},{''}, {'1'}],'FontSize', 16);
    %set(gca,'YTicklabel',[{'-0.5'},{'0'},{'0.5'}, {'1'},{'1.5'},{''}], 'FontSize',16);
    legend({'Test data','Training data'},'FontSize',16, ...
              'location','NW');
    legend boxoff
    set(gca,'fontname','arial')
    hold off
    
    % box plot
    subplot(2,2,3);
    h = boxplot(BinaryMayagaya_PredictedPLS, Binary_Ytest_MayagayaPLS,'Colors','k' );
    set(h,{'linew'},{1})
    xlabel('Actual age class', 'FontSize',14);
    ylabel('Estimated age class', 'FontSize', 14);
    set(gca,'XTickLabel',[{'0'}, {'1'}],'FontSize', 14);
    %set(gca,'YTickLabel',[{'-5'}, {'0'},{'5'},{'10'},{'15'}, {'20'}, {'25'}, {''}],'FontSize', 16);
    set(gcf, 'Position',[10,10,850, 2000]);
    set(gca,'fontname','arial');
     
     subplot(2,2,3);
     xdata_MayagayaBinaryPLS = [1 2];
     ydata_MayagayaBinaryPLS = [Average_TrueYoung_BinaryMygPLS/AverageNumberBinaryMayagaya_Predicted_Young_PLS*100 Average_WrongYoung_BinaryMygPLS/AverageNumberBinaryMayagaya_Predicted_Young_PLS*100;  Average_TrueOld_BinaryMygPLS/AverageNumberBinaryMayagaya_Predicted_Old_PLS*100 Average_WrongOld_BinaryMygPLS/AverageNumberBinaryMayagaya_Predicted_Old_PLS*100];
     E = [std(TrueYoung_BinaryMygPLS./NumberBinaryMayagaya_Predicted_Young_PLS)*100, std(WrongYoung_BinaryMygPLS./NumberBinaryMayagaya_Predicted_Young_PLS)*100; std(TrueOld_BinaryMygPLS./NumberBinaryMayagaya_Predicted_Old_PLS)*100, std(WrongOld_BinaryMygPLS./NumberBinaryMayagaya_Predicted_Old_PLS)*100];
     hb = bar(xdata_MayagayaBinaryPLS,ydata_MayagayaBinaryPLS,1);
     hold on
     xe = [0.85 1.15; 1.85 2.15]; % positioning bars on the graph
     errorbar(xe,ydata_MayagayaBinaryPLS,E,'*','CapSize',17,'LineWidth',2)
%      set(hb(2),'facecolor',[120/256 198/256 83/256])%got color value using color picker
     set(hb(2),'facecolor',[244/256 134/256 66/256])
     ylabel('% Mosquitoes','FontSize', 14);
     xlabel('Model prediction','FontSize', 14); 
     set(gca,'XTick',[1 2]);
     set(gca,'XTickLabel',[{'< 7 days (N = 139)'}, {'? 7 days (N = 223)'}],'FontSize', 14);
     ylim([0 140])
     set(gca,'YTickLabel',[{'0'}, {'20'},{'40'},{'60'}, {'80'},{'100'},{''},{''}],'FontSize', 14);
     
     legend({'Correct prediction','False prediction'},'FontSize',14, ...
              'location','NW');
     legend boxoff
     set(gca,'fontname','arial')
     
      hold off
%       print(gcf,'FigureTwo.tif','-dtiff','-r300')
%       print(gcf,'Fig4.tif','-dtiff','-r300');
  
%% Testing the hypothesis that directly training ANN binary classifier scores higher
   % accuracy than a ANN regresser interpreted as a binary classifier.
   
     [h,p_DirectClass_Accuracy_ANNB_ANNR_TwoTail,ci,stats] =ttest(AccuracyANN_BinaryClassifier_BinaryMyg, AccuracyANN_BinaryClassifier_Mayagaya);% Two tail
     [h,p_DirectClass_Accuracy_ANNB_ANNR_OneTail,ci,stats] =ttest(AccuracyANN_BinaryClassifier_BinaryMyg,AccuracyANN_BinaryClassifier_Mayagaya,'Tail','right');% One tail 
     [h,p_DirectClass_Sensitivity_ANNB_ANNR_TwoTail,ci,stats] =ttest(SensitivityANN_BinaryClassifier_BinaryMyg,SensitivityANN_BinaryClassifier_Mayagaya);% Two tail
     [h,p_DirectClass_Sensitivity_ANNB_ANNR_OneTail,ci,stats] =ttest(SensitivityANN_BinaryClassifier_BinaryMyg,SensitivityANN_BinaryClassifier_Mayagaya, 'Tail','right');% One tail
     [h,p_DirectClass_Specificity_ANNB_ANNR_TwoTail,ci,stats] =ttest(SpecificityANN_BinaryClassifier_BinaryMyg,SpecificityANN_BinaryClassifier_Mayagaya); % Two tail
     [h,p_DirectClass_Specificity_ANNB_ANNR_OneTail,ci,stats] =ttest(SpecificityANN_BinaryClassifier_BinaryMyg,SpecificityANN_BinaryClassifier_Mayagaya,'Tail','right'); % One tail
     
%% Testing the hypothesis that directly training PLS binary classier score higher
   % accuracy than a PLS regresser interpreted as a binary classifier.
     
     [h,p_DirectClass_Accuracy_PLSB_PLSR_TwoTail,ci,stats] =ttest(AccuracyPLS_BinaryClassifier_BinaryMyg, AccuracyPLS_BinaryClassifier_Mayagaya);% Two tail
     [h,p_DirectClass_Accuracy_PLSB_PLSR_OneTail,ci,stats] =ttest(AccuracyPLS_BinaryClassifier_BinaryMyg, AccuracyPLS_BinaryClassifier_Mayagaya, 'Tail','right');% One tail 
     [h,p_DirectClass_Sensitivity_PLSB_PLSR_TwoTail,ci,stats] =ttest(SensitivityPLS_BinaryClassifier_BinaryMyg, SensitivityPLS_BinaryClassifier_Mayagaya);% Two tail
     [h,p_DirectClass_Sensitivity_PLSB_PLSR_OneTail,ci,stats] =ttest(SensitivityPLS_BinaryClassifier_BinaryMyg, SensitivityPLS_BinaryClassifier_Mayagaya,'Tail','right');% One tail
     [h,p_DirectClass_Specificity_PLSB_PLSR_TwoTail,ci,stats] =ttest(SpecificityPLS_BinaryClassifier_BinaryMyg, SpecificityPLS_BinaryClassifier_Mayagaya); % Two tail
     [h,p_DirectClass_Specificity_PLSB_PLSR_OneTail,ci,stats] =ttest(SpecificityPLS_BinaryClassifier_BinaryMyg, SpecificityPLS_BinaryClassifier_Mayagaya,'Tail','right'); % One tail
     


     