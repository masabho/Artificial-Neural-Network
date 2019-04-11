% Age Grading Lab An.gambiae Using Feed-Forward Artificial Neural Nets
% Author: Masabho Peter Milali
% Modified: March 2018
% Created: September 2018 
% MATLAB R2016b

%% Loading and cleaning Lab Anopheles
  clear;
  clc;
   %fileName = 'Workspaces/DS7.mat';
   %fileName = 'Workspaces/DS4.mat';
   %fileName = 'Workspaces/DS3.mat';
   %fileName = 'Workspaces/DS2.mat';
   fileName = 'Workspaces/IFA_GA.mat';
   %fileName = 'Workspaces/DS1.mat';
   if (exist(fileName, 'file'))
       load(fileName);
   else 
      %[num,txt,raw] = xlsread('DS7.xlsx');
      %[num,txt,raw] = xlsread('DS4',1); 
      [num,txt,raw] = xlsread('IFA_GA',1); % 
      %[num,txt,raw] = xlsread('DS3.xlsx',1);
      %[num,txt,raw] = xlsread('DS2.xlsx'); 
      %[num,txt,raw] = xlsread('DS1.xlsx'); 
      save(fileName, 'num', 'txt', 'raw');
   end;
   
   %% Lab_An.arabiensis_Pemba
%    Absorbances = num(:,4:end);
%      %Absorbances = num(:,4:end);% DS7
%    Age_Labels = num(:,1:2);
%    num = Absorbances;      
%    Age_Labels = num(:,1:2);        
  %% Data visualization 

   %Figure showing Spectra before processing
    m = (350:2500)';
       %m = repmat(m,1,871);
    m = repmat(m,1,size(num,1));
    figure()
    plot(m,num');
    ylim([0.0 3.0]);
    xlim([200 2600]);
    legend({'Spectra before cleaning'},...
         'location','NW');
     legend boxoff;
%     saveas(figure(1),'Spectra before cleaning.png');
    

  %%   Mayagaya et. al pre-processing
     X_trimmed = num;
     cols2remove = [1:150, (length(num)-149):length(num)];
     X_trimmed(:,cols2remove)=[];
      %X_trimmed = cell2mat(matrix); % converting data from cell to double 
      
     
      % Figure of spectra after trimming noisy ends
      M_trimmed = (500:2350)';
      M_trimmed = repmat(M_trimmed,1,(size(X_trimmed,1)));
      figure()
      plot(M_trimmed,X_trimmed');
%       ylim([0.0 2.0])
%       xlim([200 2600]);
      legend({'Spectra with noisy ends trimmed off'},...
         'location','NW');
      legend boxoff;
        
    %% Age labels for IFA_ARA
    AgeMosq = (raw(:,1));
    nMosq = size(AgeMosq,1);
    age = zeros(nMosq,1);
    for i = 1 : nMosq
        age(i) = getMosqAge(AgeMosq{i});
        Age_Mosquitoes = age;
    end

    
%      Age_Mosquitoes;
%      Regression_MayagayaCleaning_Data = [Age_Mosquitoes X_trimmed]; % Combining X and Y values
%      %Randomizing spectra
%      CleanedLab_Arabiensis_Ifakara = Regression_MayagayaCleaning_Data;
%      save 'CleanedLab_Arabiensis_Ifakara.mat' 'CleanedLab_Arabiensis_Ifakara';

%% Saving Cleaned spectra according to mayagaya
     IFA_GA = [Age_Mosquitoes X_trimmed];
     save 'IFA_GA.mat' 'IFA_GA';
     CleanedLab_arabiensis_Pemba = [Age_Labels X_trimmed];
     save 'CleanedLab_arabiensis_Pemba.mat' 'CleanedLab_arabiensis_Pemba';
     CleanedLab_Arabiensis_Ifakara = Regression_MayagayaCleaning_Data;
     save 'CleanedLab_Arabiensis_Ifakara.mat' 'CleanedLab_Arabiensis_Ifakara';
     Cleaned_Alex_data_Ara_Ifa = [Age_Labels X_trimmed];
     save 'Cleaned_Alex_data_Ara_Ifa.mat' 'Cleaned_Alex_data_Ara_Ifa';
     cleaned_DS1 = [Age_Labels X_trimmed];
     save 'cleaned_DS1.mat' 'cleaned_DS1';
     cleaned_DS2 = [Age_Labels X_trimmed];
     save 'cleaned_DS2.mat' 'cleaned_DS2';
     cleaned_DS3 = [Age_Labels X_trimmed];
     save 'cleaned_DS3.mat' 'cleaned_DS3';
     cleaned_DS4 = [Age_Labels X_trimmed];
     save 'cleaned_DS4.mat' 'cleaned_DS4';
     cleaned_DS5 = [Age_Labels X_trimmed];
     save 'cleaned_DS5.mat' 'cleaned_DS5';
     
     cleaned_DS6 = [Age_Labels X_trimmed];
     save 'cleaned_DS6.mat' 'cleaned_DS6';
     cleaned_DS11 = [Age_Labels X_trimmed];
     save 'cleaned_DS11.mat' 'cleaned_DS11';
     
     cleaned_DS12 = [Age_Labels X_trimmed];
     save 'cleaned_DS12.mat' 'cleaned_DS12';
     
     cleaned_DS13 = [Age_Labels X_trimmed];
     save 'cleaned_DS13.mat' 'cleaned_DS13';
     
     cleaned_DS14 = [Age_Labels X_trimmed];
     save 'cleaned_DS14.mat' 'cleaned_DS14';
     
     cleaned_DS15 = [Age_Labels X_trimmed];
     save 'cleaned_DS15.mat' 'cleaned_DS15';
     

%% Loading and cleaning Lab Aedes aegypti and wolbachia infected Aedes aegypti
  clear;
  clc;
   fileName = 'Workspaces/Aedes_aegypti_Age.mat';
   %fileName = 'Workspaces/Wolbachia_Infection.mat'; % Wolbachia infected Aedes
   if (exist(fileName, 'file'))
       load(fileName);
   else 
      [num,txt,raw] = xlsread('Aedes_aegypti_Age.xlsx');% uninfected
      %[num,txt,raw] = xlsread('Wolbachia_Infection.xlsx');%Wolbachia infected Aedes
      save(fileName, 'num', 'txt', 'raw');
   end;
 
 %% Uninfected aedes
   Female = num(1:600,:);
   Female_Absorbances = Female(:,7:end);
   Female_Age_Labels = Female(:,3);
   FemaleData = [Female_Age_Labels Female_Absorbances];
   SortedFemaleData = sortrows(FemaleData,1);
   
   Male = num(601:end,:);
   Male_Absorbances = Male(:,7:end);
   Male_Age_Labels = Male(:,3);
   MaleData = [Male_Age_Labels  Male_Absorbances];
   Sorted_MaleData = sortrows(MaleData);
   
%% Wolbachia Infected aedes
   Female = num(1:600,:);
   Female_Absorbances = Female(:,7:end);
   Female_Age_Labels = Female(:,3);
   FemaleData = [Female_Age_Labels Female_Absorbances];
   SortedFemaleData = sortrows(FemaleData,1);
   
   Male = num(601:end,:);
   Male_Absorbances = Male(:,7:end);
   Male_Age_Labels = Male(:,3);
   MaleData = [Male_Age_Labels  Male_Absorbances];
   Sorted_MaleData = sortrows(MaleData);
   %SortedMosquitoes = Sorted_MaleData;
   % SortedMosquitoes = sortrows(TrimmedFemale_Aedes_Data);
  % SortedMosquitoes = [AgeClass_SevenCutOff X_trimmed];
   %Class_Mosquitoes = [Cleaned_Alex_data_Ara_Ifa(:,1) Cleaned_Alex_data_Ara_Ifa(:,3:end)];
   
%      SortedMosquitoes = sortrows(Class_Mosquitoes, 1); % sorted according to age labels
%      NumberLessSeven = size((SortedMosquitoes(SortedMosquitoes ==1 |SortedMosquitoes == 5|SortedMosquitoes == 9)),1) % size of Mosquitoes less seven days
% %      SizeGreaterOrEqualSeven = size((SortedMosquitoes(SortedMosquitoes==13|SortedMosquitoes==17|SortedMosquitoes == 21|SortedMosquitoes == 25|SortedMosquitoes == 30)),1)
%      MosqLessSeven = SortedMosquitoes(1: NumberLessSeven,:);
%      MosqLessSeven_New = [zeros(size((MosqLessSeven),1),1),MosqLessSeven];
%      MosqGreaterOrEqualSeven = SortedMosquitoes((NumberLessSeven + 1): size((SortedMosquitoes),1),:);
%      MosqGreaterOrEqualSeven_New = [ones(size((MosqGreaterOrEqualSeven),1),1),MosqGreaterOrEqualSeven];
%      
%      Unbalanced_AgeClass_SevenCutOff = [MosqLessSeven_New ; MosqGreaterOrEqualSeven_New];
       TrimmedFemale_Aedes_Data =  Unbalanced_AgeClass_SevenCutOff;
       save 'cleaned_DS10.mat' 'TrimmedFemale_Aedes_Data'; 
       save 'cleaned_DS9.mat' 'TrimmedMale_Aedes_Data';
%      AgeCategorized_Mayagaya = Unbalanced_AgeClass_SevenCutOff;
   
%% Data visualization 

   %Figure showing Spectra before pre-processing
    m = (350:2500)';
    F = repmat(m,1,size(Female,1)); % female aedes
    M = repmat(m,1,size(Male,1)); % male aedes
    figure()
    plot(m,Female_Absorbances'); %female aedes
    plot(M,Male_Absorbances'); % male aedes
    ylim([0.0 3.0]);
    xlim([200 2600]);
    legend({'Spectra before cleaning (Male aedes)'},...
         'location','NW');
     legend boxoff;
%     saveas(figure(1),'Spectra before cleaning.png');
      print(gcf,'Fig_MaleAedesBeforeCleaning_.tif','-dtiff','-r300');
    
%% Mayagaya et. al spectra pre-processing
 
     %  Mayagaya et. al cleaning
     matrix = Female_Absorbances; % female aedes
     Matrix = Male_Absorbances; % male aedes
     cols2remove = [1:150, (length(matrix)-149):length(matrix)]; % removing noisy ends from spectra
     matrix(:,cols2remove)=[];
     Matrix(:,cols2remove)=[];  
         %X_trimmed = cell2mat(matrix); % converting data from cell to double    
     TrimmedFemale_Absorbances = matrix;
     TrimmedMale_Absorbances = Matrix;
     
      %Figure of spectra after trimming noisy ends
      M_Trimmed = (500:length(TrimmedFemale_Absorbances))';
      M_Trimmed = repmat(M_Trimmed,1,size(TrimmedMale_Absorbances,1)); % Female
      m_trimmed = repmat(M_Trimmed,1,size(TrimmedMale_Absorbances,1));
      figure()
      plot(M_Trimmed,TrimmedFemale_Absorbances');
      plot(m_trimmed,TrimmedMale_Absorbances');
      ylim([0.0 2.0])
      xlim([200 2600]);
      legend({'Spectra with noisy ends trimmed off (Male Aedes)'},...
         'location','NW');
      legend boxoff;
      print(gcf,'Fig_FemaleAedesNoisyEndsTrimmedOff_.tif','-dtiff','-r300');
      print(gcf,'Fig_MaleAedesNoisyEndsTrimmedOff_.tif','-dtiff','-r300');
      
     
TrimmedFemale_Aedes_Data = [Female_Age_Labels TrimmedFemale_Absorbances];
save 'TrimmedFemale_Aedes_Data.mat' 'TrimmedFemale_Aedes_Data';
TrimmedMale_Aedes_Data = [Male_Age_Labels TrimmedMale_Absorbances];
save 'TrimmedMale_Aedes_Data.mat' 'TrimmedMale_Aedes_Data'; 

%% Burkinafaso data

  clear;
  clc;
   fileName = 'Workspaces/Gambiae_Burkinafasso.mat';
   if (exist(fileName, 'file'))
       load(fileName);
   else 
      [num,txt,raw] = xlsread('Gambiae_Burkinafasso.xlsx',1); % An. arabiensis Alex dataset
      save(fileName, 'num', 'txt', 'raw');
   end;
   
 %% For An.gambiae s.s burkinafaso
   Absorbances = num(:,3:end);
   Age_Labels = num(:,1);
   num = Absorbances;      
         
  %% Data visualization 

   %Figure showing Spectra before processing
    m = (350:2500)';
       %m = repmat(m,1,871);
    m = repmat(m,1,size(num,1));
    figure()
    plot(m,num');
    ylim([0.0 3.0]);
    xlim([200 2600]);
    legend({'Spectra before cleaning'},...
         'location','NW');
     legend boxoff;
%     saveas(figure(1),'Spectra before cleaning.png');
    

  %%   Mayagaya et. al pre-processing
     X_trimmed = num;
     %cols2remove = [1:150, 2002:2151]; % removing noisy ends from spectra
     cols2remove = [1:150, (length(num)-149):length(num)];
     X_trimmed(:,cols2remove)=[];
      %X_trimmed = cell2mat(matrix); % converting data from cell to double 
      
     
      % Figure of spectra after trimming noisy ends
      M_trimmed = (500:2350)';
      M_trimmed = repmat(M_trimmed,1,(size(X_trimmed,1)));
      figure()
      plot(M_trimmed,X_trimmed');
%       ylim([0.0 2.0])
%       xlim([200 2600]);
      legend({'Spectra with noisy ends trimmed off'},...
         'location','NW');
      legend boxoff;
      print(gcf,'Fig_GambiaeBurkinaNoisyEndsTrimmedOff_.tif','-dtiff','-r300');
      
      CleanedGambiae_ParityBurkinafaso_Data = [Age_Labels X_trimmed];
      CleanedGambiae_ParityBurkinafaso_Data_Independent =  CleanedGambiae_ParityBurkinafaso_Data([1:30,113:131,270:316],:);
       
      save 'CleanedGambiae_ParityBurkinafaso_Data_Independent.mat' 'CleanedGambiae_ParityBurkinafaso_Data_Independent';
      save 'CleanedGambiae_ParityBurkinafaso_Data.mat' 'CleanedGambiae_ParityBurkinafaso_Data';
      
   
   



   
