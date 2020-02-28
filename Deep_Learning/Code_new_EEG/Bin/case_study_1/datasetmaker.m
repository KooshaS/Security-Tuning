%%% Arranging EEG signals in higher dimensional data format %%%
clc, clear all

load 'EEG_OpenEyes.mat'

XData0 = zeros(8, 8, 8, 1200);
YData0 = zeros(1200, 1);
for i = 1:1
    subject_EEG = squeeze(OpenEyes(i, :, :));
    for j = 1:1200
        XData0(:, :, 1, (i - 1) * 1200 + j) = reshape(subject_EEG(:, 8 * (j - 1) + 1), [8, 8]);
        XData0(:, :, 2, (i - 1) * 1200 + j) = reshape(subject_EEG(:, 8 * (j - 1) + 2), [8, 8]);
        XData0(:, :, 3, (i - 1) * 1200 + j) = reshape(subject_EEG(:, 8 * (j - 1) + 3), [8, 8]);
        XData0(:, :, 4, (i - 1) * 1200 + j) = reshape(subject_EEG(:, 8 * (j - 1) + 4), [8, 8]);
        XData0(:, :, 5, (i - 1) * 1200 + j) = reshape(subject_EEG(:, 8 * (j - 1) + 5), [8, 8]);
        XData0(:, :, 6, (i - 1) * 1200 + j) = reshape(subject_EEG(:, 8 * (j - 1) + 6), [8, 8]);
        XData0(:, :, 7, (i - 1) * 1200 + j) = reshape(subject_EEG(:, 8 * (j - 1) + 7), [8, 8]);
        XData0(:, :, 8, (i - 1) * 1200 + j) = reshape(subject_EEG(:, 8 * (j - 1) + 8), [8, 8]);        
                
        YData0((i - 1) * 1200 + j) = 0;        
    end            
end

load 'EEG_ClosedEyes.mat'

XData1 = zeros(8, 8, 8, 1200);
YData1 = zeros(1200, 1);
for i = 1:1
    subject_EEG = squeeze(ClosedEyes(i, :, :));
    for j = 1:1200
        XData1(:, :, 1, (i - 1) * 1200 + j) = reshape(subject_EEG(:, 8 * (j - 1) + 1), [8, 8]);
        XData1(:, :, 2, (i - 1) * 1200 + j) = reshape(subject_EEG(:, 8 * (j - 1) + 2), [8, 8]);
        XData1(:, :, 3, (i - 1) * 1200 + j) = reshape(subject_EEG(:, 8 * (j - 1) + 3), [8, 8]);
        XData1(:, :, 4, (i - 1) * 1200 + j) = reshape(subject_EEG(:, 8 * (j - 1) + 4), [8, 8]);
        XData1(:, :, 5, (i - 1) * 1200 + j) = reshape(subject_EEG(:, 8 * (j - 1) + 5), [8, 8]);
        XData1(:, :, 6, (i - 1) * 1200 + j) = reshape(subject_EEG(:, 8 * (j - 1) + 6), [8, 8]);
        XData1(:, :, 7, (i - 1) * 1200 + j) = reshape(subject_EEG(:, 8 * (j - 1) + 7), [8, 8]);
        XData1(:, :, 8, (i - 1) * 1200 + j) = reshape(subject_EEG(:, 8 * (j - 1) + 8), [8, 8]);        
                
        YData1((i - 1) * 1200 + j) = 1;        
    end            
end

XData = zeros(8, 8, 8, 2400);
YData = zeros(2400, 1);

XData(:, :, : , 1:1200) = XData0;
XData(:, :, : , 1201:2400) = XData1;
YData(1:1200, 1) = YData0;
YData(1201:end, 1) = YData1;


