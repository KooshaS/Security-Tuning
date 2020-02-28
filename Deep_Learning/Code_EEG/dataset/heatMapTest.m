clc, clear all

% data = gallery('invhess',20);
% hmo = HeatMap(data);

load 'EEG_ClosedEyes'

subject_EEG = squeeze(ClosedEyes(1, :, :));
img = uint8(zeros(8, 8));

for i = 1:8
    for j = 1:8
        img(i, j) = subject_EEG((i - 1) * 8 + j, 100);
    end
end

imshow(img)
hmo = HeatMap(img)

