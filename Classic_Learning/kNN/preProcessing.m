function y = preProcessing (signal)

    for i = 1:size(signal, 2)

        %% Zero-Mean Normalization
%        signal(:, i) = signal(:, i) - mean(signal(:, i));                 

        %% Low-Pass Filter of Signal
        Fs = 512;
        Hd = designfilt('lowpassfir','FilterOrder',20,'CutoffFrequency',50, ...
           'DesignMethod','window','Window',{@kaiser,3},'SampleRate',Fs);
        signal(:, i) = filter(Hd, signal(:, i));

    end

    y = signal;
    
end