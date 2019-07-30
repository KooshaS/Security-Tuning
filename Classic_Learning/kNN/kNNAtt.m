%%% Brute-forcing the input to kNN classifier %%%
function sec = kNNAtt(mdl)

    class = zeros(1, 2);
    %%% Generating Adversarial Samples
    for iteration = 1: 1000

        EEG_signal = zeros(80, 1);
        for i = 1:80            
            EEG_signal(i, 1) = randi([-687, 846]);                                            
        end                                                         
        
        xAdv = fourierTransform (EEG_signal);
        [label, ~, ~] = predict(mdl, xAdv);

        class(1, label + 1) = class(1, label + 1) + 1; 
        
    end

    sec = 100 * class(1, 1) / sum(class(:));
    
end