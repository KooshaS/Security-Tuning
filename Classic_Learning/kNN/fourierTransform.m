function y = fourierTransform (signal)
                            
    L = 80;             % Length of signal
    low_F = 8;
    high_F = 13;
    y = zeros(high_F - low_F + 1, size(signal, 2));
       
    for j = 1:size(signal, 2)
        Y = fft(signal(:, j));
        P2 = abs(Y/L);
        P1 = P2(1:L/2+1);
        P1(2:end-1) = 2*P1(2:end-1);
        y(:, j) = P1(low_F:high_F); 
    end
    
end