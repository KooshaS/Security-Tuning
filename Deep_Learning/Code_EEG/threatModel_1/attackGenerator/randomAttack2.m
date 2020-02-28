clc, clear all

XDataAttack = zeros(8, 8, 8, 1000);
for i = 1:1000
    
    i
    r = randi([-1024 1024], 8, 8, 8);
    XDataAttack(:, :, :, i) = r;
    
end


