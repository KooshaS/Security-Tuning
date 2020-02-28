function sec = securityEval(net, history, XAttack, thr)

    replayCount = 0;
    successAtt = 0;
    flag = 0;
    for i = 1:size(XAttack, 4)

        img = XAttack(:, :, :, i);
        for j = 1:size(history, 4)
            distance = sum(abs(double(history(:, :, :, j)) - double(img)), 'all');
            if distance < 1000 * thr            
                flag = 1;
                replayCount = replayCount + 1;
                break;
            end
        end

        if flag
            flag = 0;
        else
            if categorical(0) == classify(net, img)
                successAtt = successAtt + 1;            
            end
        end
       
    end

    sec = 100 * successAtt / size(XAttack, 4);

end


