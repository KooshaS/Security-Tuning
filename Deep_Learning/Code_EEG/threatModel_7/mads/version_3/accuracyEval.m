function [acc, history] = accuracyEval(net, XTest, YTest)

    history = uint8(zeros(8, 8, 8, 1));
    history(:, :, :, 1) = XTest(:, :, :, 1);
    replayCount = 0;
    correct = 1;
    notCorrect = 0;
    flag = 0;
    for i = 2:size(XTest, 4)

        img = XTest(:, :, :, i);
        for j = 1:size(history, 4)
            distance = sum(abs(double(history(:, :, :, j)) - double(img)), 'all');
            if distance < 10000            
                flag = 1;
                replayCount = replayCount + 1;
                break;
            end
        end

        if flag
            notCorrect = notCorrect + 1;
            flag = 0;
        else
            if YTest(i) == classify(net, img)
                correct = correct + 1;
            else
                notCorrect = notCorrect + 1;
            end
        end

        history = cat(4, history, img);

    end

    acc = 100 * correct / (correct + notCorrect);

end


