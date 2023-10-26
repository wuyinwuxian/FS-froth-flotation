function nonUnifiormaity = sloveNonUnifiormaity(im,graylevel,d,a,t)
    ngldm = getgraymatrix(im,graylevel,d,a,t);
    [~, col] = size(ngldm);
    sumQ = sum(sum(ngldm));
    tempSum = 0;
    tempSS = 0;
    for i = 1 : col
        tempSum = sum(ngldm(: , i));
        tempSS = tempSum * tempSum + tempSS;
    end
    nonUnifiormaity = tempSum / sumQ;    %15 数值非均匀度
end