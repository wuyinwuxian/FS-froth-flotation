function secondMoment = sloveSecondMoment(im,graylevel,d,a,t)
    ngldm = getgraymatrix(im,graylevel,d,a,t);
    [row,col] = size(ngldm);
    sumQ = sum(sum(ngldm));
    tempSum = 0;
    for i = 1 : row
        for j = 1 : col
            tempSum = tempSum + ngldm(i , j)^2;
        end
    end
    secondMoment= tempSum / sumQ;   %16 ¶þ½×¾Ø
end
