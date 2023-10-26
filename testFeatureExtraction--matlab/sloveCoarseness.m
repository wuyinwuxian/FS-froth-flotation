function coarseness = sloveCoarseness(im,graylevel,d,a,t)
    ngldm = getgraymatrix(im,graylevel,d,a,t);  
    [row, col] = size(ngldm);
    sumQ = sum(sum(ngldm));
    tempSum = 0;
    for i = 1 : row
        for j = 1 : col
            tempSum = tempSum + double(ngldm(i,j)) * (j)^2;
        end
    end
    coarseness = tempSum / sumQ;    %ÅÝÄ­´Ö¶È
end