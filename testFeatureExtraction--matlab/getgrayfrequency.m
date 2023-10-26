function [px,py] = getgrayfrequency(im,d,a,type)
[row,col] = size(im);
px=zeros(row-2*d,col-2*d);
py=zeros(row-2*d,col-2*d);
count=0;
border = [];

switch type
    case 1
        for i = 1+d:row-d
            for j = 1+d:col-d
                k = im(i,j);
                count = -1;
                for m = i - d : i + d
                    for n = j - d : j + d
                        if (abs(k - im(m,n)))<=a
                            count = count + 1;             
                        end
                    end
                end
                px(i-d,j-d) = k;
                py(i-d,j-d) = count;
            end
        end
    case 2
         for i = 1+d:row-d
            for j = 1+d:col-d
                k = im(i,j);
                if a==0 
                    count = -1;
                else 
                    count=0;
                end
                for m = i - d : i + d
                    for n = j - d : j + d
                        if (abs(k - im(m,n)))==a
                            count = count + 1;             
                        end
                    end
                end
                px(i-d,j-d) = k;
                py(i-d,j-d) = count;
            end
        end
    case 3
        for i = 1+d:row-d
            for j = 1+d:col-d

                k = im(i,j);
                border = [border,im(i-d,j-d:j+d)];
                border = [border,im(i+d,j-d:j+d)];
                border = [border,im(i-d+1:i+d-1,j-d)'];
                border = [border,im(i-d+1:i+d-1,j+d)'];
                count = 0;
                for m = 1:8*d
                    if abs(k-border(1,m))<=a
                        count = count + 1; 
                    end
                end
                border = [];
                px(i-d,j-d) = k;
                py(i-d,j-d) = count;
            end
        end
    case 4
        for i = 1+d:row-d
            for j = 1+d:col-d
                k = im(i,j);
                border = [border,im(i-d,j-d:j+d)];
                border = [border,im(i+d,j-d:j+d)];
                border = [border,im(i-d+1:i+d-1,j-d)'];
                border = [border,im(i-d+1:i+d-1,j+d)'];
                count=0;
                for m = 1:8*d
                    if abs(k-border(1,m)) == a
                        count = count + 1;             
                    end
                end
                border = [];
                px(i-d,j-d) = k;
                py(i-d,j-d) = count;
            end
        end
    case 5
        for i = 1+d:row-d
            for j = 1+d:col-d
                k = im(i,j);
                count = -1;
                for m = i - d : i + d
                    for n = j - d : j + d
                        if k<=im(m , n)
                            count = count + 1;             
                        end
                    end
                end
                px(i-d,j-d) = k;
                py(i-d,j-d) = count;
            end
        end
    case 6
        for i = 1+d:row-d
            for j = 1+d:col-d
                k = im(i,j);
                count = -1;
                for m = i - d : i + d
                    for n = j - d : j + d
                        if k>=im(m , n)
                            count = count + 1;             
                        end
                    end
                end
                px(i-d,j-d) = k;
                py(i-d,j-d) = count;
            end
        end
    case 7
        for i = 1+d:row-d
            for j = 1+d:col-d
                k = im(i,j);
                count = 0;
                for m = i - d : i + d
                    for n = j - d : j + d
                        if k~=im(m,n)
                            count = count + 1;             
                        end
                    end
                end
                px(i-d,j-d) = k;
                py(i-d,j-d) = count;
            end
        end
end

