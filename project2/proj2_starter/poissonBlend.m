function im_out = poissonBlend(im_object, im_mask, im_target)
%possionBlend(im_object, im_mask, im_target)
%perform possion blending
im_out = im_target;
if size(im_object) ~= size(im_target)
    disp('Error! Image sizes must be the same');
    return;
end

[h,w,c] = size(im_target);

[y_int, x_int] = find(im_mask);

kInt = size(y_int, 1);
kPix = kInt;

B = zeros(kPix, c);

im2var = zeros(h,w);
v = 1;
for x= max(min(x_int),1): min(max(x_int),w)
    for y= max(min(y_int),1): min(max(y_int),h)
       if im_mask(y,x) > 0
           im2var(y,x) = v;
           v = v + 1;
       end
    end
end

Gx = zeros(h,w,c);
Gy = zeros(h,w,c);
for i=1:c
    Gx(:,:,i) = imfilter(im_object(:,:,i), [1,-1]);
    Gy(:,:,i) = imfilter(im_object(:,:,i), [1;-1]);
end

%compose matrix A
triplet = zeros(8 * kPix,3);
%internal pixels
e = 1;
t = 1;

disp('Constructing A - stage 1');
for k=1:kInt
    if y_int(k) < h
        if im_mask(y_int(k)+1, x_int(k))
            triplet(t,:) = [e, im2var(y_int(k), x_int(k)), 1];
            triplet(t+1,:) = [e, im2var(y_int(k)+1, x_int(k)), -1];
            B(e,:) = Gy(y_int(k), x_int(k),:);
            t = t+2;
            e = e+1;
        else
            triplet(t,:) = [e, im2var(y_int(k), x_int(k)), 1];
            B(e,:) = Gy(y_int(k), x_int(k),:) + im_target(y_int(k)+1, x_int(k),:);
            t = t+1;
            e = e+1;
        end
    end
    
    if y_int(k) > 1
        if ~im_mask(y_int(k)-1, x_int(k))
            triplet(t,:) = [e, im2var(y_int(k), x_int(k)), 1];
            B(e,:) = -1*Gy(y_int(k)-1, x_int(k),:) + im_target(y_int(k)-1, x_int(k),:);
            t = t+1;
            e = e+1;
        end
    end
    
    if x_int(k) < w
        if im_mask(y_int(k), x_int(k)+1)
            triplet(t,:) = [e, im2var(y_int(k), x_int(k)), 1];
            triplet(t+1,:) = [e, im2var(y_int(k), x_int(k)+1), -1];
            B(e,:) = Gx(y_int(k), x_int(k),:);
            t = t+2;
            e = e+1;
        else
            triplet(t,:) = [e, im2var(y_int(k), x_int(k)), 1];
            B(e,:) = Gx(y_int(k), x_int(k),:) + im_target(y_int(k), x_int(k)+1,:);
            t = t+1;
            e = e+1;
        end
    end
    
    if x_int(k) > 1
        if ~im_mask(y_int(k), x_int(k)-1)
            triplet(t,:) = [e, im2var(y_int(k), x_int(k)), 1];
            B(e,:) = -1*Gx(y_int(k), x_int(k)-1,:) + im_target(y_int(k), x_int(k)-1,:);
            t = t+1;
            e = e+1;
        end
    end
end


triplet2 = triplet(1:t-1, :);
A = sparse(triplet2(:,1), triplet2(:,2), triplet2(:,3), e-1, kPix);
disp('Done');
%process for each channel

disp('Solving...');
R = zeros(kPix, c);
for ch=1:c
    R(:,ch) = lscov(A, B(:,ch));
end

disp('Copying pixel');
for x= max(min(x_int),1): min(max(x_int),w)
    for y= max(min(y_int),1): min(max(y_int),h)
       if im_mask(y,x) > 0
           im_out(y,x,:) = R(im2var(y,x),:);
       end
    end
end

disp('All done');
end