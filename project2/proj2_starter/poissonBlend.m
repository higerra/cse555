function im_out = poissonBlend(im_object, im_mask, im_target)
%possionBlend(im_object, im_mask, im_target)
%perform possion blending
im_out = im_target;
if size(im_object) ~= size(im_target)
    disp('Error! Image sizes must be the same');
    return;
end

[h,w,c] = size(im_target);

%get mask for internal pixels and border pixels
m_internal = imerode(im_mask, strel('diamond',1));
m_border = xor(im_mask, m_internal);

[y_int, x_int] = find(m_internal);
[y_bor, x_bor] = find(m_border);

kInt = size(y_int, 1);
kBor = size(y_bor,1);
kPix = kInt + kBor;

B = zeros(kPix, c);

im2var = zeros(h,w);
v = 1;
for x= max(min(x_bor),1): min(max(x_bor),w)
    for y= max(min(y_bor),1): min(max(y_bor),h)
       if im_mask(y,x) > 0
           im2var(y,x) = v;
           v = v + 1;
       end
    end
end

%compose matrix A
triplet = zeros(8 * kPix,3);
%internal pixels
e = 1;
t = 1;

disp('Constructing A - stage 1');
for k=1:kInt
    triplet(t,:) = [e, im2var(y_int(k), x_int(k)), 1];
    triplet(t+1,:) = [e, im2var(y_int(k), x_int(k)+1), -1];
    triplet(t+2,:) = [e+1, im2var(y_int(k), x_int(k)), 1];
    triplet(t+3,:) = [e+1, im2var(y_int(k)+1, x_int(k)), -1];
    B(e,:) = im_object(y_int(k), x_int(k), :) - im_object(y_int(k), x_int(k)+1, :);
    B(e+1,:) = im_object(y_int(k), x_int(k), :) - im_object(y_int(k)+1, x_int(k), :);
    t = t + 4;
    e = e + 2;
    if m_border(y_int(k)-1, x_int(k)) == 1
        triplet(t,:) = [e, im2var(y_int(k), x_int(k)), 1];
        triplet(t+1,:) = [e, im2var(y_int(k)-1, x_int(k)), -1];
        B(e,:) = im_object(y_int(k), x_int(k), :) - im_object(y_int(k)-1, x_int(k), :);
        t = t+2;
        e = e+1;
    end
    if m_border(y_int(k), x_int(k)-1) == 1
        triplet(t,:) = [e, im2var(y_int(k), x_int(k)), 1];
        triplet(t+1,:) = [e, im2var(y_int(k), x_int(k)-1), -1];
        B(e,:) = im_object(y_int(k), x_int(k), :) - im_object(y_int(k), x_int(k)-1, :);
        t = t+2;
        e = e+1;
    end
end

%border pixels
% for k=1:kBor
%     if y_bor(k) > 1
%         if m_internal(y_bor(k)-1,x_bor(k)) == 0
%             triplet(t,:) = [e, im2var(y_bor(k), x_bor(k)), 1];
%             B(e,:) = im_object(y_bor(k), y_bor(k), :) - im_object(y_bor(k)-1, x_bor(k), :) + im_target(y_bor(k)-1, x_bor(k));
%             e = e+1;
%             t = t+1;
%         end
%     end
%     if y_bor(k) < h
%         if m_internal(y_bor(k)+1,x_bor(k)) == 0
%             triplet(t,:) = [e, im2var(y_bor(k), x_bor(k)), 1];
%             B(e,:) = im_object(y_bor(k), y_bor(k), :) - im_object(y_bor(k)+1, x_bor(k), :) + im_target(y_bor(k)-1, x_bor(k));
%             e = e+1;
%             t = t+1;
%         end
%     end
% end

disp('Constructing A - stage 2');
for k=1:kBor
    lacc = 0;
    racc = zeros(1,1,c);
    whos temp
    if y_bor(k) > 1
        if m_internal(y_bor(k)-1,x_bor(k)) == 0
            lacc = lacc + 1;
            racc = racc + im_object(y_bor(k), x_bor(k), :) - im_object(y_bor(k)-1, x_bor(k), :) + im_target(y_bor(k)-1, x_bor(k));
        end
    end
    if y_bor(k) < h
        if m_internal(y_bor(k)+1,x_bor(k)) == 0
            lacc = lacc + 1;
            racc = racc + im_object(y_bor(k), x_bor(k), :) - im_object(y_bor(k)+1, x_bor(k), :) + im_target(y_bor(k)+1, x_bor(k));
        end
    end
    if x_bor(k) > 1
        if m_internal(y_bor(k),x_bor(k)-1) == 0
            lacc = lacc + 1;
            racc = racc + im_object(y_bor(k), x_bor(k), :) - im_object(y_bor(k), x_bor(k)-1, :) + im_target(y_bor(k), x_bor(k)-1);
        end
    end
    if x_bor(k) < w
        if m_internal(y_bor(k),x_bor(k)+1) == 0
            lacc = lacc + 1;
            racc = racc + im_object(y_bor(k), x_bor(k), :) - im_object(y_bor(k), x_bor(k)+1, :) + im_target(y_bor(k), x_bor(k)+1);
        end
    end
    if lacc > 0
        triplet(t,:) = [e, im2var(y_bor(k), x_bor(k)), lacc];
        B(e,:) = racc;
        t = t + 1;
        e = e + 1;
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
for x= max(min(x_bor),1): min(max(x_bor),w)
    for y= max(min(y_bor),1): min(max(y_bor),h)
       if im_mask(y,x) > 0
           im_out(y,x,:) = R(im2var(y,x),:);
       end
    end
end

disp('All done');
end