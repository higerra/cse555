% starter script for project 3
DO_TOY = false;
DO_BLEND = true;
DO_MIXED  = false;
DO_COLOR2GRAY = false;

if DO_TOY 
    toyim = im2double(imread('./samples/toy_problem.png')); 
    % im_out should be approximately the same as toyim
    im_out = toy_reconstruct(toyim);   % you need to write this function, toy_reconstruct
    disp(['Error: ' num2str(sqrt(sum((toyim(:)-im_out(:)).^2)))])
    figure
    subplot(1,2,1);
    imshow(toyim);
    subplot(1,2,2);
    imshow(im_out);
end

if DO_BLEND
    % do a small one first, while debugging
    %im_background = imresize(im2double(imread('./samples/im2.jpg')), 0.25, 'bilinear');
    %im_object = imresize(im2double(imread('./samples/penguin-chick.jpeg')), 0.25, 'bilinear');
    im_background = im2double(imread('./samples/im2.jpg'));
    im_object = im2double(imread('./samples/penguin-chick.jpeg'));
    
    % get source region mask from the user
    objmask = getMask(im_object);
    % align im_s and mask_s with im_background
    [im_s, mask_s] = alignSource(im_object, objmask, im_background);
    % blend
    disp('Possion blending...');
    tic;
    im_blend = poissonBlend(im_s, mask_s, im_background);   % you need to write this.
    toc;
    disp('Mixed gradient blending');
    tic;
    im_mixblend = mixedBlend(im_s, mask_s, im_background);     % you need to write this.
    toc;
    imwrite(im_blend, 'possionBlend.jpg');
    imwrite(im_mixblend, 'mixedBlend.jpg');
    figure(3), hold off, imshow(im_blend)
end

if DO_MIXED
    % read images
    %...
    im_background = im2double(imread('./samples/im2.jpg'));
    im_object = im2double(imread('./samples/penguin-chick.jpeg'));
    
    % get source region mask from the user
    objmask = getMask(im_object);
    % align im_s and mask_s with im_background
    [im_s, mask_s] = alignSource(im_object, objmask, im_background);
    % blend
    tic;
    % blend
    im_blend = mixedBlend(im_s, mask_s, im_background);     % you need to write this.
    toc;
    imwrite(im_blend, 'mixedBlend.jpg');
    figure(3), hold off, imshow(im_blend);
end

if DO_COLOR2GRAY
    % also feel welcome to try this on some natural images and compare to rgb2gray
    im_rgb = im2double(imread('./samples/colorBlindTest35.png'));
    im_gr = color2gray(im_rgb);
    figure(4), hold off, imagesc(im_gr), axis image, colormap gray
end
