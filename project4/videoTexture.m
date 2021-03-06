function videoTexture(filename)
%% videoTexture(filename)
vreader = VideoReader(filename);
w = vreader.Width;
h = vreader.Height;
fps = vreader.FrameRate;
fprintf('Width: %d, height: %d, fps: %d\n', w, h, fps);
maxKFrame = 2000;
kFrame = 0;
tempframes = cell(maxKFrame);
interval = 2;
icount = 0;
for i=1:maxKFrame
    if hasFrame(vreader)
        if mod(icount,interval) == 0
            tempframes{kFrame+1} = im2double(readFrame(vreader));
            kFrame = kFrame+1;
            icount = 0;
        end
        icount = icount+1;
    end
end
kFrame = kFrame - 1;
frames = cell(kFrame);
for i=1:kFrame
    frames{i} = tempframes{i+1};
end
fprintf('%d frames read\n', kFrame);

%compute initial D matrix
disp('Computing initial D matrix...');
D = zeros(kFrame, kFrame);
for i=1:kFrame
    for j=1:kFrame
        D(i,j) = sqrt(sum(sum(sum((frames{i} - frames{j}).^2))));
%         I1 = reshape(frames{i}, 1, []);
%         I2 = reshape(frames{j}, 1, []);
%         D(i,j) = sqrt((I1-I2) * (I1-I2)');
    end
end
disp('Done');

%diagnal filter
disp('Diagnal filtering...');
Df = zeros(size(D));
m = 2;
diaW = [1 2 2 1];
for i=m+1:kFrame-m+1
    for j=m+1:kFrame-m+1
      for k=-m:m-1
          Df(i,j) = Df(i,j) + diaW(k+m+1) * D(i+k,j+k);
      end
      Df(i,j) = Df(i,j) / sum(diaW);
    end
end
D = Df;
disp('Done');

%Q-learning
disp('Q learning');
alpha = 0.995;
constp = 1;
converge_th = 0.001;
Dq = D .^ constp;
iterCount = 0;

while true
    iterCount = iterCount + 1;
    mj = zeros(kFrame+1,1);
    mj(m+1:kFrame-m+1) = min(Dq(m+1:kFrame-m+1,m+1:kFrame-m+1),[],2);
    mj(kFrame-m+2) = max(max(Dq));
    Dq2 = Dq;
    for j=m+1:kFrame-m+1
        Dq(:,j) = D(:,j).^constp + alpha*mj(j+1);
    end
    diff = sqrt(sum(sum((Dq - Dq2).^2)));
    %fprintf('iteration %d, error %.3f\n', iterCount, diff);
    if diff <= converge_th
        break;
    end
end
disp('Done');
D = Dq;

%compute P matrix
aveD = mean(mean(D(m+1:kFrame-m+1, m+1:kFrame-m+1)));

sigma = 0.02 * aveD;
fprintf('aveD:%.3f, sigma:%.3f\n', aveD, sigma);
%sigma = 5;
P = zeros(kFrame, kFrame);
for i=m+1:kFrame-m
    for j=m+1:kFrame-m
        P(i,j) = exp(-1*D(i+1,j)/sigma);
    end
   P(i,:) = P(i,:) / sum(P(i,:));
end
%pruning
%local maxima
P2 = P;
for i=m+1:kFrame-m+1
    for j=m+2:kFrame-m
        if P(i,j) < P(i,j-1) || P(i,j) < P(i,j+1)
            P2(i,j) = 0;
        end
    end
end
P = P2;
%thresholding
P(P<0.05) = 0.0;
%double probability of forward jump, set probability of backward jump to 0
amp = 2;
for i=m+1:kFrame-m+1
    for j=i+2:kFrame-m+1
        P(i,j) = P(i,j) * amp;
    end
    for j=m+1:i
        P(i,j) = 0.0;
    end
end
%re-normalization
for i=m+1:kFrame-m+1
    P(i,:) = P(i,:) / sum(P(i,m+1:kFrame-m));
end
% Dv = D./max(max(D(m+1:kFrame-m+1, m+1:kFrame-m+1)));
 figure(1),
% hold on;
% subplot(1,2,1);
% imshow(Dv);
% title('D matrix');
% subplot(1,2,2);
 Pv = P./max(max(P));
 imshow(Pv);
 title('P Matrix');
% hold off;

%cdf from pdf
cdf = cumsum(P,2);
source = m+1;
kOutput = max(kFrame * 2, 1000);

vwriter = VideoWriter(strcat('result_',filename));
vwriter.FrameRate = fps;
open(vwriter);

% figure(2),
% hold on;
x = rand(kOutput,1);
for i=1:kOutput
    %draw from distribution
    if source >= kFrame - 10
        break;
    end
    target = m+1;
    if x > cdf(source,1)
        for j=m+1:kFrame - m
            if x(i) > cdf(source,j-1) && x(i) <= cdf(source,j)
                target = j;
                break;
            end
        end
    end
    fprintf('Writing frame %d, Jump from frame %d to %d, p: %.3f\n', i, source, target, P(source,target));
    %imshow(frames{target});
    writeVideo(vwriter, frames{target});
    source = target;
    %pause(1.0/(fps+2));
end
% hold off;
close(vwriter);
end
