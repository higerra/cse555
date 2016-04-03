function videoTexture(filename)
%% videoTexture(filename)
vreader = VideoReader(filename);
w = vreader.Width;
h = vreader.Height;
fps = vreader.FrameRate;

maxKFrame = 1000;
kFrame = 0;
tempframes = cell(maxKFrame);
for i=1:maxKFrame
    if hasFrame(vreader)
        fprintf('Reading frame %d\n', i);
        tempframes{i} = im2double(readFrame(vreader));
        kFrame = kFrame+1;
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
        I1 = reshape(frames{i}, 1, []);
        I2 = reshape(frames{j}, 1, []);
        D(i,j) = sqrt((I1-I2) * (I1-I2)');
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
constp = 2;
converge_th = 0.001;
Dq = D .^ constp;
iterCount = 0;
while true
    iterCount = iterCount + 1;
    Dq2 = Dq;
    Dq3 = Dq;
    %Dq3(logical(eye(kFrame))) = max(max(D));
    mj = min(Dq3(m+1:kFrame-m+1,m+1:kFrame-m+1),[],2);
    
    Dq = D.^constp + alpha * mj;
    diff = sqrt(sum(sum((Dq - Dq2).^2)));
    
    fprintf('iteration %d, error %.3f\n', iterCount, diff);
    if diff <= converge_th
        break;
    end
end
disp('Done');
D = Dq;

%compute P matrix
aveD = mean(mean(D(m+1:kFrame-m,m+1:kFrame-m)));
sigma = 0.3 * aveD;
P = zeros(kFrame, kFrame);
for i=m+1:kFrame-m
    for j=m+1:kFrame-m
        P(i,j) = exp(-1*D(i+1,j)/sigma);
    end
   P(i,:) = P(i,:) / sum(P(i,:));
end

Dv = D./max(max(D));
figure(1),
hold on;
subplot(1,2,1);
imshow(Dv);
title('D matrix');
subplot(1,2,2);
Pv = P./max(max(P));
imshow(Pv);
title('P Matrix');
hold off;

%cdf from pdf
cdf = cumsum(P,2);
source = m+1;
kOutput = 1000;

% figure(2),
% hold on;
% x = rand(kOutput,1);
% for i=1:kOutput
%     %draw from distribution
%     target = m+1;
%     if x > cdf(source,1)
%         for j=m+1:kFrame - m
%             if x(i) > cdf(source,j-1) && x(i) <= cdf(source,j)
%                 target = j;
%                 break;
%             end
%         end
%     end
%     fprintf('Jump from frame %d to %d, p: %.3f\n', source, target, P(source,target));
%     imshow(frames{target});
%     source = target;
%     pause(1.0/(fps+2));
% end
% hold off;
end